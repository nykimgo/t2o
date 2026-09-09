#!/usr/bin/env python3
"""
GLB 파일을 USD로 변환하고, 원본 USD(object_n.usda)에 직접 geometry reference를
주입하는 스크립트 (pxr.Usd 기반 버전).

이 스크립트는 생성된 3D 결과(glb)를 USD geometry로 변환하여 원본 USD 옆의
assets 폴더(예: scene_n/objects/assets/{object_name})에 저장하고,
원본 USD(object_n.usda)의 defaultPrim 하위 "geometry" 프림에 해당 geometry USD에
대한 reference를 직접 추가/갱신하여 저장합니다.

이렇게 하면 개별 오브젝트(object_n.usda)뿐 아니라, 이를 참조하는 shot/scene/root USD를
3D 툴(Blender, USD Viewer 등)에서 열었을 때도 생성된 에셋이 즉시 보입니다.

사용법:
    python merge_glb_to_usd.py <glb_file> <original_usd_file> [--output-usd <output_usd>]

예시:
    python merge_glb_to_usd.py \
        /root/.../movie_usds/hidden_time_260615/scene_1/objects/assets/object_1/scene_1_shot_1_NA_73259.glb \
        /root/.../movie_usds/hidden_time_260615/scene_1/objects/object_1.usda

출력:
    - assets/{object_name}/{object_name}.geometry.usda: GLB에서 변환된 geometry USD (+ bin/ 텍스처)
    - 원본 object_n.usda: defaultPrim 하위 "geometry" 프림에 geometry reference가 직접 주입됨
    - 원본 object_n.usda customData: source_glb, geometry_usd 경로 기록
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

try:
    from pxr import Usd, UsdShade, Sdf
except ImportError:
    print("❌ pxr(USD Python 바인딩)를 import할 수 없습니다. USD가 설치되어 있는지 확인하세요.")
    raise


# ---------------------------------------------------------------------------
# 기본 정보 추출/경로 유틸
# ---------------------------------------------------------------------------

def extract_object_name_from_usd(usd_file: Path) -> Optional[str]:
    """
    USD 파일에서 대표 object_name을 추출합니다.

    우선 defaultPrim 이름을 사용하고,
    없으면 첫 번째 Xform prim의 이름을 사용합니다.
    """
    try:
        stage = Usd.Stage.Open(str(usd_file))
        if not stage:
            print(f"⚠️ USD Stage를 열 수 없습니다: {usd_file}")
            return None

        default_prim = stage.GetDefaultPrim()
        if default_prim:
            return default_prim.GetName()

        # fallback: 첫 번째 Xform prim
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Xform":
                return prim.GetName()

    except Exception as e:
        print(f"⚠️ USD 파일 읽기 실패: {e}")
    return None


_SHOT_SEGMENT_RE = re.compile(r'^shot_\w+$')


def resolve_scene_canonical_object_usd(original_usd: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    merge 대상 USD가 scene canonical(`.../scene_n/objects/object_N.usda`)인지 검증합니다.

    스펙 2.1 / 체크리스트: object 파이프라인은 scene canonical만 수정해야 하며,
    shot override(`.../shot_m/objects/object_N.usda`)가 들어오면 거부하고
    scene canonical 경로로 resolve를 시도합니다.

    Returns:
        (resolved_path, reason)
        - resolved_path: 사용할 scene canonical 경로 (검증/resolve 성공 시). 실패 시 None.
        - reason: None(이미 scene canonical) / "resolved_from_shot"(shot→scene 변환됨)
                  / 오류 메시지 문자열(거부 사유).
    """
    usd = original_usd.resolve()
    parent = usd.parent
    # 1) 부모 디렉토리가 "objects"여야 함
    if parent.name != "objects":
        return None, f"object 파이프라인은 'objects/' 하위 USD만 수정할 수 있습니다: {usd}"

    grandparent = parent.parent  # objects의 상위 (scene_n 또는 shot_m)
    # 2) shot override 경로면 scene canonical로 resolve 시도
    if _SHOT_SEGMENT_RE.match(grandparent.name):
        # .../scene_n/shot_m/objects/object_N.usda -> .../scene_n/objects/object_N.usda
        scene_dir = grandparent.parent  # scene_n
        canonical = scene_dir / "objects" / usd.name
        if canonical.exists():
            return canonical, "resolved_from_shot"
        return None, (
            f"shot override 경로가 입력되었고 scene canonical을 찾을 수 없습니다: {usd}\n"
            f"   기대 경로: {canonical}"
        )

    # 3) 정상 scene canonical
    return usd, None


def extract_object_info_from_glb_path(glb_path: Path) -> Optional[dict]:
    """
    GLB 파일 경로에서 scene, shot, object_name 정보를 추출합니다.

    예: .../scene_1/shot_1/object_1/scene_1_shot_1_NA_73259.glb
    -> {'scene': 'scene_1', 'shot': 'shot_1', 'object_name': 'object_1'}
    """
    parts = glb_path.parts
    try:
        scene_idx = next(i for i, p in enumerate(parts) if p.startswith('scene_'))
        if scene_idx < len(parts) - 1:
            scene = parts[scene_idx]
            shot = parts[scene_idx + 1] if scene_idx + 1 < len(parts) else None
            object_name = parts[scene_idx + 2] if scene_idx + 2 < len(parts) else None
            return {
                'scene': scene,
                'shot': shot,
                'object_name': object_name
            }
    except (StopIteration, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# 텍스처 파일명 정리 + geometry USD 후처리 (pxr 기반)
# ---------------------------------------------------------------------------

_TEXTURE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
_IMAGE_INDEX_RE = re.compile(r'^image(\d+)', re.IGNORECASE)


def _is_texture_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXTURE_EXTENSIONS


def _renamed_texture_pattern(object_name: str) -> re.Pattern[str]:
    return re.compile(
        rf'^texture_{re.escape(object_name)}_\d+\.',
        re.IGNORECASE,
    )


def _source_texture_index(stem: str) -> Optional[int]:
    match = _IMAGE_INDEX_RE.match(stem)
    return int(match.group(1)) if match else None


def rename_texture_files(usd_file: Path, object_name: str) -> Dict[str, str]:
    """
    변환기가 bin/에 생성한 원본 텍스처(image0.jpg 등)만
    texture_{object_name}_{N}.jpg 로 정규화합니다. (네이티브 변환기는 이미
    이 이름으로 만들어 매칭되지 않으므로 아무 것도 바꾸지 않는다 — 무해.)

    object마다 독립된 assets/.../bin/ 을 쓰므로, 같은 scene 내
    object 간 충돌은 없습니다. imageN → _N 매핑 후 동일 경로는 덮어씁니다.
    이미 정규화된 texture_{object_name}_* 파일은 소스에서 제외해 재실행 시
    인덱스가 밀리는 문제를 방지합니다.

    Returns:
        {old_name: new_name} 매핑 딕셔너리
    """
    bin_dir = usd_file.parent / "bin"
    if not bin_dir.exists():
        return {}

    renamed_pattern = _renamed_texture_pattern(object_name)
    all_textures = [p for p in bin_dir.iterdir() if p.is_file() and _is_texture_file(p)]
    source_files = [p for p in all_textures if not renamed_pattern.match(p.name)]
    if not source_files:
        return {}

    def _sort_key(path: Path) -> tuple:
        image_idx = _source_texture_index(path.stem)
        if image_idx is not None:
            return (0, image_idx, path.name.lower())
        return (1, 0, path.name.lower())

    name_mapping: Dict[str, str] = {}
    target_names: set[str] = set()
    fallback_idx = 0
    used_indices: set[int] = set()

    for old_file in sorted(source_files, key=_sort_key):
        ext = old_file.suffix.lower()
        image_idx = _source_texture_index(old_file.stem)
        if image_idx is not None:
            idx = image_idx
        else:
            while fallback_idx in used_indices:
                fallback_idx += 1
            idx = fallback_idx
            fallback_idx += 1

        used_indices.add(idx)
        new_name = f"texture_{object_name}_{idx}{ext}"
        new_file = bin_dir / new_name
        target_names.add(new_name)

        if old_file.resolve() != new_file.resolve():
            if new_file.exists():
                new_file.unlink()
            old_file.replace(new_file)
            print(f"   텍스처 파일명 변경: {old_file.name} → {new_name}")
        else:
            print(f"   텍스처 파일명 유지: {new_name}")

        name_mapping[old_file.name] = new_name

    for stale_file in bin_dir.iterdir():
        if not stale_file.is_file() or not _is_texture_file(stale_file):
            continue
        if renamed_pattern.match(stale_file.name) and stale_file.name not in target_names:
            stale_file.unlink()
            print(f"   stale 텍스처 삭제: {stale_file.name}")

    return name_mapping


def postprocess_geometry_usd(usd_file: Path, object_name: Optional[str] = None) -> None:
    """
    변환된 geometry USD에 대해 후처리 (pxr 기반):
      1) 텍스처 파일명을 object_name을 포함한 고유한 이름으로 변경 (bin/ 폴더)
      2) UsdPrimvarReader_float2 의 inputs:varname 'st0' → 'st'
      3) 텍스처 경로를 './bin/파일명' 또는 './파일명' 형태의 상대 경로로 통일

    → Blender / DCC 툴에서 reference로 열렸을 때도
      UV/텍스처가 안정적으로 동작하도록 하기 위한 패치.
    """
    try:
        texture_mapping: Dict[str, str] = {}
        if object_name:
            texture_mapping = rename_texture_files(usd_file, object_name)

        stage = Usd.Stage.Open(str(usd_file))
        if not stage:
            print(f"⚠️ geometry USD Stage를 열 수 없습니다: {usd_file}")
            return

        modified = False

        # -------------------------------------------------------------------
        # 1) UV PrimvarReader: varname 'st0' -> 'st'
        # -------------------------------------------------------------------
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Shader":
                shader = UsdShade.Shader(prim)
                shader_id_attr = shader.GetIdAttr()
                shader_id = shader_id_attr.Get() if shader_id_attr else None
                if shader_id == "UsdPrimvarReader_float2":
                    var_input = shader.GetInput("varname")
                    if var_input:
                        val = var_input.Get()
                        if val == "st0":
                            var_input.Set("st")
                            modified = True

        # -------------------------------------------------------------------
        # 2) 텍스처 경로를 './bin/파일명' 형태로 통일 (및 이름 매핑 적용)
        # -------------------------------------------------------------------

        def _normalize_tex_path(path: str) -> str:
            """
            기존 정규식 기반 구현과 동일한 로직을 pxr AssetPath 기반으로 구현:
              - 'bin/' 포함 시 bin/ 이후만 사용, 파일명은 매핑 적용
              - bin/ 없으면 파일명만 사용
            """
            if not path:
                return path

            path_str = path.replace("\\", "/")

            if "bin/" in path_str:
                idx = path_str.rfind("bin/")
                rel_part = path_str[idx:]  # 'bin/xxx.png'
                old_filename = os.path.basename(rel_part)
                if old_filename in texture_mapping:
                    rel_part = f"bin/{texture_mapping[old_filename]}"
            else:
                old_filename = os.path.basename(path_str)
                if old_filename in texture_mapping:
                    rel_part = f"bin/{texture_mapping[old_filename]}"
                else:
                    rel_part = os.path.basename(path_str)

            return f"./{rel_part}"

        for prim in stage.Traverse():
            if prim.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(prim)
            for inp in shader.GetInputs():
                attr = inp.GetAttr()
                type_name = attr.GetTypeName()
                if type_name == Sdf.ValueTypeNames.Asset:
                    old_asset = inp.Get()
                    if isinstance(old_asset, Sdf.AssetPath):
                        old_path = old_asset.path
                    else:
                        old_path = str(old_asset) if old_asset is not None else ""

                    if not old_path:
                        continue

                    new_path = _normalize_tex_path(old_path)
                    if new_path and new_path != old_path:
                        inp.Set(Sdf.AssetPath(new_path))
                        modified = True

        if modified:
            stage.GetRootLayer().Save()
            print(f"🛠 geometry USD 후처리 적용(pxr): {usd_file.name}")
        else:
            print(f"ℹ️ geometry USD 후처리 불필요(pxr): {usd_file.name}")

    except Exception as e:
        print(f"⚠️ geometry USD 후처리 실패 ({usd_file}): {e}")


# ---------------------------------------------------------------------------
# GLB → USD 변환 (네이티브 trimesh + pxr, glb_to_usd_native 모듈 위임)
# ---------------------------------------------------------------------------

def convert_glb_to_usd(
    glb_file: Path,
    output_usd: Path,
    root_prim_name: Optional[str] = None,
    object_name: Optional[str] = None
) -> bool:
    """
    GLB 파일을 USD로 변환합니다.

    Args:
        glb_file: 입력 GLB 파일 경로
        output_usd: 출력 USD 파일 경로
        root_prim_name: 루트 프리미티브 이름 (None이면 object_name 사용)
        object_name: object 이름 (텍스처 파일명에 사용)
    """
    try:
        print(f"🔄 GLB → USD 변환 중 (trimesh + pxr, 네이티브)...")
        print(f"   입력: {glb_file}")
        print(f"   출력: {output_usd}")

        # usd_from_gltf(빌드 지옥·업데이트 종료) 대신 순수 파이썬 변환기를 쓴다.
        # TRELLIS.2 출력은 항상 단일 메시 + PBR 텍스처 형태라 이걸로 충분하며,
        # 출력 USD 계층을 usd_from_gltf 와 구조적으로 동일하게 맞춰 뒀다.
        # 추가 의존성 없음(trimesh/pxr 은 trellis2 env 에 이미 있음).
        from glb_to_usd_native import convert as _native_convert

        output_usd.parent.mkdir(parents=True, exist_ok=True)
        _, actual_root = _native_convert(
            str(glb_file), str(output_usd),
            root_prim_name=(root_prim_name or object_name or "object"),
            texture_object_name=object_name,
        )
        print(f"✅ 변환 완료: {output_usd}  (root=/{actual_root})")

        # geometry USD 후처리 (UV/텍스처 경로 정규화).
        # 네이티브 변환기는 이미 './bin/...' 형태로 만들지만, 일관성을 위해
        # 그대로 태운다(이미 정규화돼 있으면 '불필요'로 넘어간다).
        postprocess_geometry_usd(output_usd, object_name)

        return True

    except Exception as e:
        import traceback
        print(f"❌ GLB → USD 변환 실패: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 원본 USD에 geometry reference 추가 (pxr 기반)
# ---------------------------------------------------------------------------

def _relative_asset_path(from_dir: Path, target: Path) -> str:
    """USD customData/reference용 상대 asset 경로를 계산합니다."""
    try:
        rel_path = os.path.relpath(str(target.resolve()), str(from_dir.resolve()))
        rel_path = rel_path.replace("\\", "/")
        if not rel_path.startswith((".", "/")):
            rel_path = f"./{rel_path}"
        return rel_path
    except ValueError:
        return str(target.resolve())


def _write_generated_asset_metadata(
    root_prim,
    original_dir: Path,
    geometry_usd: Path,
    glb_file: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> None:
    """object_n.usda customData에 생성 에셋 경로(source_glb, geometry_usd)를 기록합니다."""
    geometry_rel = _relative_asset_path(original_dir, geometry_usd)
    root_prim.SetCustomDataByKey('geometry_usd', Sdf.AssetPath(geometry_rel))
    if glb_file is not None:
        glb_rel = _relative_asset_path(original_dir, glb_file)
        root_prim.SetCustomDataByKey('source_glb', Sdf.AssetPath(glb_rel))
    if run_id:
        root_prim.SetCustomDataByKey('t2o_run_id', run_id)


def get_root_prim_name_from_usd(usd_file: Path) -> Optional[str]:
    """
    USD 파일에서 defaultPrim 이름을 추출합니다. (pxr 기반)
    """
    try:
        stage = Usd.Stage.Open(str(usd_file))
        if not stage:
            return None
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            return default_prim.GetName()
    except Exception:
        pass
    return None


def inject_geometry_reference_into_original(
    original_usd: Path,
    geometry_usd: Path,
    geometry_prim_name: str,
    glb_file: Optional[Path] = None,
) -> bool:
    """
    원본 USD(object_n.usda)를 직접 수정하여 geometry USD에 대한 reference를 주입합니다.

    - original_usd의 defaultPrim 하위에 "geometry" Xform 프림을 정의(또는 갱신)
    - 그 프림에 geometry USD 파일에 대한 reference를 (상대 경로로) 설정
    - defaultPrim customData에 source_glb / geometry_usd 경로를 기록
    - 기존 geometry reference가 있으면 모두 제거 후 새 reference로 교체
    - 원본 파일에 직접 저장(Save)

    이렇게 하면 object_n.usda를 단독으로 열거나, 이를 참조하는 shot/scene/root USD를
    열었을 때도 생성된 에셋이 보입니다.

    Returns:
        성공 시 True, 실패 시 False
    """
    try:
        stage = Usd.Stage.Open(str(original_usd))
        if not stage:
            print(f"❌ 원본 USD Stage를 열 수 없습니다: {original_usd}")
            return False

        root_prim = stage.GetDefaultPrim()
        if not root_prim:
            # defaultPrim이 없으면 첫 번째 Xform을 사용
            for prim in stage.Traverse():
                if prim.GetTypeName() == "Xform":
                    root_prim = prim
                    break
        if not root_prim:
            print(f"❌ 원본 USD에서 루트 prim(defaultPrim/Xform)을 찾을 수 없습니다: {original_usd}")
            return False

        root_prim_path = root_prim.GetPath()

        original_dir = original_usd.parent.resolve()
        ref_asset = _relative_asset_path(original_dir, geometry_usd)

        # defaultPrim 하위에 "geometry" Xform 프림 정의
        geometry_prim_path = root_prim_path.AppendChild("geometry")
        geom_prim = stage.DefinePrim(geometry_prim_path, "Xform")
        if not geom_prim:
            print(f"❌ geometry 프림 정의 실패: {geometry_prim_path}")
            return False

        # 기존 reference 제거 후 새 reference 설정 (중복/구버전 경로 방지)
        refs = geom_prim.GetReferences()
        refs.ClearReferences()
        refs.AddReference(ref_asset, f"/{geometry_prim_name}")

        _write_generated_asset_metadata(
            root_prim, original_dir, geometry_usd, glb_file,
            run_id=os.environ.get('RUN_ID'),
        )

        # 원본 파일에 직접 저장
        stage.GetRootLayer().Save()

        print(f"✅ 원본 USD에 geometry reference 주입 완료:")
        print(f"   대상 파일: {original_usd}")
        print(f"   프림 경로: {geometry_prim_path}")
        print(f"   Reference: @{ref_asset}@</{geometry_prim_name}>")
        if glb_file is not None:
            print(f"   customData.source_glb: @{_relative_asset_path(original_dir, glb_file)}@")
        print(f"   customData.geometry_usd: @{ref_asset}@")
        return True

    except Exception as e:
        print(f"❌ 원본 USD에 geometry reference 주입 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GLB 파일을 USD로 변환하고 원본 USD에 geometry reference를 직접 주입 (pxr.Usd 기반)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용 (GLB가 위치한 assets 폴더에 geometry USD 생성 + 원본 USD 직접 수정)
  python merge_glb_to_usd.py \\
      /root/.../scene_1/objects/assets/object_1/scene_1_shot_1_NA_73259.glb \\
      /root/.../scene_1/objects/object_1.usda

  # 출력 geometry USD 파일 경로 지정
  python merge_glb_to_usd.py \\
      /root/.../assets/object_1/scene_1_shot_1_NA_73259.glb \\
      /root/.../scene_1/objects/object_1.usda \\
      --output-usd /custom/path/object_1.geometry.usda

  # USD 변환만 수행 (원본 USD 수정 안 함)
  python merge_glb_to_usd.py \\
      /root/.../assets/object_1/scene_1_shot_1_NA_73259.glb \\
      /root/.../scene_1/objects/object_1.usda \\
      --no-merge

비고:
  - 원본 USD(object_n.usda)의 defaultPrim 하위 "geometry" 프림에 reference가 직접 주입됩니다
  - geometry USD와 텍스처(bin/)는 GLB가 위치한 assets 폴더에 저장됩니다
  - object_n.usda 또는 이를 참조하는 shot/scene/root USD를 열면 에셋이 함께 로드됩니다
        """
    )

    parser.add_argument('glb_file', type=Path, help='입력 GLB 파일 경로')
    parser.add_argument('original_usd', type=Path, help='원본 USD 파일 경로')
    parser.add_argument(
        '--output-usd',
        type=Path,
        help='출력 geometry USD 파일 경로 (기본: GLB가 위치한 assets 폴더에 {object_name}.geometry.usda)'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='원본 USD를 수정하지 않고 GLB→USD 변환만 수행'
    )

    args = parser.parse_args()

    # 파일 존재 확인
    if not args.glb_file.exists():
        print(f"❌ GLB 파일을 찾을 수 없습니다: {args.glb_file}")
        sys.exit(1)

    if not args.original_usd.exists():
        print(f"❌ 원본 USD 파일을 찾을 수 없습니다: {args.original_usd}")
        sys.exit(1)

    # merge 대상 경로 검증 (#7): scene canonical만 허용, shot override는 거부/resolve
    # --no-merge(변환만)일 때는 원본 USD를 수정하지 않으므로 검증을 건너뜁니다.
    if not args.no_merge:
        canonical_usd, reason = resolve_scene_canonical_object_usd(args.original_usd)
        if canonical_usd is None:
            print(f"❌ merge 대상이 scene canonical object USD가 아닙니다. 주입을 거부합니다.")
            print(f"   {reason}")
            print(f"   허용 경로 형식: .../scene_n/objects/object_N.usda")
            sys.exit(1)
        if reason == "resolved_from_shot":
            print(f"⚠️ shot override 경로가 입력되어 scene canonical로 resolve했습니다.")
            print(f"   입력: {args.original_usd}")
            print(f"   대상: {canonical_usd}")
        args.original_usd = canonical_usd

    # object_name 추출 (defaultPrim 우선)
    object_name = extract_object_name_from_usd(args.original_usd)
    if not object_name:
        print(f"⚠️ 원본 USD에서 object_name을 추출할 수 없습니다.")
        print(f"   GLB 경로에서 추출 시도...")
        glb_info = extract_object_info_from_glb_path(args.glb_file)
        if glb_info and glb_info.get('object_name'):
            object_name = glb_info['object_name']
            print(f"   추출된 object_name: {object_name}")
        else:
            print(f"❌ object_name을 추출할 수 없습니다. --output-usd로 직접 지정하세요.")
            sys.exit(1)

    print(f"📌 object_name: {object_name}")

    # 출력 USD 파일 경로 결정
    # 기본값: GLB가 위치한 폴더(= 원본 USD 옆 assets/{object_name})에 geometry USD를 생성합니다.
    # 이렇게 하면 결과물(glb/usd/텍스처)이 모두 한 폴더에 모입니다.
    if args.output_usd:
        output_usd = args.output_usd
    else:
        output_usd = args.glb_file.parent / f"{object_name}.geometry.usda"

    # geometry 루트 프리미티브 이름
    geometry_prim_name = f"{object_name}_geometry"

    # 1. GLB → USD 변환 (+ 루트 프림 이름, 텍스처/UV 후처리)
    if not convert_glb_to_usd(args.glb_file, output_usd, geometry_prim_name, object_name):
        sys.exit(1)

    # 2. 원본 USD(object_n.usda)에 geometry reference 직접 주입
    injected = False
    if not args.no_merge:
        print(f"\n🔄 원본 USD에 geometry reference 주입 중...")
        actual_prim_name = get_root_prim_name_from_usd(output_usd)
        if actual_prim_name:
            injected = inject_geometry_reference_into_original(
                args.original_usd, output_usd, actual_prim_name, glb_file=args.glb_file
            )
            if not injected:
                sys.exit(1)
        else:
            print(f"⚠️ 변환된 USD에서 루트 프리미티브를 찾을 수 없습니다.")
            print(f"   수동으로 reference를 추가해야 할 수 있습니다.")

    print(f"\n✅ 완료!")
    print(f"   변환된 geometry USD: {output_usd}")
    if not args.no_merge and injected:
        print(f"   원본 USD에 reference 주입됨: {args.original_usd}")
        print(f"\n💡 이제 object_n.usda 또는 이를 참조하는 shot/scene/root USD를 열면")
        print(f"   생성된 에셋이 함께 로드됩니다.")


if __name__ == '__main__':
    main()
