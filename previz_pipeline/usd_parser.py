"""
USD 파일 파싱 유틸리티

USD 파일에서 객체(object) 정보를 추출하여 JSON 형식으로 반환합니다.
(actor/character 파싱 코드는 보존하되 ENABLE_ACTOR_PARSING=False로 비활성화됨)
"""

# Object-only scope: actor/character 파싱 비활성화 (True로 변경 시 actor 파싱 재활성화)
ENABLE_ACTOR_PARSING = False

import json
import os
import re
from typing import Any, List, Optional, Dict

from bilingual import coerce_bilingual, flatten_bilingual_fields, is_blank

# USD Python 바인딩 찾기 및 로드
USD_AVAILABLE = False
Tf = None
try:
    from pxr import Usd, Sdf, Tf
    USD_AVAILABLE = True
except ImportError:
    # USD Python 바인딩을 찾아서 sys.path에 추가 시도
    import sys
    
    # 일반적인 USD 설치 경로들
    possible_paths = [
        "/usr/local/lib/python",
        "/opt/USD/lib/python",
        "/usr/lib/python",
        os.path.expanduser("~/USD/lib/python"),
        os.path.expanduser("~/.local/lib/python"),
    ]
    
    # 환경 변수에서 USD 경로 확인
    usd_root = os.environ.get("USD_ROOT") or os.environ.get("OPENUSD_ROOT")
    if usd_root:
        possible_paths.insert(0, os.path.join(usd_root, "lib", "python"))
    
    # Python 버전별 경로 추가
    import sysconfig
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for base_path in possible_paths[:]:  # 복사본 사용
        versioned_path = os.path.join(base_path, python_version, "site-packages")
        if os.path.exists(versioned_path):
            possible_paths.append(versioned_path)
        # site-packages 직접 경로
        site_packages_path = os.path.join(base_path, "site-packages")
        if os.path.exists(site_packages_path):
            possible_paths.append(site_packages_path)
    
    # 경로 찾기 및 추가
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            # pxr 모듈이 있는지 확인
            pxr_path = os.path.join(path, "pxr")
            has_pxr = False
            if os.path.exists(pxr_path):
                has_pxr = True
            else:
                try:
                    if any(
                        os.path.exists(os.path.join(path, f)) and "pxr" in f.lower()
                        for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
                    ):
                        has_pxr = True
                except (OSError, PermissionError):
                    pass
            
            if has_pxr:
                sys.path.insert(0, path)
                try:
                    from pxr import Usd, Sdf, Tf
                    USD_AVAILABLE = True
                    break
                except ImportError:
                    continue
    
    # 마지막 시도: 현재 디렉토리와 상위 디렉토리에서 찾기
    if not USD_AVAILABLE:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for search_dir in [current_dir, os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir))]:
            for root, dirs, files in os.walk(search_dir):
                if "pxr" in dirs:
                    pxr_parent = root
                    if pxr_parent not in sys.path:
                        sys.path.insert(0, pxr_parent)
                        try:
                            from pxr import Usd, Sdf, Tf
                            USD_AVAILABLE = True
                            break
                        except ImportError:
                            continue
                if USD_AVAILABLE:
                    break
            if USD_AVAILABLE:
                break


def _derive_next_base(prim_path_str: str, base_path: Optional[List[str]], child_name: str) -> List[str]:
    """
    object/actor의 논리 경로(예: scene_1/object_1, scene_1/shot_1/object_1)를 구성합니다.

    USD는 reference를 합성(compose)하므로 root/scene USD를 열면 하위 prim들이
    전체 prim 경로(예: /hidden_time/scenes/scene_1/objects/object_1)로 노출됩니다.
    이때 prim 경로에서 scene_*/shot_* 패턴의 계층 마커만 추출하여 child_name을 붙입니다.
    이는 디스크 폴더 이름('scenes' 컨테이너명, 루트 폴더명 등)이나
    objects가 scene 직속/shot 하위인지에 무관하게 동작합니다.

    scene_*/shot_* 마커를 찾지 못하면, 참조 그래프를 따라 누적된 base_path에
    child_name을 덧붙이는 방식으로 폴백합니다.
    """
    segments = [s for s in prim_path_str.split('/') if s]
    markers = [s for s in segments if re.match(r'^(scene|shot)_\w+$', s)]
    if markers:
        result = markers + [child_name]
        # 마지막 마커가 child_name과 동일하면 중복 제거
        if len(result) >= 2 and result[-1] == result[-2]:
            result = result[:-1]
        return result

    base = list(base_path) if base_path else []
    if base and base[-1] == child_name:
        return base
    return base + [child_name]


# description 후보에서 제외할 USD 스키마 노이즈 필드 (예: pxr가 자동 생성하는 문서 주석)
_DESCRIPTION_NOISE_FIELDS = ("userDocBrief",)

# 업스트림(의도분석) 리깅 분류 필드. object customData 최상위에
# `string rig_type` = biped|quadruped|bird|insect|static object 로 들어온다.
# (정본 = 의도분석 모듈 실제 산출 형태, 2026-07-28 확정. 구 `etc.rig_type` 중첩 dict 형태도 계속 읽는다.)
_VALID_RIG_TYPES = {"biped", "quadruped", "bird", "insect", "static_object"}


def _normalize_rig_type(value) -> str:
    """rig_type 값 정규화: 'static object'/'Static-Object' → 'static_object'. 미지값은 ''."""
    if not value:
        return ""
    v = "_".join(str(value).strip().lower().replace("-", "_").replace(" ", "_").split("_"))
    if v in {"static", "staticobject", "inanimate", "prop", "object"}:
        v = "static_object"
    return v if v in _VALID_RIG_TYPES else ""


def _resolve_rig_type(custom_data: Dict, prim=None) -> str:
    """object 의 rig_type 을 여러 인코딩에서 견고하게 찾는다.
    ① customData 최상위 `rig_type` (정본 — 의도분석 모듈 산출 형태)
    ② customData 중첩 dict `etc.rig_type` (구 형태, 하위호환)
    ③ (pxr) 자식 prim `etc` 의 customData/attribute `rig_type`."""
    if isinstance(custom_data, dict):
        rt = _normalize_rig_type(custom_data.get("rig_type"))
        if rt:
            return rt
        etc = custom_data.get("etc")
        if isinstance(etc, dict):
            rt = _normalize_rig_type(etc.get("rig_type"))
            if rt:
                return rt
    if prim is not None:
        try:
            for child in prim.GetChildren():
                if child.GetName() != "etc":
                    continue
                cd = child.GetCustomData() or {}
                rt = _normalize_rig_type(cd.get("rig_type"))
                if rt:
                    return rt
                attr = child.GetAttribute("rig_type")
                if attr and attr.IsValid():
                    rt = _normalize_rig_type(attr.Get())
                    if rt:
                        return rt
        except Exception:
            pass
    return ""


def _resolve_description(custom_data: Dict) -> Dict[str, str]:
    """
    customData에서 description을 {ko, en} 형태로 추출합니다.

    우선순위: description -> base_description -> appearance(shot override fallback)
    - 신규 USD 스키마는 base_description을, 구 스키마는 description을 사용합니다.
    - shot override는 base_description이 없을 수 있으므로 appearance를 마지막 폴백으로 사용합니다.
    - userDocBrief 등 USD 스키마 노이즈 필드는 description 후보에서 제외합니다.
    """
    merged = {"ko": "", "en": ""}
    for field in ("description", "base_description", "appearance"):
        if field in _DESCRIPTION_NOISE_FIELDS:
            continue
        value = custom_data.get(field, "")
        if value is None:
            continue
        candidate = coerce_bilingual(value)
        for lang in ("ko", "en"):
            if not merged[lang].strip() and candidate.get(lang, "").strip():
                merged[lang] = candidate[lang]
        if merged["ko"].strip() and merged["en"].strip():
            break
    return merged


def _resolve_bilingual_field(custom_data: Dict, field_name: str) -> Dict[str, str]:
    """customData의 단일 필드를 {ko, en}으로 정규화합니다."""
    value = custom_data.get(field_name, "")
    if value is None:
        return {"ko": "", "en": ""}
    return coerce_bilingual(value)


def _attach_bilingual_object_fields(result: Dict[str, Any], custom_data: Dict) -> None:
    """object 결과 dict에 bilingual 메타 필드를 기록합니다."""
    description = _resolve_description(custom_data)
    flatten_bilingual_fields(result, "description", description)

    for field_name in ("name", "appearance", "location", "action"):
        bilingual = _resolve_bilingual_field(custom_data, field_name)
        if not is_blank(bilingual):
            flatten_bilingual_fields(result, field_name, bilingual)


def _determine_object_scope(object_path: Optional[str], usd_file_path: Optional[str] = None) -> str:
    """
    object 항목이 scene canonical인지 shot override인지 판별합니다.

    경로(object_path 또는 usd_file_path)에 `shot_*` 세그먼트가 있으면 "shot",
    없으면 "scene"으로 간주합니다. (스펙 2.1 JSON 필터 기준)
    """
    for candidate in (object_path, usd_file_path):
        if not candidate:
            continue
        segments = [s for s in str(candidate).replace('\\', '/').split('/') if s]
        if any(re.match(r'^shot_\w+$', s) for s in segments):
            return "shot"
    return "scene"


def _suppress_usd_warnings():
    """
    USD 경고를 억제하는 컨텍스트 매니저.
    """
    class USDWarningSuppressor:
        def __init__(self):
            self.original_handler = None
            self.original_output_file = None
            
        def __enter__(self):
            if USD_AVAILABLE and Tf is not None:
                try:
                    # 경고 핸들러를 빈 함수로 설정하여 모든 경고 억제
                    self.original_handler = Tf.Warning.GetWarningHandler()
                    # 빈 핸들러 함수: 아무것도 하지 않음
                    def empty_handler(msg):
                        pass
                    Tf.Warning.SetWarningHandler(empty_handler)
                except (AttributeError, TypeError):
                    # Tf.Warning이 없는 경우 무시
                    pass
                try:
                    # 출력 파일을 None으로 설정 (경고 출력 억제)
                    self.original_output_file = Tf.Warning.GetOutputFile()
                    Tf.Warning.SetOutputFile(None)
                except (AttributeError, TypeError):
                    pass
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if USD_AVAILABLE and Tf is not None:
                try:
                    # 원래 핸들러 복원
                    if self.original_handler is not None:
                        Tf.Warning.SetWarningHandler(self.original_handler)
                except (AttributeError, TypeError):
                    pass
                try:
                    # 원래 출력 파일 복원
                    if self.original_output_file is not None:
                        Tf.Warning.SetOutputFile(self.original_output_file)
                except (AttributeError, TypeError):
                    pass
            return False
    
    return USDWarningSuppressor()


def parse_usd_file_with_api(
    usd_file_path: str,
    base_path: Optional[List[str]] = None,
    _visited: Optional[set] = None,
    parse_type: str = "both"
) -> List[Dict[str, str]]:
    """
    USD Python API를 사용하여 scene → shot → object/actor 구조를 따라가며 정보를 추출합니다.
    
    Args:
        usd_file_path: 시작 USD 파일 경로
        base_path: 상위 구조(scene/shot 등)를 나타내는 경로 조각 리스트
        _visited: 순환 참조 방지를 위한 내부용 캐시
        parse_type: 파싱할 타입 ("object", "actor", "both")
        
        Returns:
        [{
            "object_name": "object_1" (또는 "actor_name": "actor_1"),
            "object_path": "scene_1/shot_1/object_1" (또는 "actor_path": "scene_1/shot_1/actor_1"),
            "object_id": "1" (또는 "actor_id": "1"),
            "description": "...",
            "category": "...",
            "parsing_method": "pxr" or "regex",
            ...
        }, ...]
    """
    if not USD_AVAILABLE:
        # USD API가 없으면 정규식 기반 파싱 사용
        results = parse_usd_file_regex(usd_file_path, parse_type=parse_type)
        # 파싱 방법 표시
        for result in results:
            result["parsing_method"] = "regex"
        return results
    
    usd_file_path = os.path.abspath(usd_file_path)
    base_dir = os.path.dirname(usd_file_path)
    base_path = base_path or []
    _visited = _visited or set()
    
    if usd_file_path in _visited:
        return []
    _visited.add(usd_file_path)
    
    # 파일 존재 여부 확인
    if not os.path.exists(usd_file_path):
        print(f"❌ USD 파일이 존재하지 않습니다: {usd_file_path}")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
        # USD API 실패 시 regex 파싱으로 fallback
        print(f"   ⚠️ USD API 실패로 정규식 파싱으로 전환합니다...")
        results = parse_usd_file_regex(usd_file_path, parse_type=parse_type)
        for result in results:
            result["parsing_method"] = "regex"
        return results
    
    # 파일 읽기 권한 확인
    if not os.access(usd_file_path, os.R_OK):
        print(f"❌ USD 파일 읽기 권한이 없습니다: {usd_file_path}")
        print(f"   파일 권한: {oct(os.stat(usd_file_path).st_mode)[-3:]}")
        # USD API 실패 시 regex 파싱으로 fallback
        print(f"   ⚠️ USD API 실패로 정규식 파싱으로 전환합니다...")
        results = parse_usd_file_regex(usd_file_path, parse_type=parse_type)
        for result in results:
            result["parsing_method"] = "regex"
        return results
    
    try:
        with _suppress_usd_warnings():
            stage = Usd.Stage.Open(usd_file_path)
    except Exception as exc:
        print(f"⚠️ USD Stage Open 실패 ({usd_file_path}): {exc}")
        print(f"   파일 크기: {os.path.getsize(usd_file_path) if os.path.exists(usd_file_path) else 'N/A'} bytes")
        print(f"   ⚠️ USD API 실패로 정규식 파싱으로 전환합니다...")
        # USD API 실패 시 regex 파싱으로 fallback
        results = parse_usd_file_regex(usd_file_path, parse_type=parse_type)
        for result in results:
            result["parsing_method"] = "regex"
        return results
    
    if not stage:
        print(f"⚠️ USD Stage가 None입니다: {usd_file_path}")
        print(f"   ⚠️ USD API 실패로 정규식 파싱으로 전환합니다...")
        # USD API 실패 시 regex 파싱으로 fallback
        results = parse_usd_file_regex(usd_file_path, parse_type=parse_type)
        for result in results:
            result["parsing_method"] = "regex"
        return results
    
    results: List[Dict[str, str]] = []
    file_parent = os.path.basename(os.path.dirname(usd_file_path))
    file_stem = os.path.splitext(os.path.basename(usd_file_path))[0]
    
    # 객체 USD인지 판별 (objects 디렉토리 하위 파일로 추정)
    is_object_file = file_parent == "objects"
    # [OBJECT-ONLY] 액터 USD인지 판별 (actors 디렉토리 하위 파일로 추정)
    is_actor_file = ENABLE_ACTOR_PARSING and file_parent == "actors"
    
    def _build_object_result(custom_data: Dict[str, str], usd_file_path: str = None, prim=None) -> Optional[Dict[str, str]]:
        category = custom_data.get("category", "")
        object_id = custom_data.get("object_id", "")
        image_path = custom_data.get("image_path", "")
        rig_type = _resolve_rig_type(custom_data, prim)

        if base_path and len(base_path) > 0 and base_path[-1] == file_stem:
            object_path = "/".join(base_path)
        else:
            object_path = "/".join(base_path + [file_stem]) if base_path else file_stem

        object_name = file_stem
        base_path_str = os.path.dirname(base_dir)

        result = {
            "target": "object",
            "object_name": object_name,
            "object_path": object_path,
            "object_id": object_id or file_stem,
            "object_scope": _determine_object_scope(object_path, usd_file_path),
            "parsing_method": "pxr",
            "base_path": base_path_str,
        }
        _attach_bilingual_object_fields(result, custom_data)
        if category:
            result["category"] = category
        if rig_type:
            result["rig_type"] = rig_type
        if image_path:
            result["image_path"] = image_path
        if usd_file_path:
            result["usd_file_path"] = os.path.normpath(usd_file_path)

        return result

    # [OBJECT-ONLY DISABLED] actor 결과 빌더 — ENABLE_ACTOR_PARSING=True 시 사용
    def _build_actor_result(custom_data: Dict[str, str], usd_file_path: str = None) -> Optional[Dict[str, str]]:
        actor_id = custom_data.get("ID", "") or custom_data.get("object_id", "")
        name = custom_data.get("name", "")
        gender = custom_data.get("gender", "")
        age = custom_data.get("age", "")
        outfit = custom_data.get("outfit", "")
        location = custom_data.get("location", "")
        action = custom_data.get("action", "")
        pose = custom_data.get("pose", "")
        description = _resolve_description(custom_data)
        image_path = custom_data.get("image_path", "")
        asset_path = custom_data.get("asset_path", "")
        
        # actor_path 생성: base_path의 마지막 요소가 file_stem과 같으면 중복 추가하지 않음
        if base_path and len(base_path) > 0 and base_path[-1] == file_stem:
            actor_path = "/".join(base_path)
        else:
            actor_path = "/".join(base_path + [file_stem]) if base_path else file_stem
        
        # actor_name은 file_stem만 사용 (actor_1 형식)
        actor_name = file_stem
        
        # base_path: USD 파일이 있는 디렉토리(예: .../scene_1/shot_3/actors)의 상위 디렉토리(예: .../scene_1/shot_3)
        base_path_str = os.path.dirname(base_dir)
        
        result = {
            "target": "actor",
            "actor_name": actor_name,
            "actor_path": actor_path,
            "actor_id": actor_id or file_stem,
            "parsing_method": "pxr",
            "base_path": base_path_str,
        }
        if name:
            result["name"] = name
        if gender:
            result["gender"] = gender
        if age:
            result["age"] = age
        if outfit:
            result["outfit"] = outfit
        if location:
            result["location"] = location
        if action:
            result["action"] = action
        if pose:
            result["pose"] = pose
        if description:
            result["description"] = description
        if image_path:
            result["image_path"] = image_path
        if asset_path:
            result["asset_path"] = asset_path
        # USD 파일 경로 저장 (reference 경로)
        if usd_file_path:
            result["usd_file_path"] = os.path.normpath(usd_file_path)
        
        return result
    
    if is_object_file and parse_type in ("object", "both"):
        captured = False
        for prim in stage.Traverse():
            custom_data = prim.GetCustomData()
            if not custom_data:
                continue
            result = _build_object_result(custom_data, usd_file_path, prim)
            if result:
                results.append(result)
                captured = True
                break
        if not captured:
            # customData가 없더라도 최소 정보 저장
            # object_path 생성: base_path의 마지막 요소가 file_stem과 같으면 중복 추가하지 않음
            if base_path and len(base_path) > 0 and base_path[-1] == file_stem:
                object_path = "/".join(base_path)
            else:
                object_path = "/".join(base_path + [file_stem]) if base_path else file_stem
            
            # object_name은 file_stem만 사용 (object_1 형식)
            object_name = file_stem
            
            # base_path: USD 파일이 있는 디렉토리(예: .../scene_1/shot_3/objects)의 상위 디렉토리(예: .../scene_1/shot_3)
            base_path_str = os.path.dirname(base_dir)
            
            result_dict = {
                "target": "object",
                "object_name": object_name,
                "object_path": object_path,
                "object_id": file_stem,
                "object_scope": _determine_object_scope(object_path, usd_file_path),
                "parsing_method": "pxr",
                "base_path": base_path_str,
            }
            if usd_file_path:
                result_dict["usd_file_path"] = os.path.normpath(usd_file_path)
            results.append(result_dict)
        return results
    
    if ENABLE_ACTOR_PARSING and is_actor_file and parse_type in ("actor", "both"):
        captured = False
        for prim in stage.Traverse():
            custom_data = prim.GetCustomData()
            if not custom_data:
                continue
            result = _build_actor_result(custom_data, usd_file_path)
            if result:
                results.append(result)
                captured = True
                break
        if not captured:
            # customData가 없더라도 최소 정보 저장
            # actor_path 생성: base_path의 마지막 요소가 file_stem과 같으면 중복 추가하지 않음
            if base_path and len(base_path) > 0 and base_path[-1] == file_stem:
                actor_path = "/".join(base_path)
            else:
                actor_path = "/".join(base_path + [file_stem]) if base_path else file_stem
            
            # actor_name은 file_stem만 사용 (actor_1 형식)
            actor_name = file_stem
            
            # base_path: USD 파일이 있는 디렉토리(예: .../scene_1/shot_3/actors)의 상위 디렉토리(예: .../scene_1/shot_3)
            base_path_str = os.path.dirname(base_dir)
            
            result_dict = {
                "target": "actor",
                "actor_name": actor_name,
                "actor_path": actor_path,
                "actor_id": file_stem,
                "parsing_method": "pxr",
                "base_path": base_path_str,
            }
            if usd_file_path:
                result_dict["usd_file_path"] = os.path.normpath(usd_file_path)
            results.append(result_dict)
        return results
    
    # 객체/액터 파일이 아닌 경우: 하위 참조 탐색
    def extract_reference_paths(prim: Usd.Prim) -> List[tuple]:
        """
        prim의 reference 경로를 추출합니다.
        
        Returns:
            List[tuple]: (상대경로, 절대경로) 튜플 리스트
        """
        paths: List[tuple] = []
        if not prim.HasAuthoredReferences():
            return paths
        
        # 방법 1: PrimStack에서 reference 읽기 (가장 확실한 방법)
        try:
            prim_stack = prim.GetPrimStack()
            for spec in prim_stack:
                if spec and hasattr(spec, 'referenceList') and spec.referenceList:
                    # explicitItems와 prependedItems 모두 확인
                    ref_items = []
                    if hasattr(spec.referenceList, 'explicitItems'):
                        ref_items.extend(spec.referenceList.explicitItems)
                    if hasattr(spec.referenceList, 'prependedItems'):
                        ref_items.extend(spec.referenceList.prependedItems)
                    
                    for ref in ref_items:
                        # Sdf.Reference 객체인 경우
                        if hasattr(ref, 'assetPath'):
                            asset_path = ref.assetPath
                        elif isinstance(ref, str):
                            asset_path = ref
                        else:
                            # 문자열로 변환 시도
                            asset_path = str(ref)
                            # Sdf.Reference 문자열 표현에서 경로 추출
                            if '@' in asset_path:
                                # 예: "./objects/object_1.usda" 추출
                                import re
                                match = re.search(r'@([^@]+)@', asset_path)
                                if match:
                                    asset_path = match.group(1)
                        
                        if asset_path:
                            # 상대 경로 저장 (원본 유지)
                            relative_path = asset_path
                            # ./ 제거하지 않고 그대로 저장 (나중에 사용할 때 처리)
                            
                            # 절대 경로 계산
                            if asset_path.startswith("./"):
                                asset_path_clean = asset_path[2:]
                            else:
                                asset_path_clean = asset_path
                            
                            # spec의 layer 경로 기준으로 상대 경로 해석
                            if spec.layer:
                                layer_path = spec.layer.identifier
                                layer_dir = os.path.dirname(layer_path)
                                resolved = (
                                    asset_path_clean
                                    if os.path.isabs(asset_path_clean)
                                    else os.path.join(layer_dir, asset_path_clean)
                                )
                            else:
                                resolved = (
                                    asset_path_clean
                                    if os.path.isabs(asset_path_clean)
                                    else os.path.join(base_dir, asset_path_clean)
                                )
                            resolved = os.path.normpath(resolved)
                            if resolved.lower().endswith((".usd", ".usda", ".usdc")) and os.path.exists(resolved):
                                # (상대경로, 절대경로) 튜플로 저장
                                path_tuple = (relative_path, resolved)
                                if path_tuple not in paths:
                                    paths.append(path_tuple)
        except Exception as e:
            print(f"[DEBUG USD] PrimStack에서 reference 읽기 실패: {e}")
        
        return paths
    
    # "objects" Xform을 찾아서 그 안의 object들을 처리
    # 모든 "objects" Xform을 찾기 (여러 shot에 있을 수 있음)
    objects_prims = []
    if parse_type in ("object", "both"):
        for prim in stage.Traverse():
            if prim.GetName() == "objects":
                objects_prims.append(prim)
                print(f"[DEBUG USD] objects Xform 발견: {prim.GetPath()}, base_path={base_path}")
    
    if objects_prims:
        # 모든 "objects" Xform 처리
        for objects_prim in objects_prims:
            # "objects" Xform 안의 자식 prim들 확인
            print(f"[DEBUG USD] objects Xform의 자식 prim 수: {len(list(objects_prim.GetChildren()))}")
            for child_prim in objects_prim.GetChildren():
                child_name = child_prim.GetName()
                print(f"[DEBUG USD] 자식 prim: {child_name}, 경로: {child_prim.GetPath()}")
                # object_1, object_2 등 object로 시작하는 prim만 처리
                if not child_name.startswith("object_"):
                    print(f"[DEBUG USD] '{child_name}'는 object_로 시작하지 않아서 건너뜀")
                    continue
                
                # object의 reference 경로 추출
                child_paths = extract_reference_paths(child_prim)
                print(f"[DEBUG USD] '{child_name}'의 reference 경로: {child_paths}")
                if not child_paths:
                    print(f"[DEBUG USD] '{child_name}'의 reference 경로를 찾을 수 없음")
                    continue
                
                # base_path 구성: prim 경로의 scene_*/shot_* 마커 기반(폴더 구조 무관)
                prim_path = str(child_prim.GetPath())
                next_base = _derive_next_base(prim_path, base_path, child_name)
                
                print(f"[DEBUG USD] '{child_name}'의 next_base: {next_base}")
                
                # 각 reference된 USD 파일로 재귀적으로 들어가기
                for relative_path, child_path in child_paths:
                    print(f"[DEBUG USD] '{child_name}'의 USD 파일로 재귀: {child_path} (상대경로: {relative_path})")
                    # 재귀 호출 결과에 상대 경로 정보 전달
                    child_results = parse_usd_file_with_api(
                        child_path,
                        next_base,
                        _visited,
                        parse_type,
                    )
                    # 각 결과에 상대 경로 추가
                    for result in child_results:
                        if "usd_relative_path" not in result:
                            # 상대 경로 정규화 (./ 제거)
                            normalized_relative = relative_path
                            if normalized_relative.startswith("./"):
                                normalized_relative = normalized_relative[2:]
                            result["usd_relative_path"] = normalized_relative
                    results.extend(child_results)
    
    # [OBJECT-ONLY DISABLED] "actors" Xform을 찾아서 그 안의 actor들을 처리
    actors_prims = []
    if ENABLE_ACTOR_PARSING and parse_type in ("actor", "both"):
        for prim in stage.Traverse():
            if prim.GetName() == "actors":
                actors_prims.append(prim)
                print(f"[DEBUG USD] actors Xform 발견: {prim.GetPath()}, base_path={base_path}")
    
    if ENABLE_ACTOR_PARSING and actors_prims:
        for actors_prim in actors_prims:
            # "actors" Xform 안의 자식 prim들 확인
            print(f"[DEBUG USD] actors Xform의 자식 prim 수: {len(list(actors_prim.GetChildren()))}")
            for child_prim in actors_prim.GetChildren():
                child_name = child_prim.GetName()
                print(f"[DEBUG USD] 자식 prim: {child_name}, 경로: {child_prim.GetPath()}")
                # actor_1, actor_2 등 actor로 시작하는 prim만 처리
                if not child_name.startswith("actor_"):
                    print(f"[DEBUG USD] '{child_name}'는 actor_로 시작하지 않아서 건너뜀")
                    continue
                
                # actor의 reference 경로 추출
                child_paths = extract_reference_paths(child_prim)
                print(f"[DEBUG USD] '{child_name}'의 reference 경로: {child_paths}")
                if not child_paths:
                    print(f"[DEBUG USD] '{child_name}'의 reference 경로를 찾을 수 없음")
                    continue
                
                # base_path 구성: prim 경로의 scene_*/shot_* 마커 기반(폴더 구조 무관)
                prim_path = str(child_prim.GetPath())
                next_base = _derive_next_base(prim_path, base_path, child_name)
                
                print(f"[DEBUG USD] '{child_name}'의 next_base: {next_base}")
                
                # 각 reference된 USD 파일로 재귀적으로 들어가기
                for relative_path, child_path in child_paths:
                    print(f"[DEBUG USD] '{child_name}'의 USD 파일로 재귀: {child_path} (상대경로: {relative_path})")
                    # 재귀 호출 결과에 상대 경로 정보 전달
                    child_results = parse_usd_file_with_api(
                        child_path,
                        next_base,
                        _visited,
                        parse_type,
                    )
                    # 각 결과에 상대 경로 추가
                    for result in child_results:
                        if "usd_relative_path" not in result:
                            # 상대 경로 정규화 (./ 제거)
                            normalized_relative = relative_path
                            if normalized_relative.startswith("./"):
                                normalized_relative = normalized_relative[2:]
                            result["usd_relative_path"] = normalized_relative
                    results.extend(child_results)
    
    if not objects_prims and (not ENABLE_ACTOR_PARSING or not actors_prims):
        # "objects" Xform이 없는 경우: 일반적인 참조 탐색 (scene, shot 등)
        for prim in stage.Traverse():
            prim_name = prim.GetName()
            if prim_name in ("objects", "actors"):
                continue
            
            child_paths = extract_reference_paths(prim)
            if not child_paths:
                continue
            
            if prim_name.startswith("object_") and parse_type in ("object", "both"):
                next_base = base_path
            elif ENABLE_ACTOR_PARSING and prim_name.startswith("actor_") and parse_type in ("actor", "both"):
                next_base = base_path
            elif prim_name:
                next_base = base_path + [prim_name]
            else:
                next_base = base_path
            
            for relative_path, child_path in child_paths:
                child_results = parse_usd_file_with_api(
                    child_path,
                    next_base,
                    _visited,
                    parse_type,
                )
                # 각 결과에 상대 경로 추가
                for result in child_results:
                    if "usd_relative_path" not in result:
                        # 상대 경로 정규화 (./ 제거)
                        normalized_relative = relative_path
                        if normalized_relative.startswith("./"):
                            normalized_relative = normalized_relative[2:]
                        result["usd_relative_path"] = normalized_relative
                results.extend(child_results)
    
    return results


def parse_usd_file_regex(
    usd_file_path: str,
    parse_type: str = "both"
) -> List[Dict[str, str]]:
    """
    정규식을 사용하여 scene → shot → object/actor 구조를 따라가며 정보를 추출합니다 (fallback).
    
    Args:
        usd_file_path: USD 파일 경로
        parse_type: 파싱할 타입 ("object", "actor", "both")
    """
    usd_file_path = os.path.abspath(usd_file_path)
    results: List[Dict[str, str]] = []
    visited: set = set()
    
    def _extract_field(block: str, field: str) -> str:
        match = re.search(rf'string {re.escape(field)}\s*=\s*"([^"]*)"', block)
        return match.group(1) if match else ""

    def _extract_bilingual_field(block: str, field: str) -> Dict[str, str]:
        dict_match = re.search(
            rf'dictionary\s+{re.escape(field)}\s*=\s*\{{([^{{}}]*)\}}',
            block,
            re.DOTALL,
        )
        if dict_match:
            inner = dict_match.group(1)
            ko_match = re.search(r'string\s+ko\s*=\s*"([^"]*)"', inner)
            en_match = re.search(r'string\s+en\s*=\s*"([^"]*)"', inner)
            return coerce_bilingual({
                "ko": ko_match.group(1) if ko_match else "",
                "en": en_match.group(1) if en_match else "",
            })
        return coerce_bilingual(_extract_field(block, field))

    def _build_custom_data_from_block(custom_block: str) -> Dict[str, Any]:
        custom_data: Dict[str, Any] = {}
        for field_name in ("description", "base_description", "appearance", "name", "location", "action"):
            bilingual = _extract_bilingual_field(custom_block, field_name)
            if not is_blank(bilingual):
                custom_data[field_name] = bilingual
        return custom_data

    def _extract_customdata_block(content: str) -> str:
        """
        customData = { ... } 블록을 중괄호 균형 매칭으로 추출합니다.

        retrieval_result 같은 중첩 dictionary가 있어도 첫 번째 '}'에서 잘리지 않도록
        depth를 추적하여 전체 블록을 반환합니다. (regex fallback의 nested customData 지원)
        """
        idx = content.find('customData')
        if idx < 0:
            return ""
        brace_start = content.find('{', idx)
        if brace_start < 0:
            return ""
        depth = 0
        for i in range(brace_start, len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    return content[brace_start + 1:i]
        return content[brace_start + 1:]
    
    def _parse_object_file(file_path: str, base_components: List[str], relative_path: str = None) -> None:
        file_path_abs = os.path.abspath(file_path)
        try:
            with open(file_path_abs, "r", encoding="utf-8") as fh:
                content = fh.read()
        except FileNotFoundError:
            return
        
        custom_block = _extract_customdata_block(content)
        custom_data = _build_custom_data_from_block(custom_block)

        category = _extract_field(custom_block, "category")
        object_id = _extract_field(custom_block, "object_id") or os.path.splitext(os.path.basename(file_path_abs))[0]
        image_path = _extract_field(custom_block, "image_path")
        # rig_type: etc 중첩이든 최상위든 custom_block(중괄호 균형 추출)에 포함되므로 직접 매칭
        rig_type = _normalize_rig_type(_extract_field(custom_block, "rig_type"))
        
        # base_components에 이미 object 이름이 포함되어 있을 수 있음 (예: ["scene_1", "shot_1", "object_1"])
        file_stem = os.path.splitext(os.path.basename(file_path_abs))[0]
        
        # object_path 생성: base_components의 마지막 요소가 file_stem과 같으면 중복 추가하지 않음
        if base_components and len(base_components) > 0 and base_components[-1] == file_stem:
            object_path = "/".join(base_components)
        else:
            object_path = "/".join(base_components + [file_stem]) if base_components else file_stem
        
        # object_name은 file_stem만 사용 (object_1 형식)
        object_name = file_stem
        
        # base_path: USD 파일이 있는 디렉토리(예: .../scene_1/shot_3/objects)의 상위 디렉토리(예: .../scene_1/shot_3)
        base_path_str = os.path.dirname(os.path.dirname(file_path_abs))
        
        result: Dict[str, str] = {
            "target": "object",
            "object_name": object_name,
            "object_path": object_path,
            "object_id": object_id,
            "object_scope": _determine_object_scope(object_path, file_path_abs),
            "parsing_method": "regex",
            "base_path": base_path_str,
        }
        _attach_bilingual_object_fields(result, custom_data)
        if category:
            result["category"] = category
        if rig_type:
            result["rig_type"] = rig_type
        if image_path:
            result["image_path"] = image_path
        if relative_path:
            # 상대 경로 정규화 (./ 제거)
            normalized_relative = relative_path
            if normalized_relative.startswith("./"):
                normalized_relative = normalized_relative[2:]
            result["usd_relative_path"] = normalized_relative
        
        results.append(result)
    
    # [OBJECT-ONLY DISABLED] regex actor 파일 파서 — ENABLE_ACTOR_PARSING=True 시 사용
    def _parse_actor_file(file_path: str, base_components: List[str], relative_path: str = None) -> None:
        file_path_abs = os.path.abspath(file_path)
        try:
            with open(file_path_abs, "r", encoding="utf-8") as fh:
                content = fh.read()
        except FileNotFoundError:
            return
        
        custom_match = re.search(r'customData\s*=\s*\{([^}]*)\}', content, re.DOTALL)
        custom_block = custom_match.group(1) if custom_match else ""
        
        actor_id = _extract_field(custom_block, "ID") or _extract_field(custom_block, "object_id") or os.path.splitext(os.path.basename(file_path_abs))[0]
        name = _extract_field(custom_block, "name")
        gender = _extract_field(custom_block, "gender")
        age = _extract_field(custom_block, "age")
        outfit = _extract_field(custom_block, "outfit")
        location = _extract_field(custom_block, "location")
        action = _extract_field(custom_block, "action")
        pose = _extract_field(custom_block, "pose")
        description = _extract_field(custom_block, "description") or _extract_field(custom_block, "base_description")
        image_path = _extract_field(custom_block, "image_path")
        asset_path = _extract_field(custom_block, "asset_path")
        
        # base_components에 이미 actor 이름이 포함되어 있을 수 있음 (예: ["scene_1", "shot_1", "actor_1"])
        file_stem = os.path.splitext(os.path.basename(file_path_abs))[0]
        
        # actor_path 생성: base_components의 마지막 요소가 file_stem과 같으면 중복 추가하지 않음
        if base_components and len(base_components) > 0 and base_components[-1] == file_stem:
            actor_path = "/".join(base_components)
        else:
            actor_path = "/".join(base_components + [file_stem]) if base_components else file_stem
        
        # actor_name은 file_stem만 사용 (actor_1 형식)
        actor_name = file_stem
        
        # base_path: USD 파일이 있는 디렉토리(예: .../scene_1/shot_3/actors)의 상위 디렉토리(예: .../scene_1/shot_3)
        base_path_str = os.path.dirname(os.path.dirname(file_path_abs))
        
        result: Dict[str, str] = {
            "target": "actor",
            "actor_name": actor_name,
            "actor_path": actor_path,
            "actor_id": actor_id,
            "parsing_method": "regex",
            "base_path": base_path_str,
        }
        if name:
            result["name"] = name
        if gender:
            result["gender"] = gender
        if age:
            result["age"] = age
        if outfit:
            result["outfit"] = outfit
        if location:
            result["location"] = location
        if action:
            result["action"] = action
        if pose:
            result["pose"] = pose
        if description:
            result["description"] = description
        if image_path:
            result["image_path"] = image_path
        if asset_path:
            result["asset_path"] = asset_path
        if relative_path:
            # 상대 경로 정규화 (./ 제거)
            normalized_relative = relative_path
            if normalized_relative.startswith("./"):
                normalized_relative = normalized_relative[2:]
            result["usd_relative_path"] = normalized_relative
        
        results.append(result)
    
    def _recurse(file_path: str, base_components: List[str], relative_path: str = None) -> None:
        file_path_abs = os.path.abspath(file_path)
        if file_path_abs in visited:
            return
        visited.add(file_path_abs)
        
        parent = os.path.basename(os.path.dirname(file_path_abs))
        if parent == "objects" and parse_type in ("object", "both"):
            _parse_object_file(file_path_abs, base_components, relative_path)
            return
        if ENABLE_ACTOR_PARSING and parent == "actors" and parse_type in ("actor", "both"):
            _parse_actor_file(file_path_abs, base_components, relative_path)
            return
        
        try:
            with open(file_path_abs, "r", encoding="utf-8") as fh:
                content = fh.read()
        except FileNotFoundError:
            return
        
        base_dir = os.path.dirname(file_path_abs)
        
        # "objects" Xform을 찾아서 그 안의 object들을 처리
        # 중괄호 매칭을 사용하여 "objects" Xform의 전체 내용 추출
        objects_start = content.find('def Xform "objects"')
        if objects_start >= 0 and parse_type in ("object", "both"):
            # "objects" Xform의 시작 위치 찾기
            brace_start = content.find('{', objects_start)
            if brace_start >= 0:
                # 중괄호 매칭으로 "objects" Xform의 끝 찾기
                brace_count = 0
                brace_end = -1
                for i in range(brace_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i
                            break
                
                if brace_end > brace_start:
                    # "objects" Xform 안의 내용
                    objects_content = content[brace_start + 1:brace_end]
                    
                    # "objects" 안의 object_* Xform들의 reference 찾기
                    object_ref_pattern = re.compile(
                        r'def Xform\s+"(object_\d+)"\s*\([^)]*references\s*=\s*@([^@]+)@',
                        re.MULTILINE
                    )
                    
                    found_objects = False
                    for object_name, asset_ref in object_ref_pattern.findall(objects_content):
                        found_objects = True
                        # 원본 상대 경로 저장
                        original_relative = asset_ref.strip()
                        asset_ref = original_relative
                        if asset_ref.startswith("./"):
                            asset_ref = asset_ref[2:]
                        resolved = (
                            asset_ref
                            if os.path.isabs(asset_ref)
                            else os.path.join(base_dir, asset_ref)
                        )
                        resolved = os.path.normpath(resolved)
                        if not resolved.lower().endswith((".usd", ".usda", ".usdc")):
                            continue
                        if not os.path.exists(resolved):
                            continue
                        
                        # object 경로에 object 이름 추가 (예: scene_1/shot_1/object_1)
                        next_components = base_components + [object_name] if base_components else [object_name]
                        _recurse(resolved, next_components, original_relative)
                    
                    # "objects" Xform을 찾았고 object들이 있으면, 다른 참조는 처리하지 않음
                    if found_objects and parse_type == "object":
                        return
        
        # [OBJECT-ONLY DISABLED] "actors" Xform regex 파싱 — ENABLE_ACTOR_PARSING=True 시 사용
        if ENABLE_ACTOR_PARSING:
            actors_start = content.find('def Xform "actors"')
            if actors_start >= 0 and parse_type in ("actor", "both"):
                brace_start = content.find('{', actors_start)
                if brace_start >= 0:
                    brace_count = 0
                    brace_end = -1
                    for i in range(brace_start, len(content)):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i
                                break
                    
                    if brace_end > brace_start:
                        actors_content = content[brace_start + 1:brace_end]
                        
                        actor_ref_pattern = re.compile(
                            r'def Xform\s+"(actor_\d+)"\s*\([^)]*references\s*=\s*@([^@]+)@',
                            re.MULTILINE
                        )
                        
                        found_actors = False
                        for actor_name, asset_ref in actor_ref_pattern.findall(actors_content):
                            found_actors = True
                            original_relative = asset_ref.strip()
                            asset_ref = original_relative
                            if asset_ref.startswith("./"):
                                asset_ref = asset_ref[2:]
                            resolved = (
                                asset_ref
                                if os.path.isabs(asset_ref)
                                else os.path.join(base_dir, asset_ref)
                            )
                            resolved = os.path.normpath(resolved)
                            if not resolved.lower().endswith((".usd", ".usda", ".usdc")):
                                continue
                            if not os.path.exists(resolved):
                                continue
                            
                            next_components = base_components + [actor_name] if base_components else [actor_name]
                            _recurse(resolved, next_components, original_relative)
                        
                        if found_actors and parse_type == "actor":
                            return
        
        # "objects" Xform이 없거나 비어있는 경우: 일반적인 참조 탐색 (scene, shot 등)
        ref_pattern = re.compile(
            r'def [^{\n]*?"([^"]+)"\s*\([^)]*references\s*=\s*@([^@]+)@',
            re.MULTILINE
        )
        
        for prim_name, asset_ref in ref_pattern.findall(content):
            # "objects"/"actors" Xform은 이미 처리했으므로 건너뛰기
            if prim_name in ("objects", "actors"):
                continue
            
            # 원본 상대 경로 저장
            original_relative = asset_ref.strip()
            asset_ref = original_relative
            if asset_ref.startswith("./"):
                asset_ref = asset_ref[2:]
            resolved = (
                asset_ref
                if os.path.isabs(asset_ref)
                else os.path.join(base_dir, asset_ref)
            )
            resolved = os.path.normpath(resolved)
            if not resolved.lower().endswith((".usd", ".usda", ".usdc")):
                continue
            if not os.path.exists(resolved):
                continue
            
            if prim_name.startswith("object_") and parse_type in ("object", "both"):
                next_components = base_components
            elif ENABLE_ACTOR_PARSING and prim_name.startswith("actor_") and parse_type in ("actor", "both"):
                next_components = base_components
            elif prim_name:
                next_components = base_components + [prim_name]
            else:
                next_components = base_components
            
            _recurse(resolved, next_components, original_relative)
    
    _recurse(usd_file_path, [])
    return results


def _dedupe_object_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    동일 USD 파일(usd_file_path)이 두 번 이상 추출된 경우 중복을 제거합니다.

    shot prim은 `./objects/object_N.usda`와 `../../objects/object_N.usda` 두 reference를
    모두 따라가 scene canonical이 shot 경로로 중복 등록될 수 있습니다(스펙 2.1).
    같은 usd_file_path가 여러 번 등장하면 scene scope 항목을 우선 유지하여 중복을 제거합니다.
    """
    seen: Dict[str, int] = {}
    deduped: List[Dict[str, str]] = []
    for item in results:
        # object가 아니거나 usd_file_path가 없으면 그대로 유지
        if item.get("target") != "object" or not item.get("usd_file_path"):
            deduped.append(item)
            continue
        key = os.path.normpath(item["usd_file_path"])
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(item)
            continue
        # 이미 본 파일: scene scope를 우선 (canonical 경로로 정규화)
        existing_idx = seen[key]
        existing = deduped[existing_idx]
        if existing.get("object_scope") != "scene" and item.get("object_scope") == "scene":
            deduped[existing_idx] = item
    return deduped


def parse_usd_file(usd_file_path: str, parse_type: str = "object") -> List[Dict[str, str]]:
    """
    USD 파일에서 object/actor 정보를 추출합니다 (USD API 우선 사용).
    
    Args:
        usd_file_path: USD 파일 경로
        parse_type: 파싱할 타입 ("object", "actor", "both")
        
    Returns:
        [{"object_path": "scene_1/shot_1/object_1", "object_id": "1", "description": "...", "object_scope": "scene"|"shot", "parsing_method": "pxr" or "regex", ...}, ...]
        또는
        [{"actor_path": "scene_1/shot_1/actor_1", "actor_id": "1", "name": "...", "description": "...", "parsing_method": "pxr" or "regex", ...}, ...]
    """
    results = parse_usd_file_with_api(usd_file_path, [], parse_type=parse_type)
    return _dedupe_object_results(results)


# 사용 예시
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="USD 파일에서 객체(object) 및 액터(actor) 정보를 추출하여 JSON으로 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # USD 파일 파싱 및 JSON 저장 (object와 actor 모두)
  python usd_parser.py scene.usd --output results.json
  
  # object만 파싱
  python usd_parser.py scene.usd --output objects.json --type object
  
  # actor만 파싱
  python usd_parser.py scene.usd --output actors.json --type actor
  
  # 둘 다 파싱 (기본값)
  python usd_parser.py scene.usd --output results.json --type both
        """
    )
    parser.add_argument(
        "usd_file",
        help="USD 파일 경로"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="출력 JSON 파일 경로"
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["object", "actor", "both"],
        default="object",
        help="파싱할 타입: object (객체만, 기본값), actor (액터만, ENABLE_ACTOR_PARSING=True 필요), both"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"USD 파일 파싱: {args.usd_file}")
        print(f"파싱 타입: {args.type}")
        results = parse_usd_file(args.usd_file, parse_type=args.type)
        
        print(f"추출된 항목 수: {len(results)}")
        
        # 타입별 개수 확인
        objects = [r for r in results if "object_name" in r]
        actors = [r for r in results if "actor_name" in r] if ENABLE_ACTOR_PARSING else []
        print(f"  - 객체(object): {len(objects)}개")
        if ENABLE_ACTOR_PARSING:
            print(f"  - 액터(actor): {len(actors)}개")
        
        # description이 있는 항목만 필터링
        results_with_description = [
            r for r in results
            if not is_blank(r.get("description"))
            or r.get("description_en", "").strip()
            or r.get("description_ko", "").strip()
        ]
        print(f"description이 있는 항목 수: {len(results_with_description)}")
        
        # 결과 저장
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results_with_description, f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

