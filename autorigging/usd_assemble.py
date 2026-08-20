"""USD 재조립 (DESIGN §4-3) — motion_to_usd.py + movie_usd_pack.py 통합·일반화.

원 서버 검증 로직을 매니페스트 입력으로 일반화. 미러 트리 대신 실제 USD 트리에
**추가만** 한다 (원본 usda·glb·텍스처 수정 금지 — §7 불가침 원칙).

규칙 (전부 원 서버 실측 검증):
1. cm 단위 (metersPerUnit 0.01). 익스포터가 armature 오브젝트 트랜스폼을 버리므로
   UniRig 정규화 공간→원본 배치 복원 보정 행렬은 USD 쪽 루트 prim 에 넣는다.
   익스포터의 Y-up 회전 xformOp 를 덮지 말고 합성 (M_existing * M_corr).
2. 텍스처 = 기존 bin/texture.jpg 재참조 (중복 저장 금지, 익스포터 textures/ 삭제).
3. 래퍼 = 원본 루트 prim Sdf.CopySpec + geometry 참조를 모션 레이어로 교체 +
   autorig customData 블록.
4. 파일명: <이름>.rt_default.usda (원본과 절대 충돌하지 않는 접미사).
5. 검증: SkelAnimation 타임샘플 · 텍스처 resolve · Mesh bbox 원본 대비 1.00x.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import bpy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weight_transfer as WT   # noqa: E402

MOTION_SUFFIX = "rt_default"


# ---------------------------------------------------------------- 1) 모션 레이어 익스포트
def export_motion_layer(motion: dict, glb: Path, out_layer: Path, name: str) -> dict:
    """모션이 베이크된 씬(default_motion.build 산출)에 원본 GLB 를 합쳐 usda 레이어 익스포트.

    반환: 레이어 manifest dict (nframes/p95/fps/정규화 파라미터/본 수).
    """
    arm, rig_mesh = motion["tgt"], motion["mesh"]

    bpy.ops.import_scene.gltf(filepath=str(glb))
    src_ob = max((o for o in bpy.data.objects
                  if o.type == "MESH" and o is not rig_mesh and len(o.data.uv_layers) > 0),
                 key=lambda o: len(o.data.vertices))

    # align_and_transfer 와 동일 수식으로 정규화 파라미터 캡처 (USD 쪽 역보정용)
    S = WT.world_verts(src_ob)
    R = WT.world_verts(rig_mesh)
    c_s = (S.min(0) + S.max(0)) / 2
    c_r = (R.min(0) + R.max(0)) / 2
    scale = float(np.median((R.max(0) - R.min(0)) /
                            np.maximum(S.max(0) - S.min(0), 1e-9)))

    WT.align_and_transfer(src_ob, rig_mesh, arm)

    keep = {src_ob, arm}
    for o in [o for o in bpy.data.objects if o not in keep]:
        bpy.data.objects.remove(o, do_unlink=True)

    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, motion["nframes"]

    out_layer.parent.mkdir(parents=True, exist_ok=True)
    # 원본 배치 복원은 여기서 하지 않는다 — bpy USD 익스포터가 armature 오브젝트
    # 트랜스폼을 SkelRoot 에 반영하지 않음(실측). fix_scale_and_units 가 USD 쪽에서 보정.
    bpy.ops.wm.usd_export(
        filepath=str(out_layer),
        export_animation=True, export_armatures=True,
        export_materials=True, export_textures=True, overwrite_textures=True,
        convert_orientation=True,          # 기본 forward=-Z, up=Y → geometry.usda 와 동일 Y-up
        root_prim_path=f"/{name}",
    )
    mf = {"nframes": motion["nframes"], "fps": motion["fps"],
          "p95": (round(motion["p95"], 3) if motion["p95"] is not None else None),
          "motion": motion["motion"], "motion_grade": motion["motion_grade"],
          "clips": motion["clips"], "bones": len(arm.data.bones),
          "c_s": [float(v) for v in c_s], "c_r": [float(v) for v in c_r],
          "scale": scale}
    out_layer.with_suffix(".manifest.json").write_text(
        json.dumps(mf, ensure_ascii=False, indent=2))
    return mf


# ---------------------------------------------------------------- 2) 스케일/단위 보정
def _blender_to_usd_y(v):
    """Blender Z-up 좌표 → 익스포터 Y-up 변환(forward -Z, up Y)과 동일한 축 매핑."""
    x, y, z = v
    return Gf.Vec3d(x, z, -y)


def fix_scale_and_units(layer_path: Path, name: str, mf: dict):
    """애님 레이어를 movie_usd 규칙(cm, mpu 0.01)으로 보정 — 원 서버 검증 로직 그대로."""
    c_s = _blender_to_usd_y(mf["c_s"])
    c_r = _blender_to_usd_y(mf["c_r"])
    s = float(mf["scale"])

    def T(v):
        m = Gf.Matrix4d(1.0)
        m.SetTranslate(Gf.Vec3d(v))
        return m

    def S(f):
        m = Gf.Matrix4d(1.0)
        m.SetScale(Gf.Vec3d(f, f, f))
        return m

    M = T(-c_r) * S(1.0 / s) * T(c_s) * S(100.0)

    stage = Usd.Stage.Open(str(layer_path))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/{name}"))
    # 익스포터의 Y-up 변환 회전이 루트 xformOp 로 이미 들어 있다 — 덮어쓰지 말고 합성.
    lt = xf.GetLocalTransformation(Usd.TimeCode.Default())
    existing = lt[0] if isinstance(lt, tuple) else lt
    xf.MakeMatrixXform().Set(existing * M)
    stage.SetMetadata("metersPerUnit", 0.01)
    stage.GetRootLayer().Save()


# ---------------------------------------------------------------- 3) 텍스처 재참조
def repath_texture(layer_path: Path) -> tuple[bool, str]:
    """레이어의 셰이더 텍스처를 같은 폴더의 기존 bin/texture.jpg 로 재지정.

    레이어는 assets/<이름>/ 에 놓이므로 bin/ 이 이미 옆에 있다 (복사 불필요).
    bin/texture.jpg 가 없으면 익스포터 textures/ 를 그대로 두고 경고만 반환.
    """
    tex = layer_path.parent / "bin" / "texture.jpg"
    if not tex.exists():
        return False, "bin/texture.jpg 없음 — 익스포터 textures/ 유지"
    stage = Usd.Stage.Open(str(layer_path))
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader":
            attr = prim.GetAttribute("inputs:file")
            if attr and attr.Get():
                attr.Set(Sdf.AssetPath("./bin/texture.jpg"))
                n += 1
    stage.GetRootLayer().Save()
    tex_dir = layer_path.parent / "textures"
    if n and tex_dir.exists():
        shutil.rmtree(tex_dir)             # 본 실행이 만든 중복본 제거 (규칙 2)
    return n > 0, f"셰이더 {n}개 재지정"


# ---------------------------------------------------------------- 4) 래퍼 생성
def make_wrapper(entry: dict, mf: dict, source_fbx: Path) -> Path:
    """원본 usda 옆에 <이름>.rt_default.usda 래퍼 생성 (원본 무수정)."""
    name = entry["name"]
    base_usda = Path(entry["usda"])
    out = base_usda.parent / f"{name}.{MOTION_SUFFIX}.usda"
    if out.exists():
        out.unlink()                       # 본 모듈 산출물만 재생성 (원본 아님)

    base = Sdf.Layer.FindOrOpen(str(base_usda))
    new = Sdf.Layer.CreateNew(str(out))
    Sdf.CopySpec(base, f"/{name}", new, f"/{name}")   # 캐릭터 메타 customData 무손실

    geom = new.GetPrimAtPath(f"/{name}/geometry")
    if geom is None:                       # 오브젝트 빈 래퍼 등 geometry prim 부재 시 생성
        root_spec = new.GetPrimAtPath(f"/{name}")
        geom = Sdf.PrimSpec(root_spec, "geometry", Sdf.SpecifierDef, "Xform")
    geom.referenceList.prependedItems.clear()
    geom.referenceList.prependedItems.append(
        Sdf.Reference(f"./assets/{name}/{name}.{MOTION_SUFFIX}.usda"))
    cd = dict(geom.customData)
    cd["autorig"] = {
        "rigger": "UniRig",
        "bones": mf["bones"],
        "motion": mf["motion"],
        "motion_grade": mf["motion_grade"],
        "frames": mf["nframes"],
        "fps": mf["fps"],
        "stretch_p95": mf["p95"] if mf["p95"] is not None else "NA",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_fbx": source_fbx.name,
    }
    geom.customData = cd
    new.Save()

    stage = Usd.Stage.Open(str(out))
    stage.SetDefaultPrim(stage.GetPrimAtPath(f"/{name}"))
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(mf["nframes"])
    stage.SetTimeCodesPerSecond(mf["fps"])
    stage.SetFramesPerSecond(mf["fps"])
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().Save()
    return out


# ---------------------------------------------------------------- 5) 검증
def validate(wrapper: Path, entry: dict) -> dict:
    """SkelAnimation 타임샘플 · 텍스처 resolve · Mesh bbox 원본 대비 (§4-3-5)."""
    res = {"anim_samples": 0, "texture_ok": False, "bbox_ratio": None, "pass": False,
           "notes": []}
    stage = Usd.Stage.Open(str(wrapper))

    anims = [len(UsdSkel.Animation(p).GetRotationsAttr().GetTimeSamples())
             for p in stage.Traverse() if p.GetTypeName() == "SkelAnimation"]
    res["anim_samples"] = max(anims) if anims else 0

    texs = [p.GetAttribute("inputs:file").Get()
            for p in stage.Traverse() if p.GetTypeName() == "Shader"
            and p.GetAttribute("inputs:file") and p.GetAttribute("inputs:file").Get()]
    resolved = [t.resolvedPath for t in texs if t.resolvedPath]
    res["texture_ok"] = bool(resolved) and all(Path(t).exists() for t in resolved)
    if not texs:
        res["notes"].append("셰이더 텍스처 입력 없음")

    # bbox 는 Mesh prim 기준 — SkelRoot extent 힌트는 애니 전범위라 비교 금지 (실측 함정)
    geo_path = entry.get("geometry_usda")
    if geo_path and Path(geo_path).exists():
        mesh = next((p for p in stage.Traverse() if p.GetTypeName() == "Mesh"), None)
        if mesh:
            bbox = UsdGeom.BBoxCache(Usd.TimeCode(1), ["default", "render"])
            rng = bbox.ComputeWorldBound(mesh).ComputeAlignedRange()
            size = Gf.Vec3d(rng.GetSize())
            bstage = Usd.Stage.Open(str(geo_path))
            brng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]) \
                .ComputeWorldBound(bstage.GetDefaultPrim()).ComputeAlignedRange()
            bsize = Gf.Vec3d(brng.GetSize())
            ratios = [size[i] / bsize[i] for i in range(3) if bsize[i] > 1e-9]
            res["bbox_ratio"] = round(max(ratios), 3) if ratios else None
    else:
        res["notes"].append("geometry.usda 없음 — bbox 비교 생략")

    bbox_ok = res["bbox_ratio"] is None or abs(res["bbox_ratio"] - 1.0) <= 0.05
    res["pass"] = bool(res["anim_samples"] > 0 and res["texture_ok"] and bbox_ok)
    return res


# ---------------------------------------------------------------- 통합 엔트리포인트
def assemble(entry: dict, motion: dict, rigged_fbx: Path) -> dict:
    """모션 씬(default_motion.build 직후) → 레이어 + 래퍼 + 검증. 결과 dict 반환."""
    name = entry["name"]
    assets_dir = Path(entry["assets_dir"])
    layer = assets_dir / f"{name}.{MOTION_SUFFIX}.usda"

    mf = export_motion_layer(motion, Path(entry["glb"]), layer, name)
    fix_scale_and_units(layer, name, mf)
    tex_ok, tex_note = repath_texture(layer)
    wrapper = make_wrapper(entry, mf, rigged_fbx)
    v = validate(wrapper, entry)
    if not tex_ok:
        v["notes"].append(tex_note)
    return {"layer": str(layer), "wrapper": str(wrapper), "manifest": mf,
            "validation": v}
