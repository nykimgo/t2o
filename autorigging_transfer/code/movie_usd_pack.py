"""movie_usd 규칙 패킹 — motion_to_usd.py --movie-usd 산출 레이어를 규칙에 맞게 마감.

하는 일 (미러 트리 movie_chars/usd_movie/ 안에서, 실제 movie_usd 는 건드리지 않음):
1. 텍스처: 실제 movie_usd 의 assets/<char>/bin/texture.jpg 를 미러 bin/ 으로 복사하고,
   애님 레이어의 셰이더 inputs:file 을 ./bin/texture.jpg 로 재지정 (규칙: 텍스처는 bin/).
   블렌더 익스포터가 만든 textures/ 중복본은 제거.
2. 래퍼: 실제 characters/<char>.usda 의 루트 prim(캐릭터 customData 전체)을 Sdf.CopySpec 으로
   복사해 characters/<char>.rt_<motion>.usda 생성, geometry 참조를 애님 레이어로 교체,
   리타게팅 출처 customData(rt_motion) 추가, 타임코드/기본프림 메타 설정.
3. 검증: 합성 스테이지에서 SkelAnimation 타임샘플·텍스처 해석·bbox(원본 geometry 대비) 확인.

실행 (previs env — pxr):
  /home/sr/miniconda3/envs/previs/bin/python movie_usd_pack.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "movie_chars" / "usd_movie"
MOVIE_USD = Path("/home/sr/previs_proj/movie_usd")


def _blender_to_usd_y(v):
    """Blender Z-up 좌표 → 익스포터 Y-up 변환(forward -Z, up Y)과 동일한 축 매핑."""
    x, y, z = v
    return Gf.Vec3d(x, z, -y)


def fix_scale_and_units(layer_path: Path, char: str, mf: dict):
    """애님 레이어를 movie_usd 규칙(cm, mpu 0.01)으로 보정.

    익스포터가 armature 오브젝트 트랜스폼을 버리므로 레이어는 UniRig 정규화 공간이다.
    루트 prim 에 M = T(-c_r)·S(1/scale)·T(c_s)·S(100) (row-vector 순서) 를 넣어
    geometry.usda 와 같은 배치(원본 glb ×100, cm)로 되돌린다.
    """
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
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/{char}"))
    # 익스포터의 Y-up 변환 회전이 루트 xformOp 로 이미 들어 있다 — 덮어쓰지 말고 합성.
    # (row-vector: 기존 변환 먼저, 보정 나중 → M_existing * M_corr)
    existing = xf.GetLocalTransformation(Usd.TimeCode.Default())[0] \
        if isinstance(xf.GetLocalTransformation(Usd.TimeCode.Default()), tuple) \
        else xf.GetLocalTransformation(Usd.TimeCode.Default())
    xf.MakeMatrixXform().Set(existing * M)
    stage.SetMetadata("metersPerUnit", 0.01)
    stage.GetRootLayer().Save()


def repath_texture(layer_path: Path, char: str) -> bool:
    """애님 레이어의 텍스처를 bin/texture.jpg 로 재지정. 성공 시 True."""
    src_tex = MOVIE_USD / layer_path.parent.relative_to(MIRROR).parts[0] \
        / "characters" / "assets" / char / "bin" / "texture.jpg"
    if not src_tex.exists():
        return False
    bin_dir = layer_path.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    shutil.copy2(src_tex, bin_dir / "texture.jpg")

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
        shutil.rmtree(tex_dir)
    return n > 0


def make_wrapper(movie: str, char: str, motion: str, nframes: int, fps: float,
                 manifest: dict) -> Path:
    base_wrapper = MOVIE_USD / movie / "characters" / f"{char}.usda"
    out = MIRROR / movie / "characters" / f"{char}.rt_{motion}.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    base = Sdf.Layer.FindOrOpen(str(base_wrapper))
    new = Sdf.Layer.CreateNew(str(out))
    Sdf.CopySpec(base, f"/{char}", new, f"/{char}")

    geom = new.GetPrimAtPath(f"/{char}/geometry")
    geom.referenceList.prependedItems.clear()
    geom.referenceList.prependedItems.append(
        Sdf.Reference(f"./assets/{char}/{char}.rt_{motion}.usda"))
    cd = dict(geom.customData)
    cd["rt_motion"] = {
        "motion": motion, "mocap_bvh": Path(manifest["bvh"]).name,
        "rigged_fbx": Path(manifest["fbx"]).name,
        "rigger": "UniRig (EXP1 자동리깅) + direction-copy 리타게팅",
        "frames": nframes, "fps": fps, "stretch_p95": manifest["p95"],
    }
    geom.customData = cd
    new.Save()

    stage = Usd.Stage.Open(str(out))
    stage.SetDefaultPrim(stage.GetPrimAtPath(f"/{char}"))
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(nframes)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().Save()
    return out


def validate(wrapper: Path, movie: str, char: str) -> str:
    stage = Usd.Stage.Open(str(wrapper))
    anims = [(p, len(UsdSkel.Animation(p).GetRotationsAttr().GetTimeSamples()))
             for p in stage.Traverse() if p.GetTypeName() == "SkelAnimation"]
    texs = [str(p.GetAttribute("inputs:file").Get().resolvedPath)
            for p in stage.Traverse() if p.GetTypeName() == "Shader"
            and p.GetAttribute("inputs:file") and p.GetAttribute("inputs:file").Get()]
    tex_ok = all(Path(t).exists() for t in texs) and texs

    # bbox 는 Mesh prim 기준 — SkelRoot 의 extent 힌트는 전체 애니 범위(루트 모션 포함)라
    # 원본 대비 비교에 못 쓴다.
    mesh = next(p for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode(1), ["default", "render"])
    rng = bbox.ComputeWorldBound(mesh).ComputeAlignedRange()
    size = Gf.Vec3d(rng.GetSize())

    base_geo = MOVIE_USD / movie / "characters" / "assets" / char / f"{char}.geometry.usda"
    bstage = Usd.Stage.Open(str(base_geo))
    brng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]) \
        .ComputeWorldBound(bstage.GetDefaultPrim()).ComputeAlignedRange()
    bsize = Gf.Vec3d(brng.GetSize())
    ratio = max(size[i] / bsize[i] for i in range(3) if bsize[i] > 1e-9)
    return (f"anim {anims[0][1] if anims else 0}샘플 | tex {'OK' if tex_ok else 'MISS'} "
            f"| bbox 원본대비 {ratio:.2f}x")


def main():
    for layer in sorted(MIRROR.glob("*/characters/assets/*/*.rt_*.usda")):
        movie = layer.relative_to(MIRROR).parts[0]
        char = layer.parent.name
        motion = layer.stem.split(".rt_")[1]
        mf = json.loads(layer.with_suffix(".manifest.json").read_text())
        fix_scale_and_units(layer, char, mf)
        if not repath_texture(layer, char):
            print(f"⚠️ {layer.name}: bin/texture.jpg 재지정 실패 (원본 텍스처 없음?)")
        wrapper = make_wrapper(movie, char, motion, mf["nframes"], mf["fps"], mf)
        print(f"[pack] {wrapper.relative_to(MIRROR)} — {validate(wrapper, movie, char)}")


if __name__ == "__main__":
    main()
