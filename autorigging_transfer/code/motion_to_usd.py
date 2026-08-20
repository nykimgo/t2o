"""movie_usd 캐릭터: 리깅+리타게팅 모션을 텍스처 포함 USD(UsdSkel)로 익스포트.

체인: retarget.build_animation(FBX+BVH → 씬에 애니메이션 적용, 검증된 로직 재사용)
      → 원본 GLB(UV+텍스처) 임포트 → movie_char_render.align_and_transfer 로
      웨이트 전이(텍스처 렌더와 동일 로직) → 잡오브젝트 제거 →
      bpy.ops.wm.usd_export(armature+animation+material+texture).

출력: <out>/<movie>__<char>__rt_<motion>.usdc + textures/ (같은 폴더에 텍스처 파일)
      --usdz 를 주면 .usdz 단일 파일도 추가로 생성.

실행 (unirig env — bpy 4.2; USD 스켈레탈 익스포트는 4.1+):
  /home/sr/miniconda3/envs/unirig/bin/python motion_to_usd.py [--only 철수] [--motions walk]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retarget as RT                    # noqa: E402 — 검증된 리타게팅 (좌우반전 수정판)
import movie_char_render as MCR          # noqa: E402 — align_and_transfer 재사용

HERE = Path(__file__).resolve().parent
ROOT = HERE / "movie_chars"
OUT = ROOT / "usd_out"


def export_usd(fbx: Path, bvh: Path, src_glb: Path, out_usd: Path, usdz: bool):
    r = RT.build_animation(fbx, bvh)
    arm, rig_mesh, frames = r["tgt"], r["mesh"], r["frames"]

    bpy.ops.import_scene.gltf(filepath=str(src_glb))
    src_ob = max((o for o in bpy.data.objects
                  if o.type == "MESH" and o is not rig_mesh and len(o.data.uv_layers) > 0),
                 key=lambda o: len(o.data.vertices))
    MCR.align_and_transfer(src_ob, rig_mesh, arm)

    # USD 에는 텍스처 메시 + 타깃 armature 만 남긴다
    # (리깅 리메시·소스 BVH armature·glb 잔여 오브젝트 제거)
    keep = {src_ob, arm}
    for o in [o for o in bpy.data.objects if o not in keep]:
        bpy.data.objects.remove(o, do_unlink=True)

    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, len(frames)

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(export_animation=True, export_armatures=True,
                  export_materials=True, export_textures=True,
                  overwrite_textures=True)
    bpy.ops.wm.usd_export(filepath=str(out_usd), **kwargs)
    made = [out_usd]
    if usdz:
        z = out_usd.with_suffix(".usdz")
        bpy.ops.wm.usd_export(filepath=str(z), **kwargs)
        made.append(z)
    return made, len(frames), r["p95"]


def export_movie_usd_layer(fbx: Path, bvh: Path, src_glb: Path, out_layer: Path,
                           char: str):
    """movie_usd 규칙용 애셋 레이어(.usda) — Y-up 변환 + 원본 glb 스케일 복원.

    align_and_transfer 는 원본 메시를 UniRig 정규화 공간(단위큐브)으로 옮기므로,
    같은 파라미터(c_s·c_r·scale)를 역적용해 geometry.usda 와 같은 배치로 되돌린다.
    """
    import json

    import numpy as np

    r = RT.build_animation(fbx, bvh)
    arm, rig_mesh, frames = r["tgt"], r["mesh"], r["frames"]

    bpy.ops.import_scene.gltf(filepath=str(src_glb))
    src_ob = max((o for o in bpy.data.objects
                  if o.type == "MESH" and o is not rig_mesh and len(o.data.uv_layers) > 0),
                 key=lambda o: len(o.data.vertices))

    # align_and_transfer 와 동일 수식으로 정규화 파라미터 캡처 (역변환용)
    S = MCR.world_verts(src_ob)
    R = MCR.world_verts(rig_mesh)
    c_s = (S.min(0) + S.max(0)) / 2
    c_r = (R.min(0) + R.max(0)) / 2
    scale = float(np.median((R.max(0) - R.min(0)) /
                            np.maximum(S.max(0) - S.min(0), 1e-9)))

    MCR.align_and_transfer(src_ob, rig_mesh, arm)

    keep = {src_ob, arm}
    for o in [o for o in bpy.data.objects if o not in keep]:
        bpy.data.objects.remove(o, do_unlink=True)

    # 주의: 원본 배치 복원은 여기서 하지 않는다 — bpy USD 익스포터가 armature 오브젝트
    # 트랜스폼을 SkelRoot 에 반영하지 않음(실측). 레이어는 UniRig 정규화 공간 그대로
    # 내보내고, c_s·c_r·scale 을 manifest 로 넘겨 movie_usd_pack.py 가 USD 쪽에서
    # 보정 행렬(+m→cm)을 루트 prim 에 넣는다.

    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, len(frames)

    out_layer.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(out_layer),
        export_animation=True, export_armatures=True,
        export_materials=True, export_textures=True, overwrite_textures=True,
        convert_orientation=True,          # 기본 forward=-Z, up=Y → geometry.usda 와 동일 Y-up
        root_prim_path=f"/{char}",
    )
    out_layer.with_suffix(".manifest.json").write_text(json.dumps(
        {"nframes": len(frames), "p95": round(r["p95"], 3), "fps": r["fps"],
         "fbx": str(fbx), "bvh": str(bvh),
         "c_s": [float(v) for v in c_s], "c_r": [float(v) for v in c_r],
         "scale": scale}, ensure_ascii=False))
    return len(frames), r["p95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="캐릭터 이름 부분일치 필터")
    ap.add_argument("--motions", default="", help="쉼표구분 모션 필터 (walk,run,jump)")
    ap.add_argument("--usdz", action="store_true", help=".usdz 단일 파일도 생성")
    ap.add_argument("--movie-usd", action="store_true",
                    help="movie_usd 규칙 미러 트리(usd_movie/)에 .usda 애셋 레이어 생성 "
                         "(이후 movie_usd_pack.py 로 래퍼·텍스처 정리)")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

    motions = [m for m in MCR.MOTIONS if not a.motions or m[0] in a.motions.split(",")]
    chars = [c for c in MCR.CHARS if not a.only or a.only in c[1]]

    for movie, char in chars:
        stem = f"{movie}__{char}"
        src_glb = MCR.MOVIE_USD / movie / "characters" / "assets" / char / f"{char}.glb"
        fbx = MCR.RIGS / f"{stem}_rigged.fbx"
        for name, bvh in motions:
            if a.movie_usd:
                out_layer = (ROOT / "usd_movie" / movie / "characters" / "assets"
                             / char / f"{char}.rt_{name}.usda")
                if out_layer.exists():
                    print(f"skip (exists): {out_layer.name}")
                    continue
                nf, p95 = export_movie_usd_layer(fbx, bvh, src_glb, out_layer, char)
                print(f"[usda] {stem} {name} {nf}f p95={p95:.2f} -> {out_layer}")
                continue
            out_usd = OUT / f"{stem}__rt_{name}.usdc"
            if out_usd.exists():
                print(f"skip (exists): {out_usd.name}")
                continue
            made, nf, p95 = export_usd(fbx, bvh, src_glb, out_usd, a.usdz)
            for m in made:
                print(f"[usd] {stem} {name} {nf}f p95={p95:.2f} -> "
                      f"{m.name} ({m.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
