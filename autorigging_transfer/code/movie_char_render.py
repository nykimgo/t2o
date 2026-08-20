"""movie_usd 캐릭터: 텍스처 입힌 정면 스틸 + 리타게팅 모션의 텍스처 3/4 렌더.

두 산출물:
  1) <stem>_front.png        — 원본 에셋(텍스처) 정면 스틸 (rest pose)
  2) <stem>__rt_<m>_tex.webp — 리타게팅 모션을 텍스처/셰이딩까지 입혀 3/4 뷰로 렌더한 클립

핵심: UniRig 리깅 메시는 리메시되어 **UV가 없다**(텍스처 못 입힘). 그래서 리깅 메시의
스킨 웨이트를 원본(UV+텍스처) 메시로 최근접 이웃 보간 전이한 뒤 같은 armature에 바인딩한다.
UniRig 는 메시를 단위큐브로 정규화하므로(약 2배) bbox 기반 정렬이 선행된다.

렌더는 Cycles GPU (headless EEVEE 는 소프트웨어 폴백으로 ~18s/frame — 사용 불가).

실행 (unirig env):
  /home/sr/miniconda3/envs/unirig/bin/python movie_char_render.py [--only 철수] [--motions walk]
"""
from __future__ import annotations
import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import riglib as M          # noqa: E402
import retarget as RT       # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE / "movie_chars"
RIGS = ROOT / "rigs"
MOTION_OUT = ROOT / "motion_out"
RENDER_OUT = ROOT / "renders"
MOCAP = HERE / "mocap"
MOVIE_USD = Path("/home/sr/previs_proj/movie_usd")

MOTIONS = [("walk", MOCAP / "16_15.bvh"), ("run", MOCAP / "16_35.bvh"), ("jump", MOCAP / "16_01.bvh")]
CHARS = [
    ("berlin", "char_철수_base"), ("berlin", "char_영철_base"),
    ("hidden_time", "char_도균_base"), ("hidden_time", "char_수린_base"),
    ("welcom2dmk", "char_동구_base"), ("welcom2dmk", "char_여일_base"),
]

AZ_FRONT = 0.0     # 정면 (캐릭터는 -Y 정면)
AZ_TQ = 35.0       # 3/4 뷰 — riglib.render_frames 의 3/4 각과 동일
EL_CAM = 9.0       # 살짝 내려다보기 — 바닥/접지가 보이게
RES_STILL = (460, 620)
RES_MOTION = (330, 440)
SAMPLES = 28
FRAME_STEP = 2     # 64f 중 32f 만 렌더(재생 길이는 duration 2배로 동일 유지)
BG_TOP, BG_BOT = (0x6f, 0x76, 0x80), (0x4a, 0x4f, 0x57)


# ------------------------------------------------------------------ 씬 기본
def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == "CUDA")


def setup_render(res):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True          # 배경은 PIL 에서 합성
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    # AgX/Filmic 은 알베도 텍스처를 탈색시킴 → Standard
    try:
        sc.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    world = bpy.data.worlds.new("W") if not bpy.data.worlds else bpy.data.worlds[0]
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.30, 0.31, 0.34, 1.0)   # 부드러운 앰비언트 돔
        bg.inputs[1].default_value = 0.55


def cam_basis(az_deg, el_deg=0.0):
    az = math.radians(az_deg); el = math.radians(el_deg)
    d = Vector((math.sin(az) * math.cos(el),            # center → camera
                -math.cos(az) * math.cos(el),
                math.sin(el)))
    r = Vector((math.cos(az), math.sin(az), 0.0))       # 화면 수평
    u = r.cross(d).normalized()                         # 화면 수직
    if u.z < 0:
        u = -u
    return d, r, u


def frame_camera(verts, az_deg, res, margin=1.14, el_deg=0.0):
    """점군 전체가 항상 화면에 들어오는 정사영 카메라 배치(프레임 간 흔들림 없음)."""
    d, r, u = cam_basis(az_deg, el_deg)
    P = np.asarray(verts, dtype=np.float64)
    hr = P @ np.array(r); hu = P @ np.array(u); hd = P @ np.array(d)
    c_r = (hr.min() + hr.max()) / 2
    c_u = (hu.min() + hu.max()) / 2
    c_d = (hd.min() + hd.max()) / 2
    center = Vector(np.array(r) * c_r + np.array(u) * c_u + np.array(d) * c_d)
    ext_r = float(hr.max() - hr.min()); ext_u = float(hu.max() - hu.min())
    res_x, res_y = res
    # ortho_scale 은 이미지의 더 긴 변에 대응
    if res_y >= res_x:
        scale = max(ext_u, ext_r * res_y / res_x)
    else:
        scale = max(ext_r, ext_u * res_x / res_y)
    dist = max(ext_r, ext_u) * 4 + 2
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = scale * margin
    cam_data.clip_start = 0.01
    cam_data.clip_end = dist * 4
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = center + d * dist
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return center, max(ext_r, ext_u)


def add_lights(center, size, az_deg):
    """카메라 방위 기준 3점 조명 — 어두운 의상도 실루엣/입체가 읽히게."""
    def area(name, az_off, elev, energy, sz):
        az = math.radians(az_deg + az_off)
        el = math.radians(elev)
        d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = energy
        ld.size = sz
        ob = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(ob)
        dist = size * 2.2 + 1.0
        ob.location = Vector(center) + d * dist
        ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        return ob
    s = max(size, 0.5)
    area("key",  -38, 34, 480 * s * s, s * 0.55)   # 작게 → 접지 그림자가 보이게
    area("fill",  52, 8,  150 * s * s, s * 2.0)
    area("rim",  168, 42, 380 * s * s, s * 1.4)


# ------------------------------------------------------------------ 웨이트 전이
def biggest_mesh(objs):
    return max((o for o in objs if o.type == "MESH"), key=lambda o: len(o.data.vertices))


def world_verts(ob):
    mw = ob.matrix_world
    return np.array([(mw @ v.co)[:] for v in ob.data.vertices], dtype=np.float64)


def align_and_transfer(src_ob, rig_ob, arm, k=4):
    """원본(UV) 메시를 리깅 메시 공간으로 정렬하고 스킨 웨이트를 최근접 보간 전이.

    UniRig 가 메시를 단위큐브로 정규화하므로 bbox 중심/균일스케일로 맞춘 뒤,
    각 원본 정점에서 가까운 리깅 정점 k개의 본 웨이트를 역거리 가중 평균한다.
    """
    S = world_verts(src_ob)
    R = world_verts(rig_ob)
    c_s, c_r = (S.min(0) + S.max(0)) / 2, (R.min(0) + R.max(0)) / 2
    ext_s, ext_r = S.max(0) - S.min(0), R.max(0) - R.min(0)
    scale = float(np.median(ext_r / np.maximum(ext_s, 1e-9)))
    Sr = (S - c_s) * scale + c_r          # 리깅 월드 공간으로

    # 리깅 메시 정점별 본 웨이트
    vg_names = [g.name for g in rig_ob.vertex_groups]
    wmap = [dict() for _ in range(len(R))]
    for vi, v in enumerate(rig_ob.data.vertices):
        for g in v.groups:
            if g.weight > 1e-4:
                wmap[vi][vg_names[g.group]] = g.weight

    tree = KDTree(len(R))
    for i, p in enumerate(R):
        tree.insert(Vector(p), i)
    tree.balance()

    # 원본 메시에 같은 이름의 vertex group 생성
    src_ob.vertex_groups.clear()
    groups = {n: src_ob.vertex_groups.new(name=n) for n in vg_names}

    for si, p in enumerate(Sr):
        hits = tree.find_n(Vector(p), k)
        acc, tot = {}, 0.0
        for _co, idx, dist in hits:
            w = 1.0 / (dist + 1e-6)
            tot += w
            for bn, bw in wmap[idx].items():
                acc[bn] = acc.get(bn, 0.0) + bw * w
        if tot <= 0:
            continue
        for bn, bw in acc.items():
            val = bw / tot
            if val > 1e-4:
                groups[bn].add([si], min(1.0, val), "REPLACE")

    # 정렬된 좌표를 리깅 메시의 로컬 공간으로 넣고 같은 변환/armature 를 부여
    inv = rig_ob.matrix_world.inverted()
    for vi, v in enumerate(src_ob.data.vertices):
        v.co = inv @ Vector(Sr[vi])
    src_ob.matrix_world = rig_ob.matrix_world.copy()

    for m in list(src_ob.modifiers):
        src_ob.modifiers.remove(m)
    mod = src_ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    src_ob.parent = arm
    return src_ob


# ------------------------------------------------------------------ 합성/저장
def composite(png_path, res):
    from PIL import Image
    fg = Image.open(png_path).convert("RGBA")
    w, h = fg.size
    bg = Image.new("RGB", (w, h))
    top = np.array(BG_TOP, dtype=np.float64); bot = np.array(BG_BOT, dtype=np.float64)
    ramp = np.linspace(0, 1, h)[:, None]
    col = (top[None, :] * (1 - ramp) + bot[None, :] * ramp).astype(np.uint8)
    bg = Image.fromarray(np.repeat(col[:, None, :], w, axis=1))
    out = Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")
    return out


def render_current(tmpdir, tag, res):
    sc = bpy.context.scene
    p = Path(tmpdir) / f"{tag}.png"
    sc.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    return composite(p, res)


# ------------------------------------------------------------------ 산출물 1: 정면 스틸
def render_front(src_glb: Path, out_png: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(src_glb))
    mesh = biggest_mesh(bpy.data.objects)
    setup_render(RES_STILL)
    V = world_verts(mesh)
    center, size = frame_camera(V, AZ_FRONT, RES_STILL, margin=1.07, el_deg=EL_CAM)
    add_lights(center, size, AZ_FRONT)
    with tempfile.TemporaryDirectory() as td:
        img = render_current(td, "front", RES_STILL)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, optimize=True)
    return out_png


# ------------------------------------------------------------------ 산출물 2: 텍스처 모션
def render_motion_textured(fbx: Path, bvh: Path, src_glb: Path, out_webp: Path,
                           fps=30, start=60, maxframes=64, step=FRAME_STEP):
    # 검증된 리타게팅 로직 재사용(수치 동일) — 씬에 애니메이션이 얹힌 상태로 돌아온다
    r = RT.build_animation(fbx, bvh, fps=fps, start=start, maxframes=maxframes)
    arm, rig_mesh, frames = r["tgt"], r["mesh"], r["frames"]

    bpy.ops.import_scene.gltf(filepath=str(src_glb))
    src_ob = max((o for o in bpy.data.objects
                  if o.type == "MESH" and o is not rig_mesh and len(o.data.uv_layers) > 0),
                 key=lambda o: len(o.data.vertices))
    align_and_transfer(src_ob, rig_mesh, arm)

    rig_mesh.hide_render = True
    for o in bpy.data.objects:            # BVH armature 등 잡오브젝트 렌더 제외
        if o.type == "ARMATURE":
            o.hide_render = True

    setup_render(RES_MOTION)
    allv = np.concatenate([f[0] for f in frames], 0)
    center, size = frame_camera(allv, AZ_TQ, RES_MOTION, margin=1.06, el_deg=EL_CAM)
    add_lights(center, size, AZ_TQ)

    keep = list(range(1, len(frames) + 1, step))
    imgs = []
    with tempfile.TemporaryDirectory() as td:
        for f in keep:
            bpy.context.scene.frame_set(f)
            imgs.append(render_current(td, f"f{f:04d}", RES_MOTION))
    out_webp.parent.mkdir(parents=True, exist_ok=True)
    # 원본 GIF(64f@60ms)와 재생 시간 동일하게 유지
    imgs[0].save(out_webp, save_all=True, append_images=imgs[1:],
                 duration=60 * step, loop=0, quality=64, method=5)
    return out_webp, len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="캐릭터 이름 부분일치 필터")
    ap.add_argument("--motions", default="", help="쉼표구분 모션 필터")
    ap.add_argument("--skip-front", action="store_true")
    ap.add_argument("--skip-motion", action="store_true")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

    enable_gpu()
    RENDER_OUT.mkdir(parents=True, exist_ok=True)
    motions = [m for m in MOTIONS if not a.motions or m[0] in a.motions.split(",")]
    chars = [c for c in CHARS if not a.only or a.only in c[1]]

    for movie, char in chars:
        stem = f"{movie}__{char}"
        src_glb = MOVIE_USD / movie / "characters" / "assets" / char / f"{char}.glb"
        fbx = RIGS / f"{stem}_rigged.fbx"
        if not a.skip_front:
            out_png = RENDER_OUT / f"{stem}_front.png"
            if out_png.exists():
                print(f"skip front (exists): {stem}")
            else:
                render_front(src_glb, out_png)
                print(f"[front] {stem} -> {out_png.name} ({out_png.stat().st_size/1e3:.0f} KB)")
        if a.skip_motion:
            continue
        for name, bvh in motions:
            out_webp = RENDER_OUT / f"{stem}__rt_{name}_tex.webp"
            if out_webp.exists():
                print(f"skip motion (exists): {stem} {name}")
                continue
            _, n = render_motion_textured(fbx, bvh, src_glb, out_webp)
            print(f"[tex] {stem} {name} {n}f -> {out_webp.name} ({out_webp.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
