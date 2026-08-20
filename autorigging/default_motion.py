"""클래스별 기본 모션 (DESIGN §4-2) — 전부 제자리(in-place), 클래스당 1종 고정.

- biped: 합성 클립 1개 — 걷기(2초 상당) → 뛰기(26f 루프로 2초) → 점프 1회.
  CMU BVH 리타게팅(검증본 retarget.build_animation 재사용, map_clavicle=False 유지).
  클립 경계는 0.25s 포즈 블렌드로 연결, 전체를 단일 액션으로 베이크.
- 비-biped: 절차적 관절 가동 시연 (motion_grade="demo") — 본 체인 토폴로지만 사용,
  체인 분류 실패 시 전체 본 사인 웨이브 폴백. 파라미터는 PROC_PARAMS 상수 테이블.

모든 빌더는 (씬을 새로 구성하고) 다음 dict 를 반환한다:
  {tgt, mesh, nframes, fps, p95, motion, motion_grade, clips}
tgt=애니메이션이 베이크된 armature, mesh=리깅 메시. 이후 usd_assemble 이 소비한다.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import bpy
from mathutils import Quaternion, Vector

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retarget as RT          # noqa: E402 — 검증된 리타게팅 (좌우반전 수정판)

MOCAP = Path(__file__).resolve().parent / "assets" / "mocap"
FPS = 30
BLEND_SEC = 0.25               # 클립 경계 포즈 블렌드
TWO_SEC = 2 * FPS

# biped 합성 클립 소스 (DESIGN §4-2 표)
BIPED_CLIPS = [
    # (이름, bvh, start(소스frame), maxframes(출력), loop_to(부족 시 루프 목표))
    ("walk", MOCAP / "16_15.bvh", 60, TWO_SEC, None),      # 472f — 2초 발췌
    ("run",  MOCAP / "16_35.bvh", 60, 120,     TWO_SEC),   # 26f 뿐 → 2초 되도록 루프
    ("jump", MOCAP / "16_01.bvh", 0,  240,     None),      # 점프 1회 — 피크 주변 윈도우
]
JUMP_WINDOW = 45               # 점프 피크 전후 프레임 수 (총 ~3초)

# 비-biped 절차 모션 파라미터 테이블 (튜닝 여지를 코드 밖으로 — DESIGN §4-2-1)
PROC_PARAMS = {
    "quadruped":  dict(kind="legs",  amp_deg=22.0, freq_hz=1.2, bounce=0.015),
    "hexapod":    dict(kind="legs",  amp_deg=18.0, freq_hz=1.8, bounce=0.008),
    "octopod":    dict(kind="legs",  amp_deg=18.0, freq_hz=1.2, bounce=0.008),
    "avian":      dict(kind="wings", amp_deg=38.0, freq_hz=2.2, bounce=0.030),
    "serpentine": dict(kind="spine", amp_deg=24.0, freq_hz=1.0, bounce=0.0),
    "aquatic":    dict(kind="tail",  amp_deg=20.0, freq_hz=1.5, bounce=0.0),
}
PROC_NFRAMES = 2 * TWO_SEC     # 4초 시연


# ---------------------------------------------------------------- 공통: 포즈 샘플/적용
def _sample_poses(tgt, root_name, nframes):
    """씬에 베이크된 애니메이션에서 프레임별 (본별 쿼터니언, 루트 위치, 루트 월드Z) 샘플."""
    sc = bpy.context.scene
    out = []
    for f in range(1, nframes + 1):
        sc.frame_set(f)
        bpy.context.view_layer.update()
        quats = {pb.name: pb.rotation_quaternion.copy() for pb in tgt.pose.bones}
        rb = tgt.pose.bones[root_name]
        root_z = (tgt.matrix_world @ rb.matrix).to_translation().z
        out.append({"q": quats, "loc": rb.location.copy(), "z": root_z})
    return out


def _blend_pose(a, b, t):
    """포즈 a→b 를 t(0..1)로 블렌드 (쿼터니언 slerp, 루트 lerp)."""
    q = {n: a["q"][n].slerp(b["q"][n], t) for n in a["q"]}
    loc = a["loc"].lerp(b["loc"], t)
    return {"q": q, "loc": loc, "z": a["z"] * (1 - t) + b["z"] * t}


def _apply_and_bake(fbx, poses, root_name, fps=FPS):
    """FBX 를 새 씬에 로드하고 포즈 시퀀스를 단일 액션으로 베이크."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    tgt = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    tgt.animation_data_clear()
    sc = bpy.context.scene
    for i, pose in enumerate(poses, start=1):
        for pb in tgt.pose.bones:
            pb.rotation_mode = "QUATERNION"
            if pb.name in pose["q"]:
                pb.rotation_quaternion = pose["q"][pb.name]
            pb.keyframe_insert("rotation_quaternion", frame=i)
        rb = tgt.pose.bones[root_name]
        rb.location = pose["loc"]
        rb.keyframe_insert("location", frame=i)
    sc.frame_start, sc.frame_end = 1, len(poses)
    sc.render.fps = fps
    sc.frame_set(1)
    bpy.context.view_layer.update()
    return tgt, mesh


# ---------------------------------------------------------------- biped 합성 클립
def build_biped_composite(fbx: str | Path) -> dict:
    """걷기→뛰기→점프 합성 클립을 단일 액션으로 베이크해 씬에 구성."""
    fbx = Path(fbx)
    segments = []          # (이름, poses)
    p95 = 0.0
    root_name = None
    clip_info = []

    for name, bvh, start, maxframes, loop_to in BIPED_CLIPS:
        r = RT.build_animation(fbx, bvh, fps=FPS, start=start, maxframes=maxframes)
        root_name = r["roles"]["root"]
        poses = _sample_poses(r["tgt"], root_name, r["nframes"])
        p95 = max(p95, r["p95"])

        if name == "jump":
            # 점프 1회: 루트 월드Z 피크 주변 윈도우만 사용
            zs = [p["z"] for p in poses]
            peak = int(np.argmax(zs))
            lo = max(0, peak - JUMP_WINDOW)
            hi = min(len(poses), peak + JUMP_WINDOW)
            poses = poses[lo:hi]
        if loop_to and len(poses) < loop_to:
            # run 26f 한계 → 목표 길이까지 루프 반복 (마지막 키프레임 클램프는
            # retarget.build_animation 에 반영돼 있어 정지 중복 프레임이 없다)
            base = list(poses)
            while len(poses) < loop_to:
                poses.extend(base)
            poses = poses[:loop_to]

        segments.append((name, poses))
        clip_info.append({"clip": name, "bvh": bvh.name, "frames": len(poses)})

    # 경계 0.25s 포즈 블렌드로 연결
    nblend = max(2, round(BLEND_SEC * FPS))
    merged = list(segments[0][1])
    for _name, poses in segments[1:]:
        a, b = merged[-1], poses[0]
        for k in range(1, nblend + 1):
            merged.append(_blend_pose(a, b, k / (nblend + 1)))
        merged.extend(poses)

    tgt, mesh = _apply_and_bake(fbx, merged, root_name)
    return {"tgt": tgt, "mesh": mesh, "nframes": len(merged), "fps": FPS,
            "p95": p95, "motion": "default_biped_v1", "motion_grade": "verified",
            "clips": clip_info}


# ---------------------------------------------------------------- 비-biped 절차 모션
def _chains(arm):
    """루트→리프 본 체인 분해 (토폴로지만 사용 — riglib.detect 는 biped 전제라 미사용)."""
    bones = {b.name: b for b in arm.data.bones}
    parent = {n: (b.parent.name if b.parent else None) for n, b in bones.items()}
    children = {n: [c.name for c in b.children] for n, b in bones.items()}
    root = next(n for n, p in parent.items() if p is None)
    leaves = [n for n in bones if not children[n]]

    def path(leaf):
        c, out = leaf, []
        while c is not None and c != root:
            out.append(c)
            c = parent[c]
        out.reverse()
        return out

    tips = {n: np.array(bones[n].tail_local) for n in leaves}
    return root, [(path(lf), tips[lf]) for lf in leaves], bones


def _classify_chains(arm, kind):
    """체인 길이·끝점 좌표 휴리스틱으로 다리/날개/스파인 분류. 실패 시 [] 반환."""
    root, chains, bones = _chains(arm)
    if not chains:
        return root, []
    zs = [tip[2] for _ch, tip in chains]
    z_lo, z_hi = min(zs), max(zs)

    if kind == "legs":
        # 다리 = 끝점 z 가 하위 40% 대역에 있는 체인들 (2개 이상일 때만 인정)
        thr = z_lo + 0.4 * max(1e-9, z_hi - z_lo)
        legs = [ch for ch, tip in chains if tip[2] <= thr and len(ch) >= 2]
        return root, legs if len(legs) >= 2 else []
    if kind == "wings":
        # 날개 = |x| 최대 좌/우 체인 1개씩
        left = [(ch, tip) for ch, tip in chains if tip[0] < 0]
        right = [(ch, tip) for ch, tip in chains if tip[0] >= 0]
        wings = []
        if left:
            wings.append(max(left, key=lambda ct: abs(ct[1][0]))[0])
        if right:
            wings.append(max(right, key=lambda ct: abs(ct[1][0]))[0])
        wings = [w for w in wings if len(w) >= 2]
        return root, wings if len(wings) == 2 else []
    if kind in ("spine", "tail"):
        # 스파인/꼬리 = 가장 긴 체인 1개
        longest = max(chains, key=lambda ct: len(ct[0]))[0]
        return root, [longest] if len(longest) >= 3 else []
    return root, []


def build_procedural(fbx: str | Path, rig_type: str) -> dict:
    """비-biped 절차 기본 모션 — 리깅이 살아있음을 보여주는 관절 가동 시연 (demo)."""
    fbx = Path(fbx)
    params = PROC_PARAMS.get(rig_type)
    if params is None:
        raise ValueError(f"절차 모션 미정의 rig_type: {rig_type}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    tgt = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")

    root, chains = _classify_chains(tgt, params["kind"])
    fallback = not chains
    if fallback:
        # 폴백: 전체 본 사인 웨이브 (깊이 위상)
        bones = {b.name: b for b in tgt.data.bones}
        parent = {n: (b.parent.name if b.parent else None) for n, b in bones.items()}

        def depth(n):
            d = 0
            while parent[n]:
                n = parent[n]
                d += 1
            return d
        chains = [[n] for n in bones if n != root]
        depths = {n: depth(n) for n in bones}

    amp = math.radians(params["amp_deg"])
    freq = params["freq_hz"]
    bounce = params["bounce"]

    tgt.animation_data_clear()
    sc = bpy.context.scene
    rb = tgt.pose.bones[root]
    for pb in tgt.pose.bones:
        pb.rotation_mode = "QUATERNION"

    for f in range(1, PROC_NFRAMES + 1):
        t = (f - 1) / FPS
        w = 2 * math.pi * freq * t
        for ci, ch in enumerate(chains):
            # 다리: 좌우/전후 위상차(파상 보행), 스파인: 본 순차 위상(사행)
            phase = ci * (math.pi if params["kind"] in ("legs", "wings")
                          else 2 * math.pi / max(1, len(chains)))
            for bi, bn in enumerate(ch):
                if fallback:
                    ang = amp * 0.5 * math.sin(w + depths.get(bn, 0) * 0.7)
                elif params["kind"] in ("spine", "tail"):
                    ang = amp * math.sin(w + bi * 0.9)          # 순차 위상 사행
                else:
                    falloff = 1.0 / (1 + 0.5 * bi)              # 근위 크게, 원위 작게
                    ang = amp * falloff * math.sin(w + phase + bi * 0.4)
                pb = tgt.pose.bones[bn]
                pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0), ang)
                pb.keyframe_insert("rotation_quaternion", frame=f)
        if bounce:
            rb.location = Vector((0.0, 0.0, bounce * abs(math.sin(w))))
            rb.keyframe_insert("location", frame=f)

    sc.frame_start, sc.frame_end = 1, PROC_NFRAMES
    sc.render.fps = FPS
    sc.frame_set(1)
    bpy.context.view_layer.update()
    return {"tgt": tgt, "mesh": mesh, "nframes": PROC_NFRAMES, "fps": FPS,
            "p95": None, "motion": f"default_{rig_type}_v1", "motion_grade": "demo",
            "clips": [{"clip": params["kind"], "procedural": True,
                       "fallback": fallback, **{k: v for k, v in params.items()}}]}


def build(fbx: str | Path, rig_type: str) -> dict:
    if rig_type == "biped":
        return build_biped_composite(fbx)
    return build_procedural(fbx, rig_type)
