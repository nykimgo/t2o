"""EXP1 부가 — B: 무료 CMU BVH 모캡을 UniRig 자동리깅 스켈레톤에 리타게팅.

익명 본 → 토폴로지 감지(motion_test.detect)로 역할 파악 → 표준 CMU 관절명과 역할 매핑.
리타게팅 = rest 기준 월드 델타 전이:  R_t_pose = (R_s_pose · R_s_rest⁻¹) · R_t_rest.
본 로컬축/roll 무관, T-pose 차이 자동 흡수. 두 스켈레톤 모두 Z-up·-Y정면이라 월드 프레임 정렬됨.

실행 (unirig env):
  /home/sr/miniconda3/envs/unirig/bin/python retarget.py \
     --fbx <rigged.fbx> --bvh mocap/16_15.bvh --name walk --out motion_out
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix, Vector, Quaternion

import riglib as M   # detect, render_frames, eval_verts, bone_segs, get_edges, edge_stretch

# CMU(=BVH) 표준 관절명. 역할→CMU
CMU = {
    "root": "Hips",
    "legR_thigh": "RightUpLeg", "legR_shin": "RightLeg", "legR_foot": "RightFoot",
    "legL_thigh": "LeftUpLeg",  "legL_shin": "LeftLeg",  "legL_foot": "LeftFoot",
    "armR_shoulder": "RightShoulder", "armR_upper": "RightArm", "armR_fore": "RightForeArm",
    "armL_shoulder": "LeftShoulder",  "armL_upper": "LeftArm",  "armL_fore": "LeftForeArm",
    "head": "Head",
}
CMU_SPINE = ["LowerBack", "Spine", "Spine1"]

def build_map(roles, src_bones, map_clavicle=False):
    """타깃(UniRig) 본 이름 -> 소스(CMU) 관절명.

    map_clavicle=False (기본): 쇄골(shoulder)은 매핑하지 않는다. 쇄골 비율/방향은 리그마다
    크게 달라(타깃 A-pose 쇄골은 아래로 -0.38, CMU 는 위로 +0.32) 방향을 강제하면 어깨
    관절이 통째로 들려 삼각근/겨드랑이 메시가 찢어진다. 표준 리타게팅에서도 쇄골은 보통 제외.
    """
    m = {}
    m[roles["root"]] = "Hips"
    parts_arm = ("shoulder", "upper", "fore") if map_clavicle else ("upper", "fore")
    for lr in ("R", "L"):
        leg = roles[f"leg{lr}"]; arm = roles[f"arm{lr}"]
        for part in ("thigh", "shin", "foot"):
            j = CMU.get(f"leg{lr}_{part}")
            if j in src_bones: m[leg[part]] = j
        for part in parts_arm:
            j = CMU.get(f"arm{lr}_{part}")
            if j in src_bones and arm.get(part): m[arm[part]] = j
    # spine: UniRig spine(머리 제외) → CMU_SPINE 비례 매핑, head → Head
    sp = [b for b in roles["spine"] if b != roles["head"]]
    avail = [j for j in CMU_SPINE if j in src_bones]
    for i, b in enumerate(sp):
        if avail:
            k = min(len(avail)-1, round(i/max(1, len(sp)-1)*(len(avail)-1)))
            m[b] = avail[k]
    if "Head" in src_bones: m[roles["head"]] = "Head"
    return m

def rot3(mat):  # 4x4/3x3 → 정규직교 3x3 회전
    return mat.to_3x3().normalized() if hasattr(mat, "to_3x3") else mat.normalized()

def build_animation(fbx, bvh, fps=30, start=60, maxframes=64, map_clavicle=False):
    """BVH → 익명본 리타게팅을 씬에 적용하고 결과를 dict 로 반환.

    씬을 초기화(read_factory_settings)하므로, 호출 후 씬에는 리타게팅된 애니메이션이
    키프레임으로 얹힌 타깃 armature(tgt)와 스킨 메시(mesh), 소스 armature(src)가 있다.
    렌더/익스포트 없이 애니메이션만 필요할 때(예: 텍스처 렌더) 이 함수를 직접 쓴다.

    반환: {tgt, mesh, src, roles, frames, nframes, max, p95, ratio, fps}
      frames = [(verts(N,3), bone_segs)] — 프레임별 변형 메시/본 (GIF 렌더용)
    """
    class _A:  # 기존 argparse 인터페이스 유지용
        pass
    a = _A()
    a.fbx, a.bvh, a.fps, a.start, a.maxframes = str(fbx), str(bvh), fps, start, maxframes

    fbx = Path(a.fbx)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 1) 소스 BVH (Z-up, -Y정면, m스케일)
    bpy.ops.import_anim.bvh(filepath=a.bvh, axis_forward='-Z', axis_up='Y',
                            global_scale=0.056, update_scene_fps=True, update_scene_duration=True)
    src = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    src.name = "SRC"
    sc = bpy.context.scene
    src_bones = {b.name for b in src.data.bones}
    src_fps = sc.render.fps
    f0, f1 = sc.frame_start, sc.frame_end
    # 소스 액션의 실제 마지막 키프레임으로 클램프. BVH import 가 설정한 scene frame_end 가
    # 실제 모캡 길이보다 길 수 있어(16_35 run = 163프레임), 그대로 쓰면 뒷부분이 마지막
    # 포즈로 얼어붙은 중복 프레임이 된다.
    if src.animation_data and src.animation_data.action:
        last_key = max((k.co[0] for fc in src.animation_data.action.fcurves
                        for k in fc.keyframe_points), default=f1)
        f1 = min(f1, int(last_key))

    # 2) 타깃 FBX
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    tgt = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name != "SRC")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    roles = M.detect(tgt)
    tmap = build_map(roles, src_bones, map_clavicle=map_clavicle)
    print(f"[map] {len(tmap)} bones mapped:")
    for k, v in tmap.items(): print(f"    {k:9s} <- {v}")

    # rest 회전(월드) 사전계산
    sW = src.matrix_world; tW = tgt.matrix_world
    sWr = rot3(sW); tWr = rot3(tW)
    sWr_i = sWr.inverted(); tWr_i = tWr.inverted()
    R_s_rest = {b.name: rot3(sW @ b.matrix_local) for b in src.data.bones}
    R_t_rest_arm = {b.name: rot3(b.matrix_local) for b in tgt.data.bones}   # armature-space
    td_rest = {b.name: ((tW @ Vector(b.tail_local)) - (tW @ Vector(b.head_local))).normalized()
               for b in tgt.data.bones}                                     # 타깃 rest 방향(월드)
    ml = {b.name: b.matrix_local.copy() for b in tgt.data.bones}
    parent = {b.name: (b.parent.name if b.parent else None) for b in tgt.data.bones}
    order = topo([b.name for b in tgt.data.bones], parent)

    # 루트 이동 스케일: 타깃/소스 다리길이 비
    def leglen(arm_bones, hip, foot, W):
        return (Vector((W @ arm_bones[hip].matrix_local).to_translation())
                - Vector((W @ arm_bones[foot].matrix_local).to_translation())).length
    try:
        tb = {b.name: b for b in tgt.data.bones}; sb = {b.name: b for b in src.data.bones}
        ratio = leglen(tb, roles["legR"]["thigh"], roles["legR"]["foot"], tW) / \
                (leglen(sb, "RightUpLeg", "RightFoot", sW) + 1e-6)
    except Exception:
        ratio = 1.0
    hips_rest = Vector((sW @ sb["Hips"].matrix_local).to_translation())

    # 프레임 샘플링
    step = max(1, round(src_fps / a.fps))
    frames_idx = list(range(f0 + a.start, f1 + 1, step))[:a.maxframes]

    rest_v = M.eval_verts(mesh)
    edges = M.get_edges(mesh)
    rest_len = np.linalg.norm(rest_v[edges[:, 0]] - rest_v[edges[:, 1]], axis=1) + 1e-8

    frames, mx, p95 = [], 0.0, 0.0
    tgt.animation_data_clear()
    for out_i, f in enumerate(frames_idx):
        sc.frame_set(f)
        bpy.context.view_layer.update()
        # 방향 복사: 타깃 팔다리가 소스 팔다리와 같은 월드 방향을 향하게(트위스트 무시)
        dir_rot = {}
        for tb_name, sj in tmap.items():
            if tb_name == roles["root"]:
                continue                                     # 루트는 회전 복사 안 함(직립 유지)
            pbs = src.pose.bones[sj]
            ds = (sW @ pbs.tail) - (sW @ pbs.head)
            if ds.length < 1e-6:
                continue
            q = td_rest[tb_name].rotation_difference(ds.normalized())   # rest방향→pose방향 최소회전
            dir_rot[tb_name] = tWr_i @ q.to_matrix() @ tWr

        # armature-space 포즈 회전 → basis (topo 순)
        pose_rot = {}
        for n in order:
            p = parent[n]
            par_pose = pose_rot[p] if p else Matrix.Identity(3)
            if n in dir_rot:
                W = dir_rot[n] @ R_t_rest_arm[n]             # 방향 정렬 회전
            else:
                rest_rel = (R_t_rest_arm[p].inverted() @ R_t_rest_arm[n]) if p else R_t_rest_arm[n]
                W = par_pose @ rest_rel                       # 매핑 없으면 부모 상속
            pose_rot[n] = W
            rest_rel = (R_t_rest_arm[p].inverted() @ R_t_rest_arm[n]) if p else R_t_rest_arm[n]
            basis = rest_rel.inverted() @ par_pose.inverted() @ W
            pb = tgt.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = basis.to_quaternion()
            pb.location = (0, 0, 0)

        # 루트 이동(상하+좌우, 전후는 제자리 위해 억제) — 스케일 적용
        hips_now = Vector((sW @ src.pose.bones["Hips"].matrix).to_translation())
        d = (hips_now - hips_rest) * ratio
        rb = tgt.pose.bones[roles["root"]]
        loc_world = Vector((d.x, 0.0, d.z))                  # 전진(-Y) 성분 제거→제자리
        rb.location = ml[roles["root"]].to_3x3().inverted() @ (tWr_i @ loc_world)

        for pb in tgt.pose.bones:
            pb.keyframe_insert("rotation_quaternion", frame=out_i+1)
        rb.keyframe_insert("location", frame=out_i+1)
        bpy.context.view_layer.update()
        v = M.eval_verts(mesh)
        frames.append((v, M.bone_segs(tgt)))
        s_mx, s_p95, _ = M.edge_stretch(rest_len, edges, v)
        mx = max(mx, s_mx); p95 = max(p95, s_p95)

    sc.frame_start = 1; sc.frame_end = len(frames)
    sc.render.fps = a.fps      # BVH import 가 fps 를 소스값(120)으로 바꿔놓음 → 출력 fps 로 복원
    return {"tgt": tgt, "mesh": mesh, "src": src, "roles": roles, "frames": frames,
            "nframes": len(frames), "max": mx, "p95": p95, "ratio": ratio, "fps": a.fps}


def export_glb(glb, tgt, mesh, src, nframes):
    """타깃만 애니메이션과 함께 GLB 로 내보낸다.

    주의: 소스 BVH armature 를 hide 만 하면 glTF exporter(기본 ACTIONS 모드)가 소스의
    BVH 액션을 타깃에 묶어 내보내 **정적 GLB**(키프레임 2개)가 나온다. 소스 오브젝트와
    액션을 실제로 삭제하고 SCENE 모드로 현재 씬 애니메이션을 베이크해야 한다.
    """
    src_act = src.animation_data.action if src.animation_data else None
    bpy.data.objects.remove(src, do_unlink=True)
    if src_act is not None:
        bpy.data.actions.remove(src_act, do_unlink=True)
    for o in bpy.data.objects:
        o.select_set(o in (tgt, mesh))
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format="GLB",
                              use_selection=True, export_animations=True,
                              export_animation_mode="SCENE", export_frame_range=True,
                              export_bake_animation=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", required=True)
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--name", default="clip")
    ap.add_argument("--out", default="motion_out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=int, default=60)   # 정착 프레임 스킵
    ap.add_argument("--maxframes", type=int, default=64)
    a = ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:])

    fbx = Path(a.fbx); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = build_animation(fbx, a.bvh, fps=a.fps, start=a.start, maxframes=a.maxframes)
    frames, mx, p95 = r["frames"], r["max"], r["p95"]

    gif = out / f"{fbx.stem}__rt_{a.name}.gif"
    M.render_frames(frames, gif, f"{fbx.stem}  [retarget:{a.name}]")
    glb = out / f"{fbx.stem}__rt_{a.name}.glb"
    try:
        export_glb(glb, r["tgt"], r["mesh"], r["src"], r["nframes"])
    except Exception as e:
        print(f"  glb export fail: {e}")
    print(f"[{a.name}] {len(frames)}f  stretch max={mx:.2f} p95={p95:.3f}  ratio={r['ratio']:.2f} -> {gif.name}, {glb.name}")
    (out / f"{fbx.stem}__rt_{a.name}_stress.json").write_text(
        json.dumps({"p95": round(p95, 3), "max": round(mx, 2), "frames": len(frames)}, indent=2))


def topo(names, parent):
    out, seen = [], set()
    def visit(n):
        if n in seen: return
        if parent[n]: visit(parent[n])
        seen.add(n); out.append(n)
    for n in names: visit(n)
    return out

if __name__ == "__main__":
    main()
