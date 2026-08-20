"""EXP1 부가 — 자동리깅 모션 QA 공용 유틸.

익명 본 토폴로지 감지 + 변형 메시 읽기 + 측면/3-4뷰 점군·본 GIF 렌더 + 엣지 신장률 지표.
retarget.py(mocap 리타게팅, 채택안)가 import 해서 쓴다.

실행 env: /home/sr/miniconda3/envs/unirig/bin/python  (bpy 4.2 + numpy + matplotlib + PIL)
"""
from __future__ import annotations
import math
import numpy as np
import bpy


# ---------------------------------------------------------------- FBX 로드
def load(fbx):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    return arm, mesh


# ---------------------------------------------------------------- 토폴로지 감지
def detect(arm):
    """익명 본 → {root, spine[], head, armL/R{shoulder,upper,fore,hand}, legL/R{thigh,shin,foot}}.
    이름 무관, 기하(발=최저Z 리프, 손=최대|X| 리프, 머리=최고Z)+계층으로 역할 추론."""
    bones = arm.data.bones
    B = {b.name: b for b in bones}
    head = {n: np.array(b.head_local) for n, b in B.items()}   # armature-space rest head
    tail = {n: np.array(b.tail_local) for n, b in B.items()}
    parent = {n: (b.parent.name if b.parent else None) for n, b in B.items()}
    children = {n: [c.name for c in b.children] for n, b in B.items()}
    root = next(n for n, p in parent.items() if p is None)

    def chain_to(leaf):                       # root 제외, root->leaf 경로
        c, out = leaf, []
        while c is not None:
            out.append(c); c = parent[c]
        out.reverse()
        return out
    leaves = [n for n in B if not children[n]]
    tips = {n: tail[n] for n in leaves}

    # 발 = tip z 최저 2개 (좌우) / 손 = tip |x| 최대 2개 / 머리 = tip z 최고 (중앙)
    by_z = sorted(leaves, key=lambda n: tips[n][2])
    feet = by_z[:2]
    rest = [n for n in leaves if n not in feet]
    # 손: |x| 큰 리프를 좌/우로 나눠 각 편에서 1개씩 선택(손가락 본이 있으면
    # 같은 손의 손가락 2개가 전역 top-2를 모두 차지해 반대편이 누락될 수 있음).
    xmax = max((abs(tips[n][0]) for n in rest), default=0.0)
    far = [n for n in rest if abs(tips[n][0]) > 0.5 * xmax]
    right_far = [n for n in far if tips[n][0] >= 0]
    left_far = [n for n in far if tips[n][0] < 0]
    hands = []
    if right_far: hands.append(max(right_far, key=lambda n: abs(tips[n][0])))
    if left_far: hands.append(max(left_far, key=lambda n: abs(tips[n][0])))
    remain = [n for n in rest if n not in hands]
    head_leaf = max(remain, key=lambda n: tips[n][2]) if remain else max(by_z, key=lambda n: tips[n][2])

    def side(n):
        """해부학적 좌우. 에셋은 Z-up·정면 -Y 이므로 캐릭터의 오른쪽 =
        cross(forward,up) = cross((0,-1,0),(0,0,1)) = (-1,0,0) → **-X 가 R**.
        (CMU BVH 도 같은 정렬이라 RightArm rest dir 이 -X 다. 예전엔 +X를 R로 라벨해서
        좌우가 뒤집혔고, 그 결과 팔이 반대편 방향으로 강제돼 겨드랑이가 찢어졌다.)"""
        return "R" if tips[n][0] < 0 else "L"

    legs = {}
    for f in feet:
        ch = [n for n in chain_to(f) if n != root]     # thigh, shin, foot, (toe)
        legs[side(f)] = {"thigh": ch[0], "shin": ch[1] if len(ch) > 1 else ch[0],
                         "foot": ch[2] if len(ch) > 2 else ch[-1], "chain": ch}
    arms = {}
    for h in hands:
        ch = [n for n in chain_to(h) if n != root]     # ...spine..., shoulder, upper, fore, hand, fingers
        # 팔은 spine 뒤에서 갈라짐 → |x| 가 커지기 시작하는 지점부터가 팔
        xs = [abs(head[n][0]) for n in ch]
        start = len(ch) - 1
        for i in range(1, len(ch)):
            if xs[i] > 0.05 and xs[i] > xs[i-1]:
                start = i; break        # 팔이 몸통에서 갈라지는 첫 본(=쇄골/어깨)
        arm_ch = ch[start:]
        arms[side(h)] = {"shoulder": arm_ch[0],
                         "upper": arm_ch[1] if len(arm_ch) > 1 else arm_ch[0],
                         "fore": arm_ch[2] if len(arm_ch) > 2 else arm_ch[-1],
                         "hand": arm_ch[3] if len(arm_ch) > 3 else arm_ch[-1],
                         "chain": arm_ch}
    spine = [n for n in chain_to(head_leaf) if n != root]
    return {"root": root, "spine": spine, "head": head_leaf,
            "armR": arms.get("R"), "armL": arms.get("L"),
            "legR": legs.get("R"), "legL": legs.get("L"),
            "_head_local": head, "_parent": parent}


def print_roles(r):
    print(f"  root  = {r['root']}")
    print(f"  spine = {r['spine']}  head={r['head']}")
    for k in ("legR", "legL"):
        s = r[k]; print(f"  {k}: thigh={s['thigh']} shin={s['shin']} foot={s['foot']}  chain={s['chain']}")
    for k in ("armR", "armL"):
        s = r[k]; print(f"  {k}: shoulder={s['shoulder']} upper={s['upper']} fore={s['fore']} hand={s['hand']}")


def deg(d): return d * math.pi / 180


# ---------------------------------------------------------------- 변형 메시 읽기
def eval_verts(mesh):
    dg = bpy.context.evaluated_depsgraph_get()
    ob = mesh.evaluated_get(dg)
    me = ob.to_mesh()
    mw = ob.matrix_world
    v = np.array([(mw @ vv.co)[:] for vv in me.vertices], dtype=np.float32)
    ob.to_mesh_clear()
    return v


def bone_segs(arm):
    mw = arm.matrix_world
    segs = []
    for pb in arm.pose.bones:
        segs.append(((mw @ pb.head)[:], (mw @ pb.tail)[:]))
    return segs


# ---------------------------------------------------------------- 렌더
def render_frames(frames, out_gif, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    allv = np.concatenate([f[0] for f in frames], 0)
    views = [(90, "side"), (35, "3/4")]
    def proj(v, a):
        c, s = math.cos(deg(a)), math.sin(deg(a))
        return v[:, 0]*c - v[:, 1]*s, v[:, 2]
    lims = {}
    for a, _ in views:
        hx, vy = proj(allv, a)
        lims[a] = (hx.min(), hx.max(), vy.min(), vy.max())
    imgs = []
    for i, (verts, segs) in enumerate(frames):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), facecolor="#141414")
        for ax, (a, nm) in zip(axes, views):
            ax.set_facecolor("#141414")
            hx, vy = proj(verts, a)
            ax.scatter(hx, vy, s=0.5, c="#8ab4ff", alpha=0.30, linewidths=0, rasterized=True)
            for h, t in segs:
                H = np.array([h]); T = np.array([t])
                hxh, vyh = proj(H, a); hxt, vyt = proj(T, a)
                ax.plot([hxh[0], hxt[0]], [vyh[0], vyt[0]], c="#ff4444", lw=1.4, alpha=0.9)
            x0, x1, y0, y1 = lims[a]
            pad = 0.05*max(x1-x0, y1-y0)
            ax.set_xlim(x0-pad, x1+pad); ax.set_ylim(y0-pad, y1+pad)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(nm, color="#999", fontsize=8)
        fig.suptitle(f"{title}  f{i+1}/{len(frames)}", color="#ddd", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        imgs.append(Image.fromarray(buf[:, :, :3].copy()))
        plt.close(fig)
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=60, loop=0, optimize=True)


# ---------------------------------------------------------------- 스킨 스트레스
def edge_stretch(rest_len, edges, cur_v):
    c = np.linalg.norm(cur_v[edges[:, 0]] - cur_v[edges[:, 1]], axis=1)
    ratio = c / rest_len
    return float(ratio.max()), float(np.percentile(ratio, 95)), float(ratio.mean())


def get_edges(mesh, n=4000):
    me = mesh.data
    e = np.array([(ed.vertices[0], ed.vertices[1]) for ed in me.edges], dtype=np.int64)
    if len(e) > n:
        e = e[np.linspace(0, len(e)-1, n).astype(int)]
    return e
