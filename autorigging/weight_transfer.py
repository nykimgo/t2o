"""웨이트 전이 — movie_char_render.py 의 align_and_transfer 발췌 모듈화 (DESIGN §6).

UniRig 리깅 메시는 리메시되어 UV 가 없으므로, 리깅 메시의 스킨 웨이트를
원본(UV+텍스처) 메시로 최근접 이웃 보간 전이한 뒤 같은 armature 에 바인딩한다.
UniRig 는 메시를 단위큐브로 정규화하므로 bbox 기반 정렬이 선행된다.
로직은 원 서버 검증본과 동일 (수정 금지 — 검증 수치의 전제).
"""
from __future__ import annotations

import numpy as np
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree


def biggest_mesh(objs):
    return max((o for o in objs if o.type == "MESH"), key=lambda o: len(o.data.vertices))


def world_verts(ob):
    mw = ob.matrix_world
    return np.array([(mw @ v.co)[:] for v in ob.data.vertices], dtype=np.float64)


def align_and_transfer(src_ob, rig_ob, arm, k=4):
    """원본(UV) 메시를 리깅 메시 공간으로 정렬하고 스킨 웨이트를 최근접 보간 전이."""
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
