"""edit3d coords — 메시 ↔ 복셀 좌표 (open3d-free, o_voxel 경로).

규약(트렐리스와 동일): coords 는 정수 복셀 인덱스, 컬럼 `[batch, x, y, z]`, 범위 [0, resolution).
좌표 손실의 진짜 원인은 (1) AABB 정규화 불일치, (2) 메시 왕복 — PLAN §0-2.
그래서 (1)은 `preprocess_mesh` 규약을 그대로 재현/재사용해 없애고, (2)만 0-2 로 측정한다.

자체 생성 에셋은 생성 시점 coords 를 GLB 옆에 캐시로 남겨 손실 0 으로 만든다.
"""
from __future__ import annotations
import os
import numpy as np

_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def preprocess_reference(vertices: np.ndarray, faces: np.ndarray):
    """`Trellis2TexturingPipeline.preprocess_mesh` 와 **동일한** 정규화 재현.

    center/isotropic scale → [-0.5,0.5], 그리고 Y/Z 축 스왑(new_y=-old_z, new_z=old_y).
    파이프라인이 로드돼 있으면 `mesh_to_coords(..., preprocess_fn=tex.preprocess_mesh)` 로
    실제 메서드를 재사용하는 편이 규약 드리프트를 막는다. 이건 파이프라인 없이 쓸 때의 사본.
    """
    v = np.asarray(vertices, dtype=np.float64).copy()
    vmin, vmax = v.min(0), v.max(0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    v = (v - center) * scale
    tmp = v[:, 1].copy()
    v[:, 1] = -v[:, 2]
    v[:, 2] = tmp
    assert np.all(v >= -0.5) and np.all(v <= 0.5), 'vertices out of range'
    return v, np.asarray(faces, dtype=np.int64)


def mesh_to_voxel_indices(vertices: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    """정규화된 메시([-0.5,0.5]) → 점유 복셀 인덱스 (N,3) int, [0,resolution).

    encode_shape_slat 이 쓰는 것과 동일한 o_voxel 호출/파라미터. import 는 지연(무거움)."""
    import torch
    import o_voxel
    vt = torch.from_numpy(np.asarray(vertices)).float().cpu()
    ft = torch.from_numpy(np.asarray(faces)).long().cpu()
    voxel_indices, _dual, _inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        vt, ft, grid_size=resolution, aabb=_AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False,
    )
    return voxel_indices.cpu().numpy().astype(np.int64)


def indices_to_coords(voxel_indices: np.ndarray, batch: int = 0) -> np.ndarray:
    """(N,3) → (N,4) `[batch,x,y,z]` (SparseTensor.coords 규약)."""
    vi = np.asarray(voxel_indices, dtype=np.int64)
    b = np.full((len(vi), 1), batch, dtype=np.int64)
    return np.concatenate([b, vi], axis=1)


def mesh_to_coords(vertices, faces, resolution: int, preprocess_fn=None):
    """(원시 메시) → (voxel_indices (N,3), coords (N,4)). preprocess_fn 있으면 재사용."""
    if preprocess_fn is not None:
        import trimesh
        pm = preprocess_fn(trimesh.Trimesh(vertices=np.asarray(vertices),
                                           faces=np.asarray(faces), process=False))
        v, f = np.asarray(pm.vertices), np.asarray(pm.faces)
    else:
        v, f = preprocess_reference(vertices, faces)
    vi = mesh_to_voxel_indices(v, f, resolution)
    return vi, indices_to_coords(vi)


def downsample_indices(voxel_indices: np.ndarray, ratio: int) -> np.ndarray:
    """고해상 인덱스 → 저해상(예: 1024→32 는 ratio=32). idx//ratio 후 unique.

    파이프라인 내부(max_pool3d>0.5)와 등가인 '미세복셀 하나라도 차면 점유' 규칙."""
    vi = np.asarray(voxel_indices, dtype=np.int64) // int(ratio)
    return np.unique(vi, axis=0)


# ---- 자체 생성 에셋 캐시 (손실 0 경로) --------------------------------------

def cache_path_for(glb_path: str) -> str:
    return os.path.splitext(glb_path)[0] + '.coords.npz'


def save_coords_cache(glb_path: str, voxel_indices: np.ndarray, resolution: int) -> str:
    p = cache_path_for(glb_path)
    np.savez_compressed(p, voxel_indices=np.asarray(voxel_indices, dtype=np.int64),
                        resolution=int(resolution))
    return p


def load_coords_cache(glb_path: str):
    """(voxel_indices, resolution) 또는 None."""
    p = cache_path_for(glb_path)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d['voxel_indices'], int(d['resolution'])
