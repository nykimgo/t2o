"""edit3d 평가 지표 — 순수 기하 연산 (trellis2 의존 없음).

Phase 0-1: Chamfer distance + 실루엣 IoU (encode→decode 왕복 손실).
Phase 0-2: coords 집합 IoU (좌표 왕복).
모든 지표는 메시를 단위 큐브 [-0.5,0.5] 로 정규화한 뒤 계산한다 —
절대 위치/스케일이 아니라 '형상 일치'를 재기 위함(축 정규화 불일치가 손실로 오염되지 않게).
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:  # pragma: no cover
    _HAVE_KDTREE = False


# ---- 정규화 / 샘플링 --------------------------------------------------------

def normalize_unit_cube(vertices: np.ndarray) -> np.ndarray:
    """center+isotropic scale 로 [-0.5,0.5] 에 맞춘다. preprocess_mesh 와 같은 규약(축 스왑 제외)."""
    v = np.asarray(vertices, dtype=np.float64)
    vmin, vmax = v.min(0), v.max(0)
    center = (vmin + vmax) / 2
    extent = (vmax - vmin).max()
    if extent < 1e-12:
        return v - center
    return (v - center) * (0.99999 / extent)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """면적 가중 균일 표면 샘플링 (trimesh 불필요, numpy 만)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]                                  # (F,3,3)
    ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(ab, ac), axis=1)
    total = areas.sum()
    if total < 1e-12 or len(f) == 0:
        return tri.reshape(-1, 3)[:n] if len(f) else np.zeros((0, 3))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=n, p=areas / total)
    u = rng.random(n)[:, None]
    w = rng.random(n)[:, None]
    over = (u + w > 1).ravel()
    u[over], w[over] = 1 - u[over], 1 - w[over]
    return tri[idx, 0] + u * ab[idx] + w * ac[idx]


# ---- 0-1 지표 ---------------------------------------------------------------

def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if _HAVE_KDTREE:
        d, _ = cKDTree(b).query(a, k=1)
        return d
    # 폴백: 청크 브루트포스
    out = np.empty(len(a))
    for i in range(0, len(a), 2048):
        chunk = a[i:i + 2048]
        d2 = ((chunk[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[i:i + 2048] = np.sqrt(d2.min(1))
    return out


def chamfer_distance(va, fa, vb, fb, n: int = 50000, seed: int = 0) -> dict:
    """양방향 Chamfer. 단위 큐브 정규화 후. 값이 작을수록 왕복 손실이 적다.

    반환: {'chamfer': 평균 양방향, 'a_to_b': , 'b_to_a': , 'hausdorff': 최대}
    """
    pa = sample_surface(normalize_unit_cube(va), fa, n, seed)
    pb = sample_surface(normalize_unit_cube(vb), fb, n, seed + 1)
    if len(pa) == 0 or len(pb) == 0:
        return {'chamfer': float('nan'), 'a_to_b': float('nan'),
                'b_to_a': float('nan'), 'hausdorff': float('nan')}
    d_ab, d_ba = _nn_dist(pa, pb), _nn_dist(pb, pa)
    return {
        'chamfer': float((d_ab.mean() + d_ba.mean()) / 2),
        'a_to_b': float(d_ab.mean()),
        'b_to_a': float(d_ba.mean()),
        'hausdorff': float(max(d_ab.max(), d_ba.max())),
    }


def silhouette_iou(va, fa, vb, fb, res: int = 128, n: int = 300000, seed: int = 0) -> dict:
    """3개 정사영 뷰(x/y/z 축 제거)의 실루엣 IoU 평균.

    표면 점을 조밀 샘플→2D 격자에 찍고, dilation+fill_holes 로 **채워진** 실루엣을 만든다.
    (성긴 점 마스크를 그대로 쓰면 같은 형상도 샘플 노이즈로 IoU 가 낮게 나온다.)
    같은 프레임/AABB 로 정규화하므로 세 축 뷰가 두 메시 간 일치한다.
    """
    from scipy import ndimage
    pa = sample_surface(normalize_unit_cube(va), fa, n, seed)
    pb = sample_surface(normalize_unit_cube(vb), fb, n, seed + 1)
    if len(pa) == 0 or len(pb) == 0:
        return {'silhouette_iou': float('nan'), 'per_view': [float('nan')] * 3}

    def mask(pts, drop):
        keep = [k for k in range(3) if k != drop]
        ij = np.clip(((pts[:, keep] + 0.5) * (res - 1)).astype(int), 0, res - 1)
        m = np.zeros((res, res), dtype=bool)
        m[ij[:, 0], ij[:, 1]] = True
        m = ndimage.binary_dilation(m, iterations=2)   # 샘플 간극 연결
        m = ndimage.binary_fill_holes(m)               # 내부 채움 → 실루엣
        return m

    ious = []
    for drop in range(3):
        ma, mb = mask(pa, drop), mask(pb, drop)
        inter = np.logical_and(ma, mb).sum()
        union = np.logical_or(ma, mb).sum()
        ious.append(float(inter / union) if union else float('nan'))
    return {'silhouette_iou': float(np.nanmean(ious)), 'per_view': ious}


# ---- 0-2 지표 ---------------------------------------------------------------

def coords_iou(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """정수 복셀 인덱스 집합(N,3)의 IoU. 좌표 왕복 손실(외부 반입 에셋 편집 신뢰도 상한)."""
    if len(coords_a) == 0 and len(coords_b) == 0:
        return 1.0
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0
    sa = set(map(tuple, np.asarray(coords_a, dtype=np.int64)))
    sb = set(map(tuple, np.asarray(coords_b, dtype=np.int64)))
    return len(sa & sb) / len(sa | sb)
