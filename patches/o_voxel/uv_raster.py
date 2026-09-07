"""Dependency-free UV-space triangle rasterizer, replacing nvdiffrast in to_glb.

Why this exists
---------------
`o_voxel.postprocess.to_glb` bakes the texture by rasterizing the UV atlas and
asking, for every texel, "which triangle covers me and at what barycentric?".
It does that with `nvdiffrast` (`dr.RasterizeCudaContext` / `dr.rasterize` /
`dr.interpolate`), which ships under the NVIDIA Source Code License — research
and evaluation only, no commercial use. That single call site is the only hard
nvdiffrast dependency on the GLB export path, so replacing it clears the whole
shipping pipeline.

Nothing here needs to be differentiable: to_glb only reads the coverage mask and
the interpolated positions. That drops the reason to reach for a full
differentiable-rendering library (PyTorch3D, Kaolin) and lets this stay ~150
lines of plain torch with zero new dependencies.

Approach
--------
Bounding-box scatter. The atlas is packed so that F triangles tile roughly T^2
texels, which after decimation means ~2 texels per triangle — the bboxes are
tiny, so enumerating every (texel, triangle) candidate pair costs ~3*T^2 work
total and vectorizes cleanly. No tiling/binning machinery needed.

Conventions match nvdiffrast so the output is a drop-in `rast` tensor:
  * pixel centers sit at `uv * resolution - 0.5`
  * `rast[..., 0:2]` are the barycentrics of vertices 0 and 1 (vertex 2 gets
    `1 - u - v`), `rast[..., 2]` is z (always 0 here), `rast[..., 3]` is
    `triangle_id + 1` with 0 meaning "not covered"
  * row index increases with v, i.e. no vertical flip. nvdiffrast is documented
    as OpenGL bottom-left-origin, but its CUDA rasterizer writes rows in
    increasing-v order; test_conventions.py measures this rather than trusting
    the docs, and `flip_y=True` exists only to re-check that calibration.

Ties (a texel center landing exactly on a shared edge) resolve to the highest
triangle index. nvdiffrast's own tie-break under an all-zero depth buffer is
unspecified, and to_glb's chunked `torch.where` loop already biases toward later
chunks, i.e. higher indices — so this matches the existing bias and, unlike a
z-buffer race, is deterministic. Measured to barely matter: switching to lowest
index moves the disagreement count by ~3%.

Verified against nvdiffrast on real TRELLIS.2 output — 99.7%+ of geometry-baked
texels bit-identical, 69-85 dB PSNR, and where coverage disagrees an fp64
recomputation puts this implementation on the correct side. Harness, numbers and
method: experiments/uv_raster_swap/ (see FINDINGS.md).
"""
from typing import Optional, Tuple

import torch


def _edge(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """2D cross product (b - a) x (p - a); >0 iff p is left of the a->b edge."""
    return (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - \
           (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])


def rasterize_uv(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    flip_y: bool = False,
    max_candidates: int = 1 << 24,
    tie_break: str = "amax",
) -> torch.Tensor:
    """Rasterize a UV atlas into an nvdiffrast-compatible `rast` tensor.

    Args:
        uvs: (V, 2) UV coordinates in [0, 1].
        faces: (F, 3) vertex indices.
        resolution: output texture is (resolution, resolution).
        flip_y: mirror rows vertically. False matches nvdiffrast (see module docstring).
        max_candidates: cap on candidate (texel, triangle) pairs held at once;
            bounds peak memory on dense atlases.
        tie_break: which triangle wins a texel claimed by several — "amax"
            (highest index) or "amin" (lowest).

    Returns:
        (1, resolution, resolution, 4) float32 `rast` on the input device.
    """
    device = uvs.device
    T = int(resolution)
    faces_l = faces.long()

    # Triangle corners in pixel-center space.
    tri = uvs[faces_l].float() * T - 0.5                     # (F, 3, 2)
    if flip_y:
        tri = torch.stack([tri[..., 0], (T - 1) - tri[..., 1]], dim=-1)

    # Integer bbox per triangle, clamped to the texture. A triangle entirely
    # outside collapses to a 1x1 bbox whose candidate fails the coverage test.
    lo = tri.amin(dim=1).floor().clamp(0, T - 1).long()      # (F, 2)
    hi = tri.amax(dim=1).ceil().clamp(0, T - 1).long()       # (F, 2)
    bw = (hi[:, 0] - lo[:, 0] + 1)
    bh = (hi[:, 1] - lo[:, 1] + 1)
    counts = bw * bh                                          # (F,)

    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    total = int(ends[-1].item()) if counts.numel() else 0

    # Pass 1: resolve which covering triangle owns each texel.
    assert tie_break in ("amax", "amin"), tie_break
    sentinel = -1 if tie_break == "amax" else torch.iinfo(torch.int64).max
    tri_id = torch.full((T * T,), sentinel, dtype=torch.int64, device=device)

    for base in range(0, total, max_candidates):
        n = min(max_candidates, total - base)
        flat = torch.arange(base, base + n, device=device)
        # Which triangle does each candidate slot belong to?
        f = torch.searchsorted(ends, flat, right=True).clamp_(max=counts.numel() - 1)
        local = flat - starts[f]
        px = lo[f, 0] + local % bw[f]
        py = lo[f, 1] + local // bw[f]

        p = torch.stack([px, py], dim=-1).float()
        a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
        w0 = _edge(b, c, p)
        w1 = _edge(c, a, p)
        w2 = _edge(a, b, p)
        # Accept either winding; a degenerate (zero-area) triangle covers nothing.
        area = _edge(a, b, c)
        inside = torch.where(
            area > 0,
            (w0 >= 0) & (w1 >= 0) & (w2 >= 0),
            (w0 <= 0) & (w1 <= 0) & (w2 <= 0),
        ) & (area != 0)

        if inside.any():
            tri_id.scatter_reduce_(
                0,
                (py * T + px)[inside],
                f[inside],
                reduce=tie_break,
                include_self=True,
            )
        del flat, f, local, px, py, p, a, b, c, w0, w1, w2, area, inside

    # Pass 2: exact barycentrics for the winning triangle of each covered texel.
    rast = torch.zeros((T * T, 4), dtype=torch.float32, device=device)
    covered = (tri_id >= 0) & (tri_id != sentinel)
    if covered.any():
        idx = covered.nonzero(as_tuple=True)[0]
        f = tri_id[idx]
        p = torch.stack([idx % T, idx // T], dim=-1).float()
        a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
        area = _edge(a, b, c)
        u = _edge(b, c, p) / area
        v = _edge(c, a, p) / area
        rast[idx, 0] = u
        rast[idx, 1] = v
        rast[idx, 3] = (f + 1).float()

    return rast.view(1, T, T, 4)


def interpolate(
    attr: torch.Tensor,
    rast: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Barycentric interpolation of per-vertex `attr`, mirroring dr.interpolate.

    Returns `(values, None)` so the call site's `[0]` indexing is unchanged.
    """
    faces_l = faces.long()
    u = rast[..., 0]
    v = rast[..., 1]
    w = 1.0 - u - v
    tri = (rast[..., 3].long() - 1).clamp(min=0)             # background -> tri 0, masked out by caller

    if attr.dim() == 3:                                       # (1, V, C) like dr.interpolate
        attr = attr[0]
    corners = attr[faces_l[tri]]                              # (1, H, W, 3, C)
    out = (u.unsqueeze(-1) * corners[..., 0, :] +
           v.unsqueeze(-1) * corners[..., 1, :] +
           w.unsqueeze(-1) * corners[..., 2, :])
    out = torch.where((rast[..., 3:4] > 0), out, torch.zeros_like(out))
    return out, None
