"""Calibrate uv_raster_torch against nvdiffrast on synthetic atlases.

This pins the conventions that are easy to get silently wrong — image origin
(top-left vs bottom-left), which barycentric goes in which channel, and the
`triangle_id + 1` offset. Real-mesh agreement is measured separately by
compare_rasterizers.py; this file only asserts that the two rasterizers mean the
same thing by "rast".

Requires nvdiffrast (reference only — the shipping path must not need it).

Run:
    conda activate trellis2
    CUDA_VISIBLE_DEVICES=1 python experiments/uv_raster_swap/test_conventions.py
"""
import sys

import torch
import nvdiffrast.torch as dr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import uv_raster_torch as urt


def nvdiffrast_rast(uvs: torch.Tensor, faces: torch.Tensor, T: int) -> torch.Tensor:
    """Exactly the rasterization to_glb performs (postprocess.py:230-243)."""
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([
        uvs * 2 - 1,
        torch.zeros_like(uvs[:, :1]),
        torch.ones_like(uvs[:, :1]),
    ], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, T, T, 4), device="cuda", dtype=torch.float32)
    for i in range(0, faces.shape[0], 100000):
        chunk, _ = dr.rasterize(ctx, uvs_rast, faces[i:i + 100000], resolution=[T, T])
        m = chunk[..., 3:4] > 0
        chunk[..., 3:4] += i
        rast = torch.where(m, chunk, rast)
    return rast


# nvdiffrast snaps vertices to a fixed-point subpixel grid before rasterizing, so
# its edges land slightly off the exact ones and it drops roughly one boundary texel
# per triangle. Verified in fp64: on those texels ours is the correct side of the
# call. A disagreement further from an edge than this is a real bug, not snapping.
EDGE_BAND_PX = 0.5


def _edge_dist_px(rast: torch.Tensor, uvs: torch.Tensor, faces: torch.Tensor,
                  T: int) -> torch.Tensor:
    """Perpendicular distance in texels from each covered texel to the nearest
    edge of its triangle.

    Barycentric depth is not usable here: a triangle covering 3 texels gives a
    large barycentric for a texel sitting a hundredth of a pixel inside its edge.
    """
    tri = uvs[faces.long()].float() * T - 0.5                # (F, 3, 2)
    f = (rast[..., 3].long() - 1).clamp(min=0)
    a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
    H, W = rast.shape[1], rast.shape[2]
    ys, xs = torch.meshgrid(torch.arange(H, device=rast.device),
                            torch.arange(W, device=rast.device), indexing="ij")
    p = torch.stack([xs, ys], dim=-1).float().unsqueeze(0).expand(rast.shape[0], -1, -1, -1)

    def perp(A, B, P):  # |cross| / |B - A|
        num = ((B[..., 0] - A[..., 0]) * (P[..., 1] - A[..., 1]) -
               (B[..., 1] - A[..., 1]) * (P[..., 0] - A[..., 0])).abs()
        return num / (B - A).norm(dim=-1).clamp_min(1e-12)

    return torch.minimum(torch.minimum(perp(b, c, p), perp(c, a, p)), perp(a, b, p))


def report(name: str, uvs: torch.Tensor, faces: torch.Tensor, T: int,
           exact: bool = False) -> bool:
    """Compare against nvdiffrast.

    `exact=True` demands bit-identical coverage — only fair when no two triangles
    share an edge or overlap, so no tie-break is in play.
    """
    ref = nvdiffrast_rast(uvs, faces, T)
    got = urt.rasterize_uv(uvs, faces, T)

    ref_cov = ref[..., 3] > 0
    got_cov = got[..., 3] > 0
    only_ref = ref_cov & ~got_cov
    only_got = got_cov & ~ref_cov
    both = ref_cov & got_cov
    n_both = int(both.sum())
    same_tri = int((ref[..., 3][both] == got[..., 3][both]).sum())

    # Every disagreement must sit within the snapping band of a triangle edge.
    # Measured against whichever rasterizer claims coverage, since the other one
    # has no triangle to measure from.
    worst_depth = 0.0
    for m, r in ((only_ref, ref), (only_got, got)):
        if bool(m.any()):
            worst_depth = max(worst_depth, float(_edge_dist_px(r, uvs, faces, T)[m].max()))

    # to_glb consumes interpolated positions, not raw barycentrics, and those are
    # insensitive to which of two edge-sharing triangles won a boundary texel.
    pos3 = torch.cat([uvs, torch.zeros_like(uvs[:, :1])], dim=-1).contiguous()
    ref_pos = dr.interpolate(pos3.unsqueeze(0), ref, faces)[0][0]
    got_pos = urt.interpolate(pos3.unsqueeze(0), got, faces)[0][0]
    agree = both[0] & (ref[0, ..., 3] == got[0, ..., 3])
    dpos = (ref_pos - got_pos).norm(dim=-1)[agree]
    max_dpos = float(dpos.max()) if dpos.numel() else 0.0

    ok = max_dpos < 1e-5 and worst_depth <= EDGE_BAND_PX
    if exact:
        ok = ok and not bool(only_ref.any()) and not bool(only_got.any()) and same_tri == n_both

    print(f"[{'PASS' if ok else 'FAIL'}] {name}: T={T} F={faces.shape[0]} "
          f"covered={n_both} only_ref={int(only_ref.sum())} only_got={int(only_got.sum())} "
          f"tri_match={same_tri}/{n_both} max_pos_err={max_dpos:.2e} "
          f"worst_disagreement_depth={worst_depth:.2e}px")
    return ok


def main() -> None:
    torch.manual_seed(0)
    all_ok = True

    # 1. One large asymmetric triangle: catches origin flips and axis swaps,
    #    which a symmetric shape would hide.
    uvs = torch.tensor([[0.1, 0.1], [0.8, 0.2], [0.3, 0.9]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    all_ok &= report("single triangle", uvs, faces, 64, exact=True)

    # 2. Reversed winding: to_glb never guarantees a consistent winding in UV space.
    all_ok &= report("flipped winding", uvs,
                     torch.tensor([[0, 2, 1]], dtype=torch.int32, device="cuda"), 64,
                     exact=True)

    # 3. A quad split into two triangles: exercises the shared-edge tie-break.
    quad_uv = torch.tensor([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], device="cuda")
    quad_f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
    all_ok &= report("shared edge quad", quad_uv, quad_f, 128)

    # 4. Dense random atlas at production texture size — the realistic regime of
    #    many sub-pixel triangles.
    n_tri = 20000
    centers = torch.rand(n_tri, 1, 2, device="cuda") * 0.9 + 0.05
    verts = (centers + (torch.rand(n_tri, 3, 2, device="cuda") - 0.5) * 0.01).reshape(-1, 2)
    rfaces = torch.arange(n_tri * 3, dtype=torch.int32, device="cuda").reshape(n_tri, 3)
    all_ok &= report("dense random", verts.clamp(0, 1), rfaces, 2048)

    print("\nRESULT:", "all conventions match" if all_ok else "MISMATCH — do not swap")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
