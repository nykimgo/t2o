"""Two controls the headline comparison needs before its numbers mean anything.

1. Is nvdiffrast even self-consistent? Everything measured so far attributes all
   difference to the rasterizer swap. If nvdiffrast disagrees with itself across
   two runs on an identical atlas, part of that difference is just noise.

2. Does the tie-break rule explain the chart-seam swaps? to_glb's chunked
   `torch.where` loop biases toward later chunks, so uv_raster_torch resolves
   contested texels to the highest triangle index. If nvdiffrast instead keeps the
   first triangle to pass its depth test, switching to lowest-index should collapse
   the seam-swap count — a cheap experiment worth running before accepting them.

Operates on rast/positions only; no baking, so it iterates in seconds.

Run:
    conda activate trellis2
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/sr/previs_proj/t2o_pipeline/trellis2_src \
        python experiments/uv_raster_swap/tiebreak_probe.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_glb_split

TEXTURE_SIZE = 2048


def stats(label: str, ref, got, n_ref_cov: int) -> None:
    (rm, rp), (gm, gp) = ref, got
    both = rm & gm
    only_ref = int((rm & ~gm).sum())
    only_got = int((gm & ~rm).sum())
    d = (rp[both] - gp[both]).norm(dim=-1)
    seam = int((d > 1e-4).sum())
    print(f"    {label:<28} only_A={only_ref:<7} only_B={only_got:<7} "
          f"seam_swaps={seam:<7} ({seam / max(n_ref_cov, 1) * 100:.4f}%)  "
          f"max_dpos={float(d.max()) if d.numel() else 0:.3e}")


def main() -> None:
    here = Path(__file__).resolve().parent
    caps = sorted((here / "captures").glob("*.pt"))
    cache_dir = here / "atlas_cache"
    cache_dir.mkdir(exist_ok=True)

    for cap in caps:
        payload = torch.load(cap, weights_only=False)
        print(f"\n=== {cap.stem} ===")
        state = to_glb_split.prefix_cached(
            str(cache_dir / f"{cap.stem}_atlas.pt"),
            vertices=payload["vertices"], faces=payload["faces"],
            attr_volume=payload["attr_volume"], coords=payload["coords"],
            attr_layout=payload["attr_layout"], aabb=payload["aabb"],
            voxel_size=payload["voxel_size"], decimation_target=1000000,
            remesh=True, remesh_band=1, remesh_project=0,
        )
        uvs, faces, verts = state["out_uvs"], state["out_faces"], state["out_vertices"]
        print(f"    atlas: V={tuple(verts.shape)} F={tuple(faces.shape)}")

        def run(backend):
            return to_glb_split.BACKENDS[backend](uvs, faces, verts, TEXTURE_SIZE)

        nv1 = run("nvdiffrast")
        n_cov = int(nv1[0].sum())
        print(f"    nvdiffrast coverage: {n_cov}")

        nv2 = run("nvdiffrast")
        stats("CONTROL nvdiffrast x2", nv1, nv2, n_cov)
        stats("torch (amax tie-break)", nv1, run("torch"), n_cov)
        stats("torch (amin tie-break)", nv1, run("torch_amin"), n_cov)

        del state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
