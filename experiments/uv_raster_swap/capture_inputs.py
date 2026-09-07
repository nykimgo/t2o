"""Capture the exact inputs o_voxel.postprocess.to_glb receives in a real run.

Why: the nvdiffrast-vs-torch rasterizer comparison must run on real TRELLIS.2
output (a real UV atlas with real chart seams), but re-running 4B inference for
every comparison iteration is wasteful and non-deterministic across driver/env
changes. So we run inference once, freeze the to_glb kwargs to disk, and do all
subsequent comparison offline.

Run:
    conda activate trellis2
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/sr/previs_proj/t2o_pipeline/trellis2_src \
        python experiments/uv_raster_swap/capture_inputs.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")  # matches the shipping core
os.environ.setdefault("SPCONV_ALGO", "native")

import argparse
from pathlib import Path

import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
# Mirrors trellis2_inference_core.py:95 / :331 — the shipping call site.
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
_SIMPLIFY_TARGET = 16777216  # trellis2_inference_core.py:246 default

# TRELLIS.2's own examples: 'T.png' is a thin/open-surface letter (stresses chart
# seams), the hash-named webps are dense organic assets. Different UV atlas shapes
# exercise different rasterizer edge cases.
DEFAULT_IMAGES = [
    "assets/example_image/T.png",
    "assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp",
    "assets/example_image/154c88671d9e8785bd909e9283bc87fb2709ac7ce13890832603ea7533981a46.webp",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis2-root", default="/home/sr/TRELLIS.2",
                    help="source of the example images")
    ap.add_argument("--model", default=str(_REPO / "hf_models" / "TRELLIS.2-4B"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "captures"))
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.trellis2_root)
    images = args.images or [str(root / p) for p in DEFAULT_IMAGES]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    for img_path in images:
        name = Path(img_path).stem[:24]
        dest = out_dir / f"{name}.pt"
        if dest.exists():
            print(f"[skip] {dest} exists")
            continue

        print(f"[run ] {img_path}")
        torch.manual_seed(args.seed)
        mesh = pipeline.run(Image.open(img_path), seed=args.seed)[0]
        mesh.simplify(_SIMPLIFY_TARGET)

        # Everything to_glb() needs, on CPU so the capture is portable and can be
        # reloaded without holding the 4B pipeline in VRAM.
        payload = {
            "source_image": str(img_path),
            "seed": args.seed,
            "vertices": mesh.vertices.cpu(),
            "faces": mesh.faces.cpu(),
            "attr_volume": mesh.attrs.cpu(),
            "coords": mesh.coords.cpu(),
            "attr_layout": mesh.layout,
            "voxel_size": mesh.voxel_size,
            "aabb": _AABB,
        }
        torch.save(payload, dest)
        print(f"[save] {dest}  V={payload['vertices'].shape} F={payload['faces'].shape} "
              f"attrs={payload['attr_volume'].shape}")

        del mesh
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
