"""End-to-end check that the shipping GLB path runs with nvdiffrast unavailable.

The comparison work used a vendored copy of to_glb. This exercises the real
installed `o_voxel.postprocess.to_glb` plus the exact module chain
trellis2_inference_core.py imports, with nvdiffrast and nvdiffrec hard-blocked at
the import hook — so a lingering dependency fails loudly instead of quietly
working on this box because the package happens to still be installed.

Run:
    conda activate trellis2
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/sr/previs_proj/t2o_pipeline/trellis2_src \
        python experiments/uv_raster_swap/verify_shipping_path.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPCONV_ALGO", "native")

import sys
from pathlib import Path

BLOCKED = ("nvdiffrast", "nvdiffrec_render", "nvdiffrec")


class _Blocked:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"{name} is blocked: non-commercial license")
        return None


sys.meta_path.insert(0, _Blocked())

import numpy as np
import torch
from PIL import Image

# The module chain trellis2_inference_core.py pulls in at import time.
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
from trellis2.utils import render_utils  # noqa: F401
from trellis2.renderers import EnvMap  # noqa: F401
import o_voxel
import o_voxel.postprocess

print("✅ shipping import chain clean under blocked", BLOCKED)
assert "nvdiffrast" not in sys.modules

_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
here = Path(__file__).resolve().parent
out_dir = here / "verify_out"
out_dir.mkdir(exist_ok=True)

failures = []
for cap in sorted((here / "captures").glob("*.pt")):
    payload = torch.load(cap, weights_only=False)
    print(f"\n=== {cap.stem} ===")

    glb = o_voxel.postprocess.to_glb(
        vertices=payload["vertices"].cuda(),
        faces=payload["faces"].cuda(),
        attr_volume=payload["attr_volume"].cuda(),
        coords=payload["coords"].cuda(),
        attr_layout=payload["attr_layout"],
        voxel_size=payload["voxel_size"],
        aabb=_AABB,
        decimation_target=1000000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
    )
    path = out_dir / f"{cap.stem}.glb"
    glb.export(str(path), extension_webp=False)

    # Reload to confirm the file is a valid, textured GLB rather than just an
    # object that exported without raising.
    import trimesh
    loaded = trimesh.load(str(path), force="mesh", process=False)
    tex = loaded.visual.material.baseColorTexture
    arr = np.array(tex)
    black = float((arr[..., :3].max(axis=-1) == 0).mean())
    print(f"    V={len(loaded.vertices)} F={len(loaded.faces)} "
          f"texture={arr.shape} black_texels={black * 100:.2f}%")

    ok = len(loaded.vertices) > 0 and len(loaded.faces) > 0 and black < 0.5
    if not ok:
        failures.append(cap.stem)
    print(f"    --> {'OK' if ok else 'FAIL'}")

    del glb, loaded
    torch.cuda.empty_cache()

print(f"\nRESULT: {'all good' if not failures else 'FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
