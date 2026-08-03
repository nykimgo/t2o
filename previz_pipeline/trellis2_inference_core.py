"""TRELLIS.2 inference core (Phase 3) — image-to-3D backend for the t2o pipeline.

Drop-in replacement for :class:`TrellisInferenceCore` (TRELLIS v1, text-to-3D).
TRELLIS.2 is image-conditioned, so this core inserts a Text→Image (FLUX) bridge
before the 3D stage:

    prompt --(FLUX)--> reference image --(TRELLIS.2)--> MeshWithVoxel (PBR)
           --> o_voxel.to_glb (PNG) --> GLB  --> [downstream usd_from_gltf]

Design
------
* Subclasses ``TrellisInferenceCore`` to REUSE all path/naming/manifest/CSV/
  generation-json logic unchanged; only pipeline loading and the per-object
  generation+export+render are overridden.
* GLB is exported with ``extension_webp=False`` (PNG textures) so the downstream
  GLB->USD step gets plain .png/.jpg (no EXT_texture_webp). The converter is now
  the native trimesh+pxr path (glb_to_usd_native.py), which also extracts jpg
  textures — PNG/jpg keeps that simple and portable. PBR metallic/roughness maps
  ARE glTF-core and convert fine.
* HDRI .exr is read via the ``OpenEXR`` package (the installed
  opencv-python-headless is built with OpenEXR:NO). Render is best-effort.

STATUS: written against the confirmed TRELLIS.2 API but NOT yet end-to-end
tested — blocked on gated ``facebook/dinov3-vitl16-pretrain-lvd1689m`` access.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _detect_cuda_arch(default: str = "8.0") -> str:
    """Compute capability of GPU 0 as a TORCH_CUDA_ARCH_LIST value (e.g. '8.9').

    Queried via nvidia-smi because this runs before ``import torch`` — the arch
    list must be set before any JIT extension build. A100=8.0, RTX 4090=8.9,
    H100=9.0. Falls back to ``default`` when nvidia-smi is unavailable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        cap = out.stdout.strip().splitlines()[0].strip()
        return cap if cap else default
    except Exception:
        return default


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _detect_cuda_arch())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image

import trellis_inference_core as _base_mod
from trellis_inference_core import TrellisInferenceCore  # base: reuse all helpers

# The base module guards on TRELLIS **v1** availability (its module-level import
# of trellis.pipelines). In the trellis2 env v1 is absent, so the base's
# __init__ guard would raise. We supply our own v2 pipeline, so neutralize it.
_base_mod.TRELLIS_AVAILABLE = True

# The base module sets ATTN_BACKEND=xformers (TRELLIS v1). TRELLIS.2 must use
# flash_attn (xformers isn't installed in the trellis2 env). Force it BEFORE
# importing trellis2, whose backend is selected at import time.
os.environ["ATTN_BACKEND"] = "flash_attn"
os.environ["SPCONV_ALGO"] = "native"

try:
    import imageio
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    import o_voxel
    TRELLIS2_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    print(f"❌ TRELLIS.2 import 실패: {e}")
    print("💡 trellis2_src 에서 실행하거나 PYTHONPATH를 설정하세요")
    TRELLIS2_AVAILABLE = False

# Repo root (= <repo>/t2o_pipeline), derived so the same checkout works on any
# server. Absolute paths here used to be pinned to the original host.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Default HDRI for PBR preview render (best-effort only).
_DEFAULT_HDRI = str(_REPO_ROOT / "trellis2_src/assets/hdri/forest.exr")
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

# FLUX runs in an isolated env — invoked as a subprocess. The t2i env is a
# sibling of the active (trellis2) env under <conda>/envs/; override with
# T2I_PYTHON / T2I_SCRIPT if the layout differs.
_T2I_PYTHON = os.environ.get(
    "T2I_PYTHON", str(Path(sys.prefix).parent / "t2i" / "bin" / "python"))
_T2I_SCRIPT = os.environ.get(
    "T2I_SCRIPT", str(_REPO_ROOT / "previz_pipeline" / "text_to_image.py"))


def _read_exr_rgb(path: str) -> np.ndarray:
    """Read an .exr HDRI to an (H,W,3) float32 RGB array via the OpenEXR pkg."""
    import OpenEXR
    f = OpenEXR.File(path)
    return np.asarray(f.channels()["RGB"].pixels, dtype=np.float32)


class Trellis2InferenceCore(TrellisInferenceCore):
    """Image-to-3D core using TRELLIS.2-4B, with a FLUX Text→Image front-end."""

    def __init__(
        self,
        model_path: str = str(_REPO_ROOT / "hf_models" / "TRELLIS.2-4B"),
        base_output_dir: str = os.environ.get(
            "TRELLIS_BASE_OUTPUT", str(_REPO_ROOT / "t2o_results")),
        t2i_model_path: str = str(_REPO_ROOT / "hf_models" / "FLUX.1-schnell"),
        hdri_path: str = _DEFAULT_HDRI,
        pipeline_type: Optional[str] = None,   # None -> model default '1024_cascade'
    ) -> None:
        if not TRELLIS2_AVAILABLE:
            raise ImportError("TRELLIS.2 modules are not available")
        super().__init__(model_path=model_path, base_output_dir=base_output_dir)
        self.t2i_model_path = t2i_model_path
        self.hdri_path = hdri_path
        self.pipeline_type = pipeline_type
        self.envmap = None

    # -- overridden: load TRELLIS.2 + FLUX + envmap (replaces TrellisTextTo3D) --
    def load_pipeline(self) -> None:
        logging.info(f"🔄 Loading TRELLIS.2 pipeline from: {self.model_path}")
        src = self.model_path if os.path.exists(self.model_path) else "microsoft/TRELLIS.2-4B"
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(src)
        if torch.cuda.is_available():
            self.pipeline.cuda()
            logging.info("✅ TRELLIS.2 pipeline on GPU")
        else:
            logging.warning("ℹ️ GPU not available")

        # HDRI for PBR preview render (optional).
        try:
            self.envmap = EnvMap(torch.tensor(
                _read_exr_rgb(self.hdri_path), dtype=torch.float32, device="cuda"))
            logging.info(f"✅ HDRI envmap loaded: {self.hdri_path}")
        except Exception as e:
            logging.warning(f"⚠️ HDRI load failed (render will be skipped): {e}")
            self.envmap = None

    def _t2i_generate(self, prompt: str, seed: int, out_path: str) -> Image.Image:
        """Run FLUX in the isolated ``t2i`` env (subprocess) and load the PNG.

        Kept out-of-process because FLUX and TRELLIS.2 need different
        transformers versions — the two cannot share one interpreter.

        VRAM: schnell in bf16 barely fits a 24GB card even with CPU offload
        (measured ~23.4GB peak on an empty 4090). TRELLIS.2 is already resident
        here, so sharing the card OOMs. Set ``T2I_GPU`` to hand FLUX its own
        device; on a single-GPU box leave it unset and the two stages must be
        split into separate runs instead.
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            _T2I_PYTHON, _T2I_SCRIPT,
            "--prompt", prompt, "--out", out_path,
            "--seed", str(int(seed)), "--model", self.t2i_model_path,
        ]
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        t2i_gpu = os.environ.get("T2I_GPU")
        if t2i_gpu:
            # The subprocess then sees it as cuda:0 regardless of the index.
            env["CUDA_VISIBLE_DEVICES"] = t2i_gpu
            logging.info(f"🎨 T2I (subprocess/t2i env, GPU {t2i_gpu}): {prompt[:60]}")
        else:
            logging.info(f"🎨 T2I (subprocess/t2i env): {prompt[:60]}")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"T2I subprocess failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")
        return Image.open(out_path).convert("RGB")

    # -- overridden: per-object generation for TRELLIS.2 (T2I -> mesh -> GLB) --
    def _generate_single(
        self,
        prompt: str,
        predefined_name: Optional[str],
        config: Dict,
        formats: List[str],
        postprocessing_config: Dict,
        llm_model: Optional[str] = None,
        record_context: Optional[Dict[str, str]] = None,
    ) -> Dict:
        import random
        start_time = time.time()

        object_name = predefined_name or self.generate_unique_name(prompt)
        seed_val = config.get("seed", "random")
        seed = random.randint(0, 999999) if str(seed_val).lower() == "random" else int(seed_val)

        context = record_context or {}
        scene_name = self._sanitize_path_segment(context.get("scene"), "scene_unknown")
        shot_name = self._sanitize_path_segment(context.get("shot"), "shot_unknown")
        target_dir_name = self._sanitize_path_segment(
            context.get("target_dir_name") or object_name, object_name or "item_unknown")

        # Output dirs — identical rule to the base (assets next to source USD).
        usd_file_path = context.get("usd_file_path")
        preview_dir = self._preview_output_path(scene_name, shot_name, target_dir_name)
        if usd_file_path:
            object_dir = Path(usd_file_path).resolve().parent / "assets" / target_dir_name
        else:
            object_dir = preview_dir
        object_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        asset_label_name = self._resolve_asset_label_name(context, context.get("target_type") or "item")
        file_prefix = self._build_file_prefix(scene_name, shot_name, asset_label_name, seed)

        # --- Stage A: Text -> Image (FLUX) ---
        gen_start = time.time()
        ref_path = preview_dir / f"{file_prefix}_ref.png"
        image = self._t2i_generate(prompt, seed, str(ref_path))
        t2i_time = time.time() - gen_start
        logging.info(f"⏱️  [T2I] {object_name}: {t2i_time:.1f}s")

        # --- Stage B: Image -> 3D (TRELLIS.2) ---
        i2o_start = time.time()
        try:
            mesh = self.pipeline.run(
                image,
                seed=seed,
                preprocess_image=True,                     # internal rembg
                pipeline_type=self.pipeline_type,
                sparse_structure_sampler_params=config.get("sparse_structure_sampler_params", {}),
                shape_slat_sampler_params=config.get("shape_slat_sampler_params", {}),
                tex_slat_sampler_params=config.get("tex_slat_sampler_params", {}),
            )[0]
            mesh.simplify(int(postprocessing_config.get("simplify_target", 16777216)))
        except Exception as e:
            logging.error(f"❌ TRELLIS.2 run failed: {e}")
            raise
        i2o_time = time.time() - i2o_start
        logging.info(f"⏱️  [I2O] {object_name}: {i2o_time:.1f}s")
        generation_time = time.time() - gen_start

        # --- Stage C: PBR preview render (best-effort) ---
        render_start = time.time()
        video = None
        if self.envmap is not None:
            try:
                video = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh, envmap=self.envmap))
            except Exception as e:
                logging.warning(f"⚠️ PBR render failed (skipping): {e}")
        render_time = time.time() - render_start

        # --- Stage D: export GLB (PNG textures for portable GLB->USD) + previews ---
        saved_files: List[str] = []
        glb_path_saved: Optional[Path] = None
        save_start = time.time()

        if "glb" in formats:
            glb_filename = self._get_unique_filename(object_dir, f"{file_prefix}.glb")
            glb_path = object_dir / glb_filename
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=_AABB,
                decimation_target=int(postprocessing_config.get("decimation_target", 1000000)),
                texture_size=int(postprocessing_config.get("texture_size", 2048)),
                remesh=bool(postprocessing_config.get("remesh", True)),
                remesh_band=postprocessing_config.get("remesh_band", 1),
                remesh_project=postprocessing_config.get("remesh_project", 0),
            )
            glb.export(str(glb_path), extension_webp=False)  # PNG, NOT webp
            saved_files.append(str(glb_path))
            glb_path_saved = glb_path
            logging.info(f"💾 GLB saved (PNG textures): {glb_filename}")
            self._write_glb_meta(glb_path, {
                "run_id": context.get("run_id") or self.run_id,
                "object_path": context.get("object_path") or context.get("file_identifier"),
                "prompt_used": prompt,
                "reference_image": str(ref_path),
                "t2i_prompt": context.get("t2i_prompt"),
                "description_en": context.get("description_en"),
                "seed": seed,
                "backend": "trellis2",
            })

        if "mp4" in formats and video is not None:
            mp4_name = self._get_unique_filename(preview_dir, f"{file_prefix}_pbr.mp4")
            mp4_path = preview_dir / mp4_name
            imageio.mimsave(str(mp4_path), video, fps=15)
            saved_files.append(str(mp4_path))
            logging.info(f"💾 PBR video saved: {mp4_name}")

        if ("jpg" in formats or video is not None) and video is not None:
            try:
                for sec in (2, 4, 6):
                    idx = sec * 15
                    if len(video) > idx:
                        thumb = self._get_unique_filename(preview_dir, f"{file_prefix}_{sec:03d}s.jpg")
                        Image.fromarray(video[idx]).save(str(preview_dir / thumb), "JPEG", quality=90)
                        saved_files.append(str(preview_dir / thumb))
            except Exception as e:
                logging.warning(f"⚠️ Thumbnail generation failed: {e}")

        save_time = time.time() - save_start
        total_time = time.time() - start_time
        logging.info(
            f"⏱️  [객체 합계] {object_name}: {total_time:.1f}s "
            f"(T2I {t2i_time:.1f}s / I2O {i2o_time:.1f}s / "
            f"render {render_time:.1f}s / save {save_time:.1f}s)")

        preview_only = [Path(p).name for p in saved_files
                        if Path(p).parent.resolve() == preview_dir.resolve()]
        self._write_generation_json(
            preview_dir, prompt=prompt, context=context or {}, seed=seed,
            glb_path=str(glb_path_saved) if glb_path_saved else None,
            preview_files=preview_only,
        )

        return {
            "prompt": prompt,
            "object_name": object_name,
            "seed": seed,
            "model_name": self.model_name,
            "llm_model": llm_model,
            "run_id": context.get("run_id") or self.run_id,
            "object_path": context.get("object_path") or context.get("file_identifier"),
            "reference_image": str(ref_path),
            "generation_time": round(generation_time, 2),
            "t2i_time": round(t2i_time, 2),
            "i2o_time": round(i2o_time, 2),
            "render_time": round(render_time, 2),
            "save_time": round(save_time, 2),
            "total_time": round(total_time, 2),
            "success": True,
            "saved_files": saved_files,
            "save_path": str(object_dir),
            "preview_path": str(preview_dir) if preview_dir != object_dir else None,
            "timestamp": datetime.now().isoformat(),
        }
