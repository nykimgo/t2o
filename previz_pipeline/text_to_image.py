"""Text→Image bridge for the TRELLIS.2 pipeline (Phase 2).

TRELLIS.2 is image-conditioned only, so the t2o (text-to-object) flow needs a
Text→Image stage between the (translated/filtered) prompt and TRELLIS.2.
This module wraps **FLUX.1-schnell** and produces a single clean, centered,
white-background reference image suitable for image-to-3D.

Why schnell (not dev): FLUX.1-schnell is **Apache-2.0 licensed** → unrestricted
commercial use, which the productization goal requires. FLUX.1-dev is
non-commercial and must not ship in a commercial product.

schnell specifics (differ from dev — do NOT copy dev params):
* Guidance-distilled → ``guidance_scale`` is effectively unused (keep 0.0).
* Very few steps — 4 is the sweet spot (1–4 works).
* ``max_sequence_length`` must be <= 256 (dev allows 512).

Notes
-----
* TRELLIS.2's ``pipeline.run(image, preprocess_image=True)`` already runs rembg
  internally, so a hard background cut here is optional; we still bias FLUX
  toward a plain background for cleaner geometry.
* Requires ``diffusers`` in the isolated ``t2i`` env (transformers<5).
* Loading FLUX and TRELLIS.2-4B together can be tight on one GPU; pass
  ``device`` to place FLUX on a separate GPU, or call :meth:`unload`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
_LOCAL_DEFAULT = "/data/previs_object/t2o_pipeline/hf_models/FLUX.1-schnell"

# Prompt scaffolding tuned for single-object, image-to-3D friendly renders.
_POSITIVE_SUFFIX = (
    "single object, centered, full object in frame, plain white background, "
    "studio product shot, soft even lighting, high detail, photorealistic"
)


class TextToImage:
    """FLUX.1-schnell text-to-image generator producing 3D-ready reference images."""

    def __init__(
        self,
        model_path: str = _LOCAL_DEFAULT,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.pipe = None

    def load(self) -> None:
        """Load the FLUX.1-schnell diffusers pipeline onto ``self.device``."""
        if self.pipe is not None:
            return
        try:
            from diffusers import FluxPipeline
        except ImportError as e:  # pragma: no cover - env guard
            raise ImportError(
                "diffusers is required for the T2I stage. "
                "Install into the t2i env: pip install 'diffusers>=0.32' accelerate"
            ) from e

        src = self.model_path if os.path.exists(self.model_path) else _MODEL_ID
        logging.info(f"🔄 Loading FLUX.1-schnell from: {src}")
        self.pipe = FluxPipeline.from_pretrained(src, torch_dtype=self.dtype)
        self.pipe.to(self.device)
        # Modest VRAM relief so FLUX can coexist with TRELLIS.2 on the box.
        self.pipe.enable_attention_slicing()
        logging.info("✅ FLUX pipeline loaded")

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        steps: int = 4,            # schnell: 1–4 steps
        guidance: float = 0.0,     # schnell is guidance-distilled → unused
        size: int = 1024,
        max_seq_len: int = 256,    # schnell requires <= 256
        out_path: Optional[str] = None,
    ) -> Image.Image:
        """Generate one reference image for ``prompt``.

        Args:
            prompt: English object description (post translate/filter).
            seed: RNG seed (kept aligned with the TRELLIS.2 seed upstream).
            steps: FLUX inference steps (schnell: 4).
            guidance: guidance scale (schnell: 0.0, unused).
            size: square output resolution.
            max_seq_len: T5 sequence length cap (schnell: <=256).
            out_path: if given, also save the PNG there.
        Returns:
            PIL.Image in RGB.
        """
        if self.pipe is None:
            self.load()

        full_prompt = f"{prompt.strip()}. {_POSITIVE_SUFFIX}"
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        logging.info(f"🎨 T2I generate (seed={seed}, steps={steps}): {prompt[:60]}")
        image = self.pipe(
            full_prompt,
            height=size,
            width=size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=max_seq_len,
            generator=generator,
        ).images[0]

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            logging.info(f"💾 T2I image saved: {out_path}")
        return image

    def unload(self) -> None:
        """Free FLUX from GPU memory (call before heavy TRELLIS.2 inference)."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logging.info("🧹 FLUX unloaded")


def _main() -> None:
    """CLI entry — runs in the isolated ``t2i`` conda env (FLUX needs
    transformers<5, which conflicts with the trellis2 env's transformers 5.x).
    The trellis2 core invokes this as a subprocess and reads the saved PNG.

    Usage:
        python text_to_image.py --prompt "..." --out img.png [--seed 42]
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--model", default=_LOCAL_DEFAULT)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t2i = TextToImage(model_path=args.model)
    img = t2i.generate(
        args.prompt, seed=args.seed, steps=args.steps, guidance=args.guidance,
        size=args.size, max_seq_len=args.max_seq_len, out_path=args.out,
    )
    print(f"T2I_OK {img.size[0]}x{img.size[1]} -> {args.out}")


if __name__ == "__main__":
    _main()
