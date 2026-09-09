"""ERNIE-Image-Turbo reference-image stage in the trellis2 conda environment.

Uses the visually selected BF16 / 8 steps / CFG 1 / 1024-square configuration.
Prompt enhancement is enabled by default, following the user's quality review.
The subprocess boundary returns VRAM before image-to-3D inference.
"""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from t2i_config import DEFAULT_T2I_MODEL_PATH, DEFAULT_T2I_STEPS, DEFAULT_T2I_GUIDANCE

_OFFLOAD_VRAM_THRESHOLD_GB = float(os.environ.get("T2I_OFFLOAD_THRESHOLD_GB", "40"))
_POSITIVE_SUFFIX = os.environ.get("T2I_POSITIVE_SUFFIX", "")
_USE_PE = os.environ.get('T2I_PROMPT_ENHANCER', '1').strip().lower() not in ('0', 'false', 'no')


class TextToImage:
    """Generate a TRELLIS.2 reference image using ERNIE-Image-Turbo."""

    def __init__(self, model_path: str = DEFAULT_T2I_MODEL_PATH,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 use_pe: bool = _USE_PE) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.use_pe = use_pe
        self.last_revised_prompt = None
        self.pipe = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import ErnieImagePipeline
        except ImportError as exc:
            raise ImportError("ERNIE requires diffusers==0.39.0 and transformers==5.16.1 "
                              "in trellis2; see docs/ERNIE_DEFAULT_SETUP.md") from exc
        logging.info("Loading ERNIE-Image-Turbo from %s", self.model_path)
        self.pipe = ErnieImagePipeline.from_pretrained(
            self.model_path, torch_dtype=self.dtype,
            **({} if self.use_pe else {'pe': None, 'pe_tokenizer': None}))
        if self.use_pe and (self.pipe.pe is None or self.pipe.pe_tokenizer is None):
            raise RuntimeError('Prompt enhancer is ON but its weights/tokenizer are missing; download pe/ and pe_tokenizer/.')
        device = torch.device(self.device)
        offload_env = os.environ.get("T2I_OFFLOAD")
        if offload_env is not None:
            offload = offload_env.strip().lower() in ("1", "true", "yes")
        elif device.type == "cuda" and torch.cuda.is_available():
            total_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
            offload = total_gib < _OFFLOAD_VRAM_THRESHOLD_GB
        else:
            offload = False
        if offload:
            if device.type != "cuda":
                raise ValueError("T2I_OFFLOAD requires a CUDA device")
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            self.pipe.to(device)
        self.pipe.vae.enable_tiling()
        logging.info("ERNIE ready: %s", "model CPU offload" if offload else str(device))

    def generate(self, prompt: str, seed: int = 42, steps: int = DEFAULT_T2I_STEPS,
                 guidance: float = DEFAULT_T2I_GUIDANCE, size: int = 1024,
                 out_path: Optional[str] = None) -> Image.Image:
        """Generate an RGB reference; retain the original and enhanced prompts."""
        if self.pipe is None:
            self.load()
        full_prompt = f"{prompt.strip()}. {_POSITIVE_SUFFIX}" if _POSITIVE_SUFFIX else prompt.strip()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        logging.info("ERNIE generate: seed=%s, steps=%s, size=%s", seed, steps, size)
        device = torch.device(self.device)
        index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == 'cuda' else None
        # The official PE samples from the global RNG, independently of the
        # diffusion generator. Seed and restore just the selected device.
        with torch.inference_mode(), torch.random.fork_rng(devices=[] if index is None else [index]):
            torch.random.default_generator.manual_seed(int(seed))
            if index is not None:
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(int(seed))
            result = self.pipe(prompt=full_prompt, height=size, width=size,
                              num_inference_steps=steps, guidance_scale=guidance,
                              use_pe=self.use_pe, generator=generator)
            image = result.images[0].convert('RGB')
        self.last_revised_prompt = result.revised_prompts[0] if result.revised_prompts else None
        if self.use_pe and not self.last_revised_prompt:
            raise RuntimeError('Prompt enhancement did not return an enhanced prompt')
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            Path(out_path).with_suffix('.json').write_text(json.dumps({
                'model': self.model_path, 'prompt': full_prompt,
                'revised_prompt': self.last_revised_prompt, 'prompt_enhancer': self.use_pe,
                'seed': int(seed), 'steps': steps, 'guidance': guidance, 'size': size,
            }, ensure_ascii=False, indent=2) + '\n')
        return image

    def unload(self) -> None:
        if self.pipe is not None:
            self.pipe = None
            torch.cuda.empty_cache()


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ERNIE-Image-Turbo reference image (trellis2 env)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_T2I_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_T2I_GUIDANCE)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--model", default=DEFAULT_T2I_MODEL_PATH,
                        help="ERNIE-Image-Turbo directory or HF repo ID")
    parser.add_argument('--prompt-enhancer', action=argparse.BooleanOptionalAction, default=_USE_PE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t2i = TextToImage(model_path=args.model, use_pe=args.prompt_enhancer)
    try:
        img = t2i.generate(args.prompt, seed=args.seed, steps=args.steps,
                           guidance=args.guidance, size=args.size, out_path=args.out)
        print(f"T2I_OK {img.size[0]}x{img.size[1]} -> {args.out}")
    finally:
        t2i.unload()


if __name__ == "__main__":
    _main()
