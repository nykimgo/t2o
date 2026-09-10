"""Shared-prompt candidate adapter; same checkpoints/sampling as production T2I."""
import gc
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'previz_pipeline'))


class Backend:
    def __init__(self, name):
        import torch
        self.torch = torch
        self.name = name
        if name == 'ernie':
            from text_to_image import TextToImage
            self.owner = TextToImage(use_pe=True)
            self.owner.load()
            self.pipe = self.owner.pipe
            self.config = dict(model=self.owner.model_path, steps=8, guidance_scale=1.0,
                               resolution=[1024, 1024], prompt_enhancer=True)
        elif name == 'qwen_int8':
            sys.path.insert(0, str(REPO / 'experiments/t2i_quality'))
            from run_model import load_backend
            infer, self.config = load_backend(name, torch)
            self.pipe = infer.pipeline
        else:
            raise ValueError('Unknown candidate model')
        self.embeds = None
        self.mask = None

    def sync(self):
        self.torch.cuda.synchronize()

    def enhance(self, prompt, seed):
        if self.name != 'ernie':
            return None
        torch = self.torch
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            result = self.pipe._enhance_prompt_with_pe(prompt, self.pipe._execution_device,
                                                       width=1024, height=1024)
        if not result or not result.strip():
            raise ValueError('프롬프트 자동 보강 결과가 비어 있습니다.')
        return result

    def encode(self, prompt):
        torch = self.torch
        with torch.inference_mode():
            if self.name == 'ernie':
                self.embeds = self.pipe.encode_prompt(prompt, self.pipe._execution_device)
            else:
                embeds, mask = self.pipe.encode_prompt(prompt=prompt, device=torch.device('cpu'))
                self.embeds = embeds.to('cuda:0')
                self.mask = mask.to('cuda:0') if mask is not None else None

    def generate(self, seeds, progress=None):
        torch = self.torch
        def step(pipe, index, timestep, values):
            if progress:
                progress(index + 1, self.config['steps'])
            return values
        kwargs = dict(prompt_embeds=self.embeds, width=1024, height=1024,
                      num_images_per_prompt=len(seeds), num_inference_steps=self.config['steps'],
                      generator=[torch.Generator('cuda:0').manual_seed(s) for s in seeds],
                      callback_on_step_end=step)
        if self.name == 'ernie':
            # Enhancement has already run once and is persisted before diffusion.
            kwargs.update(use_pe=False, guidance_scale=1.0)
        else:
            kwargs.update(prompt_embeds_mask=self.mask, true_cfg_scale=1.0)
        with torch.inference_mode():
            images = self.pipe(**kwargs).images
        if len(images) != len(seeds):
            raise RuntimeError('이미지 수와 seed 수가 일치하지 않습니다.')
        return [im.convert('RGB') for im in images]

    def clear_oom(self):
        # OOM tracebacks must be released by the caller before entering here.
        self.pipe.maybe_free_model_hooks()
        gc.collect()
        self.torch.cuda.empty_cache()

    def peaks(self):
        return dict(peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=self.torch.cuda.max_memory_reserved()/1024**3)
