"""Resident offline workers for quality candidates and quantization comparisons."""
import argparse
import fcntl
import math
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / '.runtime'
MODEL_NAMES = ('qwen', 'qwen_int8', 'qwen_nf4', 'ernie', 'ernie_pe', 'zimage', 'hidream')

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def load_backend(name, torch, qwen_profile='lightning'):
    if qwen_profile not in ('lightning', 'base'):
        raise ValueError('Unknown Qwen sampling profile')
    base_name = 'qwen' if name.startswith('qwen') else ('ernie' if name == 'ernie_pe' else name)
    path = str(RUNTIME / 'models' / base_name)
    common = dict(torch_dtype=torch.bfloat16, local_files_only=True)
    if name.startswith('qwen'):
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
        class SplitQwenImagePipeline(QwenImagePipeline):
            @property
            def _execution_device(self):
                # The CPU text encoder is run explicitly before this pipeline.
                return torch.device('cuda:0')
        quant_args = {}
        if name != 'qwen':
            from diffusers import BitsAndBytesConfig
            class CastWarningOnce(logging.Filter):
                seen = False

                def filter(self, record):
                    if record.getMessage().startswith('MatMul8bitLt: inputs will be cast'):
                        if self.seen:
                            return False
                        self.seen = True
                    return True

            logging.getLogger('bitsandbytes.autograd._functions').addFilter(CastWarningOnce())
            quant = (BitsAndBytesConfig(load_in_8bit=True) if name == 'qwen_int8' else
                     BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                        bnb_4bit_compute_dtype=torch.bfloat16))
            quant_args = {'quantization_config': quant, 'device_map': {'': 0}}
        transformer = QwenImageTransformer2DModel.from_pretrained(
            path, subfolder='transformer', **common, **quant_args)
        pipe = SplitQwenImagePipeline.from_pretrained(path, transformer=transformer, **common)
        # Lightning was distilled with shift=3 and no terminal stretching.
        # The base-model scheduler leaves strong residual noise at eight steps.
        if qwen_profile == 'lightning':
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config, base_shift=math.log(3), max_shift=math.log(3),
                shift_terminal=None)
            pipe.load_lora_weights(str(RUNTIME / 'models' / 'qwen_lightning'),
                                   weight_name='Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors',
                                   local_files_only=True)
        # CUDA_VISIBLE_DEVICES is set before importing torch in main(): logical
        # 0 and 1 below map ONLY to the two explicitly selected physical GPUs.
        if name == 'qwen':
            from accelerate import dispatch_model
            blocks = len(pipe.transformer.transformer_blocks)
            device_map = {key: 0 for key, _ in pipe.transformer.named_children() if key != 'transformer_blocks'}
            device_map.update({f'transformer_blocks.{i}': int(i >= blocks // 2) for i in range(blocks)})
            pipe.transformer = dispatch_model(pipe.transformer, device_map=device_map, main_device=0)
        pipe.vae.to('cuda:0')
        # Preserve full encoder precision; avoid competing with either DiT half.
        torch.set_num_threads(8)
        pipe.vae.enable_tiling()
        config = {'steps': 8, 'true_cfg_scale': 1.0, 'dtype': 'bfloat16', 'quantization': 'none',
                  'offload': 'transformer split over two GPUs; text encoder on CPU',
                  'gpu_count': 2, 'cpu_threads': 8,
                  'resolution': [1024, 1024], 'prompt_enhancer': False,
                  'scheduler_shift': 3.0, 'scheduler_shift_terminal': None}
        if name != 'qwen':
            config.update(quantization='bitsandbytes INT8' if name == 'qwen_int8' else 'bitsandbytes NF4',
                          gpu_count=1, offload='quantized transformer GPU resident; text encoder CPU BF16')
        if qwen_profile == 'base':
            config.update(steps=50, true_cfg_scale=4.0, negative_prompt=' ', lightning=False,
                          scheduler=dict(pipe.scheduler.config))
            config.pop('scheduler_shift')
            config.pop('scheduler_shift_terminal')
        def infer(prompt, seed, width=1024, height=1024):
            embeds, mask = pipe.encode_prompt(prompt=prompt, device=torch.device('cpu'))
            embeds = embeds.to('cuda:0')
            mask = mask.to('cuda:0') if mask is not None else None
            negative = {}
            if qwen_profile == 'base':
                neg_embeds, neg_mask = pipe.encode_prompt(prompt=' ', device=torch.device('cpu'))
                negative = dict(negative_prompt_embeds=neg_embeds.to('cuda:0'),
                                negative_prompt_embeds_mask=neg_mask.to('cuda:0') if neg_mask is not None else None)
            result = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, width=width, height=height,
                        num_inference_steps=config['steps'], true_cfg_scale=config['true_cfg_scale'], **negative,
                        generator=torch.Generator('cuda').manual_seed(seed)).images[0]
            return result
        infer.pipeline = pipe
    elif name == 'ernie_pe':
        sys.path.insert(0, str(ROOT / 't2o_pipeline' / 'previz_pipeline'))
        from text_to_image import TextToImage
        worker = TextToImage(model_path=path, use_pe=True)
        worker.load()
        config = {'steps': 8, 'guidance_scale': 1.0, 'dtype': 'bfloat16', 'quantization': 'none',
                  'offload': 'model_cpu_offload', 'resolution': [1024, 1024],
                  'prompt_enhancer': True, 'pe_rng_seeded': True}
        def infer(prompt, seed):
            result = worker.generate(prompt, seed=seed)
            infer.revised_prompt = worker.last_revised_prompt
            return result
    elif name == 'ernie':
        from diffusers import ErnieImagePipeline
        pipe = ErnieImagePipeline.from_pretrained(path, pe=None, pe_tokenizer=None, **common)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        config = {'steps': 8, 'guidance_scale': 1.0, 'dtype': 'bfloat16', 'quantization': 'none',
                  'offload': 'model_cpu_offload', 'resolution': [1024, 1024], 'prompt_enhancer': False}
        def infer(prompt, seed):
            return pipe(prompt=prompt, width=1024, height=1024, num_inference_steps=8,
                        guidance_scale=1.0, use_pe=False,
                        generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    elif name == 'zimage':
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(path, **common)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        config = {'steps': 28, 'guidance_scale': 4.0, 'cfg_normalization': False, 'negative_prompt': '',
                  'dtype': 'bfloat16', 'quantization': 'none', 'offload': 'model_cpu_offload',
                  'resolution': [1024, 1024], 'prompt_enhancer': False}
        def infer(prompt, seed):
            return pipe(prompt=prompt, negative_prompt='', width=1024, height=1024,
                        num_inference_steps=28, guidance_scale=4.0, cfg_normalization=False,
                        generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    else:
        upstream = RUNTIME / 'HiDream-O1-Image'
        sys.path.insert(0, str(upstream))
        from inference import add_special_tokens, get_tokenizer
        from transformers import AutoProcessor
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from models import pipeline as hd
        from models import qwen3_vl_transformers as modeling
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cuda', local_files_only=True).eval()
        add_special_tokens(get_tokenizer(processor))
        # Official pipeline hardcodes FlashAttention; use its documented fallback
        # when that optional extension is unavailable, without editing upstream.
        has_flash = modeling._flash_attn_func is not None
        if not has_flash:
            def select_attention(_module, args, kwargs):
                kwargs['use_flash_attn'] = False
                return args, kwargs
            model.register_forward_pre_hook(select_attention, with_kwargs=True)
        config = {'steps': 28, 'guidance_scale': 0.0, 'shift': 1.0, 'scheduler': 'flash',
                  'noise_scale_start': 7.5, 'noise_scale_end': 7.5, 'noise_clip_std': 2.5,
                  'dtype': 'bfloat16', 'quantization': 'none', 'offload': 'none',
                  'resolution': [2048, 2048], 'attention': 'flash' if has_flash else 'upstream SDPA fallback',
                  'prompt_enhancer': False,
                  'upstream_revision': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()}
        def infer(prompt, seed):
            return hd.generate_image(model=model, processor=processor, prompt=prompt, ref_image_paths=[],
                                     width=2048, height=2048, num_inference_steps=28, guidance_scale=0,
                                     shift=1.0, timesteps_list=hd.DEFAULT_TIMESTEPS, scheduler_name='flash',
                                     noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5, seed=seed)
    return infer, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--gpu-second', type=int, help='Second idle physical GPU; required for Qwen BF16')
    parser.add_argument('--cases', type=Path, default=HERE / 'cases.json')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'comparison_v1')
    parser.add_argument('--limit', type=int, default=0, help='Maximum pending samples; zero means all')
    parser.add_argument('--prompt', help='Generate one custom prompt instead of the frozen comparison cases')
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    if args.model == 'qwen' and (args.gpu_second is None or args.gpu_second == args.gpu):
        parser.error('Qwen BF16 requires two distinct GPUs: --gpu N --gpu-second M')
    if args.model != 'qwen' and args.gpu_second is not None:
        parser.error('--gpu-second is only used by Qwen')
    selected = [args.gpu] + ([args.gpu_second] if args.gpu_second is not None else [])
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(RUNTIME / 'huggingface')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; execute with GPU access enabled.')
    for logical, physical in enumerate(selected):
        free, total = torch.cuda.mem_get_info(logical)
        if free < 22 * 1024**3:
            raise SystemExit(f'GPU {physical} has only {free / 1024**3:.1f} GiB free; choose an idle GPU.')
    suite = json.loads(args.cases.read_text())
    if args.prompt:
        suite = {'cases': [{'id': 'custom_' + hashlib.sha256(args.prompt.encode()).hexdigest()[:12],
                            'prompt': args.prompt}], 'seeds': [args.seed]}
    suite_hash = hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()
    out = args.output / args.model
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another worker is already writing this model output directory.')
    meta_path = out / 'metadata.json'
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if old and old['suite_sha256'] != suite_hash:
        raise SystemExit('Different prompts/seeds already exist here. Choose a new --output directory.')
    samples_path = out / 'samples.jsonl'
    prior = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
    done = {(r['case_id'], r['seed']) for r in prior if r['status'] == 'ok' and (out / r['image']).exists()}
    pending = [(case, seed) for case in suite['cases'] for seed in suite['seeds'] if (case['id'], seed) not in done]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print('All requested samples already exist.', flush=True)
        return
    manifests = {}
    base_model = 'qwen' if args.model.startswith('qwen') else ('ernie' if args.model == 'ernie_pe' else args.model)
    for name in [base_model] + (['qwen_lightning'] if base_model == 'qwen' else []):
        manifests[name] = json.loads((RUNTIME / 'manifests' / f'{name}.json').read_text())
        if manifests[name]['status'] != 'downloaded':
            raise SystemExit(f'{name} download has not completed.')
    if args.model == 'ernie_pe' and not manifests['ernie'].get('prompt_enhancer'):
        raise SystemExit('Download ERNIE with --include-pe first.')
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    infer, config = load_backend(args.model, torch)
    for logical in range(len(selected)):
        torch.cuda.synchronize(logical)
    load_sec = time.perf_counter() - t_load
    if old and old['config'] != config:
        raise SystemExit('Runtime settings changed; choose a new --output directory.')
    meta = {'model': args.model, 'suite_sha256': suite_hash, 'suite': suite,
            'config': config, 'manifests': manifests, 'gpu': torch.cuda.get_device_name(),
            'physical_gpu': args.gpu, 'physical_gpus': selected, 'gpu_total_gib': total / 1024**3,
            'load_sec': load_sec, 'load_peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'versions': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'diffusers', 'accelerate']}}
    atomic_json(meta_path, meta)
    print(f'LOADED {args.model} in {load_sec:.1f}s: {json.dumps(config)}', flush=True)
    failures = 0
    for index, (case, seed) in enumerate(pending):
        for logical in range(len(selected)):
            torch.cuda.reset_peak_memory_stats(logical)
        start = time.perf_counter()
        record = {'case_id': case['id'], 'prompt': case['prompt'], 'seed': seed, 'model': args.model,
                  'first_after_load': index == 0, 'session_load_sec': load_sec, 'config': config}
        filename = f'{case["id"]}_{seed}.png'
        try:
            with torch.inference_mode():
                image = infer(case['prompt'], seed)
            for logical in range(len(selected)):
                torch.cuda.synchronize(logical)
            record['infer_sec'] = time.perf_counter() - start
            if hasattr(infer, 'revised_prompt'):
                record['revised_prompt'] = infer.revised_prompt
            temp = out / (filename + '.tmp')
            image.save(temp, format='PNG')
            temp.replace(out / filename)
            record.update(status='ok', image=filename, width=image.width, height=image.height)
        except Exception as exc:
            failures += 1
            record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        record.update(total_sec=time.perf_counter() - start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                      peak_reserved_gib_per_gpu=[torch.cuda.max_memory_reserved(i) / 1024**3 for i in range(len(selected))])
        with samples_path.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
        print(json.dumps({k:v for k,v in record.items() if k not in ('traceback','prompt','config')}), flush=True)
        if record['status'] == 'failed':
            # Do not burn the whole suite on a broken backend.
            break
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
