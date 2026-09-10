"""One GPU stage per process: references, image preprocessing, lift, or bake.

Uses the main prompt builder, ERNIE implementation and TRELLIS.2 loader. The
approved reference never changes during a lift retry. Checkpoints are local
numeric NPZ, not user-supplied pickle. Each process exits to release GPU memory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / 'previz_pipeline'), str(REPO / 'trellis2_src')]
os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['caption', 'reference', 'prepare', 'mesh', 'export'])
    parser.add_argument('spec')
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    folder = Path(spec['folder'])
    if args.stage == 'caption':
        from caption import clean_caption
        write_json(folder/'caption.metrics.json',clean_caption(spec))
        return
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(8)
    start = time.monotonic()
    import hashlib
    import importlib.metadata
    metrics = {'stage': args.stage, 'device':torch.cuda.get_device_name(0),
               'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
               'torch':torch.__version__,
               'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.stage == 'reference':
        caption = json.loads((folder/'caption.metrics.json').read_text())
        from t2i_prompt_builder import build_t2i_prompt
        prompt, body_plan = build_t2i_prompt(
            object_name=spec['object_name'], appearance=None,
            base_description=caption['cleaned_prompt'], category=spec.get('category', ''),
            rig_type=spec.get('rig_type') or None)
        if spec['model'] == 'ernie':
            from text_to_image import TextToImage
            model = TextToImage(use_pe=True)
            model.generate(prompt, seed=spec['seed'], out_path=str(folder / 'reference.png'))
            enhanced = model.last_revised_prompt
            metrics['model'] = model.model_path
        else:
            sys.path.insert(0, str(REPO / 'experiments/t2i_quality'))
            from run_model import load_backend
            infer, config = load_backend('qwen_int8', torch)
            with torch.inference_mode():
                infer(prompt, spec['seed']).save(folder / 'reference.png')
            enhanced = None
            metrics['configuration'] = config
        metrics.update(prompt_used=prompt, revised_prompt=enhanced, body_plan=body_plan,
                       prompt_enhancer=spec['model'] == 'ernie', seed=spec['seed'],
                       raw_prompt=spec['prompt'],cleaned_prompt=caption['cleaned_prompt'])
        metrics['diffusers'] = importlib.metadata.version('diffusers')
        metrics['template_sha256'] = hashlib.sha256((REPO/'previz_pipeline/t2i_prompt_builder.py').read_bytes()).hexdigest()
    elif args.stage == 'prepare':
        # The same rembg class/checkpoint as the main core, without loading 3D weights.
        from trellis2_inference_core import _REMBG_MIT
        from trellis2.pipelines.rembg import BiRefNet
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        image = Image.open(spec['reference']).convert('RGBA')
        alpha = np.asarray(image.getchannel('A'))
        if np.all(alpha == 255):
            remover = BiRefNet(model_name=_REMBG_MIT)
            remover.to('cuda')
            cutout = remover(image.convert('RGB'))
        else:
            cutout = image
        if not np.any(np.asarray(cutout.getchannel('A')) > 204):
            raise ValueError('배경 제거 후 객체가 남지 않았습니다. 다른 참조 이미지를 선택하세요.')
        cutout.save(folder / 'cutout.png')
        # The alpha-bearing image takes the exact preprocessing crop path without rembg.
        from types import SimpleNamespace
        context = SimpleNamespace(low_vram=False,rembg_model=lambda _: cutout)
        prepared = Trellis2ImageTo3DPipeline.preprocess_image(context, cutout)
        prepared.save(folder / 'prepared.png')
        metrics['rembg'] = _REMBG_MIT
    elif args.stage == 'mesh':
        from trellis2_inference_core import Trellis2InferenceCore
        core = Trellis2InferenceCore(base_output_dir=str(folder / 'scratch'))
        core.load_pipeline()
        image = Image.open(folder / 'prepared.png').convert('RGB')
        def lift(seed):
            with torch.inference_mode():
                mesh = core.pipeline.run(image, seed=seed, preprocess_image=False)[0]
                mesh.simplify(16777216)
            ext = (mesh.vertices.max(0).values - mesh.vertices.min(0).values).detach().cpu().numpy()
            return mesh, float(ext.min() / max(float(ext.max()), 1e-9))
        mesh, ratio = lift(spec['seed'])
        actual_seed = spec['seed']
        if ratio < 0.02:
            retry, retry_ratio = lift(spec['seed'] + 7919)
            if retry_ratio > ratio:
                mesh, ratio, actual_seed = retry, retry_ratio, spec['seed'] + 7919
            metrics['gate_retry'] = True
        temp = folder / 'mesh.tmp.npz'
        np.savez(temp, vertices=mesh.vertices.detach().cpu().numpy().astype(np.float32),
                 faces=mesh.faces.detach().cpu().numpy().astype(np.int32),
                 coords=mesh.coords.detach().cpu().numpy().astype(np.int32),
                 attrs=mesh.attrs.detach().cpu().numpy().astype(np.float32),
                 voxel_size=np.float64(mesh.voxel_size),
                 layout_json=np.asarray(json.dumps({k: [v.start, v.stop] for k, v in mesh.layout.items()})))
        temp.replace(folder / 'mesh.npz')
        metrics.update(thickness_ratio=ratio, actual_seed=actual_seed,
                       warning='깊이가 매우 얇습니다. 측면을 확인하세요.' if ratio < 0.02 else '')
        metrics.update(model=core.model_path,pipeline_type=core.pipeline_type or '1024_cascade',
                       pipeline_config_sha256=hashlib.sha256((Path(core.model_path)/'pipeline.json').read_bytes()).hexdigest())
    else:
        from trellis2_inference_core import _ensure_o_voxel_commercial_override
        _ensure_o_voxel_commercial_override()
        import o_voxel
        with np.load(folder / 'mesh.npz', allow_pickle=False) as data:
            glb = o_voxel.postprocess.to_glb(
                vertices=torch.from_numpy(data['vertices']).cuda(),
                faces=torch.from_numpy(data['faces']).cuda(),
                coords=torch.from_numpy(data['coords']).cuda(),
                attr_volume=torch.from_numpy(data['attrs']).cuda(),
                attr_layout={k: slice(*v) for k, v in json.loads(str(data['layout_json'])).items()},
                voxel_size=float(data['voxel_size']), aabb=[[-0.5]*3, [0.5]*3],
                decimation_target=spec['faces'], texture_size=spec['texture'],
                remesh=True, remesh_band=1, remesh_project=0)
        temporary = folder / 'result.tmp.glb'
        glb.export(str(temporary), extension_webp=False)
        temporary.replace(folder / 'result.glb')
        metrics.update(faces_target=spec['faces'], texture_size=spec['texture'])
    torch.cuda.synchronize()
    metrics.update(seconds=round(time.monotonic() - start, 3),
                   peak_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3, 3))
    write_json(folder / f'{args.stage}.metrics.json', metrics)


if __name__ == '__main__':
    main()
