"""Measure resident sequential vs batch generation on the Studio GPU, offline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='1')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lock = (root / '.runtime' / f'generation-studio-gpu-{args.gpu}.lock').open('a')
    print('Waiting for Studio GPU lock', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    import torch
    from t2i_batch_backend import Backend
    torch.set_num_threads(8)
    prompt = 'A car, car, single centered object, neutral background, full object visible in frame, unoccluded, three-quarter view'
    seeds = [197861, 197862, 197863, 197864]
    profile = {'revision': 1, 'gpu': torch.cuda.get_device_name(0),
               'total_memory': torch.cuda.get_device_properties(0).total_memory,
               'torch': torch.__version__, 'models': {}}
    for name in ('ernie', 'qwen_int8'):
        start = time.monotonic(); backend = Backend(name); backend.sync()
        result = {'load_seconds': time.monotonic()-start, 'prompt': prompt, 'seeds': seeds, 'config': backend.config, 'runs': []}
        start=time.monotonic(); revised=backend.enhance(prompt, seeds[0]); backend.sync()
        result.update(revised_prompt=revised, pe_seconds=time.monotonic()-start)
        start=time.monotonic(); backend.encode(revised or prompt); backend.sync()
        result['encode_seconds']=time.monotonic()-start
        backend.generate(seeds[:1]); backend.sync()  # Separate warm-up, excluded from comparisons.
        for size in (1, 2, 3, 4):
            run={'chunk_size':size,'chunks':[], 'ok':True}
            torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
            for offset in range(0, 4, size):
                batch_seeds=seeds[offset:offset+size]; tick=time.monotonic(); oom=False
                try:
                    images=backend.generate(batch_seeds); backend.sync()
                    seconds=time.monotonic()-tick
                    for i, im in enumerate(images):
                        im.save(out/f'{name}_chunk{size}_seed{batch_seeds[i]}.png')
                    del images
                    run['chunks'].append({'count':len(batch_seeds),'seconds':seconds})
                except torch.OutOfMemoryError:
                    run['ok']=False; oom=True
                if oom:
                    backend.clear_oom(); break
            run.update(wall_seconds=time.monotonic()-start, **backend.peaks())
            result['runs'].append(run)
            print(json.dumps({'model':name,**run}), flush=True)
            (out/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        # Re-use measured per-chunk costs to choose fixed chunk size for N=2..4.
        costs={}
        for run in result['runs']:
            if run['ok']:
                for c in run['chunks']:
                    costs.setdefault(c['count'],[]).append(c['seconds'])
        costs={n:sum(v)/len(v) for n,v in costs.items()}
        choices={}
        for n in (2,3,4):
            feasible={k:sum(costs[min(k,n-i)] for i in range(0,n,k)) for k in costs
                      if k<=n and all(min(k,n-i) in costs for i in range(0,n,k))}
            choices[str(n)]=min(feasible,key=feasible.get)
        profile['models'][name]={'chunk_sizes':choices,'chunk_seconds':costs,
                                 'measured_runs':result['runs'], 'config':backend.config}
        (out/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2))
        del backend
        import gc
        gc.collect(); torch.cuda.empty_cache()
    print('BENCHMARK_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
