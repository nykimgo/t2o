"""One resident model per candidate set, with durable shared prompt and progress."""
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'previz_pipeline'))


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


class Progress:
    def __init__(self, folder, attempt):
        self.path = folder/'progress.json'
        self.data = {'attempt':attempt, 'phases':[], 'started':time.time(), 'state':'running'}
        self.sync = lambda: None
        self.write()

    def write(self):
        self.data['updated'] = time.time()
        atomic_json(self.path, self.data)

    @contextmanager
    def phase(self, key, label, **extra):
        self.sync()
        entry = dict(key=key, label=label, started=time.time(), state='running', **extra)
        self.data['phases'].append(entry)
        self.data['message'] = label
        tick = time.monotonic(); self.write()
        try:
            yield entry
            self.sync()
        except BaseException:
            entry['state']='failed'
            raise
        else:
            entry['state']='complete'
        finally:
            entry.update(seconds=time.monotonic()-tick, ended=time.time())
            self.write()

    def step(self, done, total):
        self.data['phases'][-1].update(step=done, steps=total)
        self.write()


def choose_chunk(model, count, profile_path):
    try:
        value=json.loads(Path(profile_path).read_text())
        import torch
        if value['revision'] != 1 or value['gpu'] != torch.cuda.get_device_name(0):
            return 1
        if value['total_memory'] > torch.cuda.get_device_properties(0).total_memory:
            return 1
        return max(1,min(count,int(value['models'][model]['chunk_sizes'][str(count)])))
    except (OSError, ValueError, KeyError):
        return 1


def run(spec, backend_factory=None):
    folder=Path(spec['folder'])
    progress=Progress(folder, spec.get('attempt',1))
    try:
        with progress.phase('startup','실행 준비', shared=True):
            import torch
            torch.set_num_threads(8)
            if backend_factory is None:
                from t2i_batch_backend import Backend
                backend_factory=Backend
        shared_path=folder/'shared.json'
        shared=json.loads(shared_path.read_text()) if shared_path.exists() else {}
        first=spec['candidates'][0]
        with progress.phase('caption','외형 정제',shared=True) as phase:
            if 'caption' not in shared:
                from caption import clean_caption
                shared['caption']=clean_caption(first)
                atomic_json(shared_path,shared)
                phase['cached']=shared['caption'].get('cached',False)
            else:
                phase['cached']=True
            from t2i_prompt_builder import build_t2i_prompt
            if 'prompt_used' not in shared:
                prompt, body_plan=build_t2i_prompt(object_name=first['object_name'], appearance=None,
                    base_description=shared['caption']['cleaned_prompt'],category=first.get('category',''),
                    rig_type=first.get('rig_type') or None)
                if first['model']=='ernie':
                    suffix=os.environ.get('T2I_POSITIVE_SUFFIX','')
                    if suffix: prompt=f'{prompt.strip()}. {suffix}'
                shared.update(prompt_used=prompt,body_plan=body_plan,pe_seed=spec['pe_seed'])
                atomic_json(shared_path,shared)
        with progress.phase('load','모델 로딩',shared=True):
            backend=backend_factory(first['model'])
            progress.sync=backend.sync
        with progress.phase('enhance','프롬프트 보강',shared=True) as phase:
            if first['model']!='ernie':
                phase['skipped']=True
                shared['revised_prompt']=None
            elif 'revised_prompt' in shared:
                phase['cached']=True
            else:
                shared['revised_prompt']=backend.enhance(shared['prompt_used'],shared['pe_seed'])
            atomic_json(shared_path,shared)
        with progress.phase('encode','텍스트 인코딩',shared=True):
            backend.encode(shared['revised_prompt'] or shared['prompt_used'])
        pending=[c for c in spec['candidates'] if not complete(c)]
        size=choose_chunk(first['model'],spec['count'],spec['profile_path'])
        progress.data.update(chunk_size=size,count=spec['count'],completed=spec['count']-len(pending))
        while pending:
            candidates=pending[:size]; seeds=[c['seed'] for c in candidates]
            oom=False
            try:
                with progress.phase('generate',f'{len(candidates)}장 '+('함께 ' if len(candidates)>1 else '')+f"생성 중 · 전체 {spec['count']}장",
                                    shared=False, seeds=seeds, count=len(candidates)):
                    images=backend.generate(seeds,progress.step)
            except torch.OutOfMemoryError:
                oom=True
            if oom:
                backend.clear_oom()
                if size==1: raise RuntimeError('GPU 메모리가 부족합니다. 완료된 후보는 보존했습니다.')
                size=max(1,size//2)
                progress.data.update(chunk_size=size,notice=f'메모리 여유에 맞춰 {size}장씩 처리합니다.')
                progress.write()
                continue
            with progress.phase('save','이미지 저장',shared=False,seeds=seeds,count=len(candidates)):
                for c, image in zip(candidates,images):
                    dest=Path(c['folder']); dest.mkdir(parents=True,exist_ok=True)
                    image.save(dest/'reference.tmp.png')
                    (dest/'reference.tmp.png').replace(dest/'reference.png')
                    metric=dict(raw_prompt=c['prompt'],cleaned_prompt=shared['caption']['cleaned_prompt'],
                                prompt_used=shared['prompt_used'],body_plan=shared['body_plan'],
                                revised_prompt=shared['revised_prompt'],prompt_enhancer=c['model']=='ernie',
                                seed=c['seed'],pe_seed=shared['pe_seed'],batch_id=spec['batch_id'],
                                shared_prompt=True,configuration=backend.config,**backend.peaks())
                    atomic_json(dest/'caption.metrics.json',shared['caption'])
                    # The marker is written last; only marked images can be reused.
                    atomic_json(dest/'reference.metrics.json',metric)
            del images
            pending=pending[len(candidates):]
            progress.data['completed']=spec['count']-len(pending)
            progress.write()
        progress.data.update(state='complete',ended=time.time(),message='완료')
    except BaseException:
        progress.data.update(state='failed',ended=time.time(),message='생성 실패 · 완료된 후보는 보존했습니다.')
        raise
    finally:
        progress.write()


def complete(candidate):
    folder=Path(candidate['folder'])
    return (folder/'reference.png').is_file() and (folder/'reference.metrics.json').is_file()


if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
