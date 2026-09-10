"""Batch scheduling while retaining existing per-image version/job identifiers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
ACTIVE=('queued','waiting','running','cancelling')
RETRY=('failed','cancelled','interrupted')


def members(studio, bid):
    return sorted([j for j in studio.store.list('job') if j.get('batch_id')==bid],key=lambda j:j['candidate_index'])


def control(studio, bid, action):
    with studio.lock:
        batch=studio.store.get(bid,'batch')
        jobs=members(studio,bid)
        if action=='retry' and any(j['state'] in ACTIVE for j in jobs):
            raise ValueError('진행 중인 묶음은 재시도할 수 없습니다.')
        changed=False
        for j in jobs:
            if action=='retry' and j['state'] in RETRY:
                studio.store.update(j['id'],state='queued',ended=None,message='미완료 후보를 이어서 생성합니다.')
                studio.store.update(j['version'],state='queued'); changed=True
            elif action=='cancel' and j['state'] in ACTIVE:
                state='cancelling' if j['state'] in ('running','cancelling') else 'cancelled'
                studio.store.update(j['id'],state=state,ended=None if state=='cancelling' else time.time(),message='후보 묶음 중지')
                studio.store.update(j['version'],state=state); changed=True
        if not changed:
            raise ValueError('처리할 미완료 후보가 없습니다.')
        if action=='cancel' and studio.current in [j['id'] for j in jobs] and studio.proc and studio.proc.poll() is None:
            os.killpg(studio.proc.pid,signal.SIGTERM)
        studio.store.update(bid,queued_at=time.time() if action=='retry' else batch.get('queued_at',batch['created']),state='queued' if action=='retry' else 'cancelling' if any(j['state']=='running' for j in jobs) else 'cancelled')
        studio.wake.set()
    return {'ok':True,'batch_id':bid}


def ingest(studio,batch,folder):
    path=folder/'progress.json'
    if path.is_file():
        value=json.loads(path.read_text())
        studio.store.update(batch['id'],progress=value)
        for j in members(studio,batch['id']):
            if j['state']=='running':
                studio.store.update(j['id'],message=value.get('message','후보 생성 중'))
    shared_path=folder/'shared.json'
    if shared_path.is_file():
        studio.store.update(batch['id'],shared=json.loads(shared_path.read_text()))
    for j in members(studio,batch['id']):
        dest=studio.store.folder(j['version'])
        marker=dest/'reference.metrics.json'
        if j['state']!='complete' and marker.is_file() and (dest/'reference.png').is_file():
            v=studio.store.get(j['version']); metrics=v.get('metrics',{})
            metrics['reference']=json.loads(marker.read_text())
            metrics['caption']=json.loads((dest/'caption.metrics.json').read_text())
            studio.store.update(j['version'],state='complete',reference=str(dest/'reference.png'),metrics=metrics)
            if j['state']!='complete':
                studio.store.update(j['id'],state='complete',ended=time.time(),completed=['caption','reference'],message='완료')


def run_batch(studio,job):
    bid=job['batch_id']; batch=studio.store.get(bid,'batch'); folder=studio.store.folder(bid)
    studio.current=job['id']
    with studio.lock:
        jobs=members(studio,bid)
        if job['id'] not in [j['id'] for j in jobs if j['state'] in ('queued','waiting')]:
            studio.current=None; return
        attempt=batch.get('attempt',0)+1; now=time.time()
        previous=batch.get('progress')
        history=batch.get('attempts',[])+([previous] if previous else [])
        studio.store.update(bid,state='running',started=now,ended=None,attempt=attempt,attempts=history,progress=None)
        for j in jobs:
            if j['state'] in ('queued','waiting'):
                studio.store.update(j['id'],state='running',started=now,ended=None,stage='reference',message='실행 준비')
                studio.store.update(j['version'],state='running')
        candidates=[{**studio.store.get(j['version']),'folder':str(studio.store.folder(j['version']))} for j in jobs]
        spec=dict(folder=str(folder),batch_id=bid,count=batch['count'],pe_seed=batch['pe_seed'],
                  candidates=candidates,attempt=attempt,profile_path=str(HERE/'candidate_profile.json'))
        (folder/'spec.json').write_text(json.dumps(spec,ensure_ascii=False))
        (folder/'progress.json').unlink(missing_ok=True)
    state='failed'; message='후보 생성에 실패했습니다. 미완료 후보를 재시도할 수 있습니다.'
    try:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':studio.gpu,'PYTHONUNBUFFERED':'1',
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'8',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
        with (folder/'worker.log').open('ab') as log:
            with studio.lock:
                if any(j['state']=='cancelling' for j in members(studio,bid)):
                    raise InterruptedError()
                studio.proc=subprocess.Popen([studio.python,str(HERE/'candidate_worker.py'),str(folder/'spec.json')],
                    stdout=log,stderr=subprocess.STDOUT,env=env,cwd=HERE.parent,start_new_session=True)
            deadline=time.monotonic()+1800; cancel_started=None
            while studio.proc.poll() is None:
                with studio.lock:
                    ingest(studio,batch,folder)
                    cancel=studio.stop.is_set() or any(j['state']=='cancelling' for j in members(studio,bid))
                if cancel:
                    if cancel_started is None:
                        cancel_started=time.monotonic(); os.killpg(studio.proc.pid,signal.SIGTERM)
                    elif time.monotonic()-cancel_started>5:
                        os.killpg(studio.proc.pid,signal.SIGKILL)
                if time.monotonic()>deadline:
                    os.killpg(studio.proc.pid,signal.SIGKILL); studio.proc.wait()
                    raise RuntimeError('후보 묶음 실행 제한 시간(30분)을 초과했습니다.')
                time.sleep(.5)
            ingest(studio,batch,folder)
            if studio.stop.is_set() or any(j['state']=='cancelling' for j in members(studio,bid)) or cancel_started:
                raise InterruptedError()
            if studio.proc.returncode:
                raise RuntimeError(message)
            if not all(j['state']=='complete' for j in members(studio,bid)):
                raise RuntimeError('완료 파일이 없는 후보가 있습니다.')
            state='complete'; message='완료'
    except InterruptedError:
        state='interrupted' if studio.stop.is_set() else 'cancelled'
        message='묶음 중지됨 · 완료된 후보는 보존했습니다.'
    except Exception as exc:
        message=str(exc)
    finally:
        with studio.lock:
            ingest(studio,batch,folder)
            if all(j['state']=='complete' for j in members(studio,bid)):
                state='complete'; message='완료'
            for j in members(studio,bid):
                if j['state']!='complete':
                    studio.store.update(j['id'],state=state,ended=time.time(),message=message)
                    studio.store.update(j['version'],state=state)
            progress=studio.store.get(bid).get('progress') or {}
            progress.update(state=state,ended=time.time(),message=message,
                            completed=sum(j['state']=='complete' for j in members(studio,bid)))
            for phase in progress.get('phases',[]):
                if phase['state']=='running':
                    phase.update(state=state,ended=time.time(),seconds=max(0,time.time()-phase['started']))
            studio.store.update(bid,state=state,ended=time.time(),progress=progress)
            studio.proc=None; studio.current=None
