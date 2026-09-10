"""LAN review surface for USD-generated assets. Run via generation_ui/manage.py."""
import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit
import zipfile

from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from store import Store

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROJECT = REPO.parent
STAGES = {'caption':'외형 설명 정제', 'reference': '참조 이미지 생성', 'prepare': '배경 제거', 'mesh': '3D 형상 생성', 'export': '텍스처·GLB 저장'}
ACTIVE = ('queued', 'waiting', 'running', 'cancelling')


def safe_name(value):
    return re.sub(r'[^\w가-힣 .-]', '_', str(value)).strip(' .')[:80] or 'asset'


def glb_chunks(path):
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b'glTF' or struct.unpack_from('<II', data, 4) != (2, len(data)):
        raise ValueError('올바른 GLB 2.0 파일이 아닙니다.')
    chunks, pos = [], 12
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError('GLB 파일이 불완전합니다.')
        size, kind = struct.unpack_from('<II', data, pos)
        pos += 8
        if pos + size > len(data):
            raise ValueError('GLB 파일이 불완전합니다.')
        chunks.append((kind, data[pos:pos+size]))
        pos += size
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ValueError('GLB 구성 정보가 없습니다.')
    doc = json.loads(chunks[0][1])
    for item in doc.get('images', []) + doc.get('buffers', []):
        if item.get('uri') and not item['uri'].startswith('data:'):
            raise ValueError('외부 파일을 참조하는 GLB입니다. 텍스처를 포함해 다시 내보내세요.')
    return doc, chunks


def export_glb(path, height=None, pivot='keep', yaw=0):
    if height is None and pivot == 'keep' and yaw == 0:
        return Path(path).read_bytes()
    import numpy as np
    import trimesh
    doc, chunks = glb_chunks(path)
    scene = trimesh.load(str(path), force='scene', process=False)
    bounds = np.asarray(scene.bounds)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError('에셋 크기를 계산할 수 없습니다.')
    scale = height / max(float(bounds[1, 1] - bounds[0, 1]), 1e-9) if height is not None else 1.0
    center = bounds.mean(0)
    if pivot == 'floor':
        center[1] = bounds[0, 1]
    if pivot == 'keep':
        center[:] = 0
    angle = math.radians(yaw)
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = -rotation @ center * scale
    scene_doc = doc['scenes'][doc.get('scene', 0)]
    # A node named "world" collides with trimesh's synthetic base frame when
    # wrapped. Rename that generic node; node indices/skin/animation links stay intact.
    for i,node in enumerate(doc.get('nodes',[])):
        if node.get('name') == 'world':
            node['name'] = f'Studio_source_root_{i}'
    index = len(doc.setdefault('nodes', []))
    doc['nodes'].append({'name': 'Studio_export', 'matrix': matrix.T.flatten().tolist(),
                         'children': scene_doc.get('nodes', [])})
    scene_doc['nodes'] = [index]
    encoded = json.dumps(doc, ensure_ascii=False, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    chunks[0] = (0x4E4F534A, encoded)
    body = b''.join(struct.pack('<II', len(buf), kind) + buf for kind, buf in chunks)
    return b'glTF' + struct.pack('<II', 2, len(body) + 12) + body


class Studio:
    def __init__(self, root, gpu='1', python=None, import_roots=None):
        self.store = Store(root)
        self.gpu = str(gpu)
        self.python = python or sys.executable
        self.import_roots = [Path(p).resolve() for p in (import_roots or [PROJECT / 'movie_usd', REPO / 't2o_results'])]
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.proc = None
        self.current = None

    def allowed(self, path, suffixes):
        p = Path(path).expanduser().resolve()
        if not any(p.is_relative_to(r) for r in self.import_roots):
            raise ValueError('등록된 결과 폴더 안의 파일만 연결할 수 있습니다.')
        if not p.is_file() or p.suffix.lower() not in suffixes or p.stat().st_size > 250*1024**2:
            raise ValueError('지원하지 않는 파일 또는 크기입니다.')
        return p

    def media(self, version, key):
        value = version.get(key)
        if not value:
            raise KeyError(key)
        p = Path(value).resolve()
        if not p.is_relative_to(self.store.root / 'files') or not p.is_file():
            raise KeyError(key)
        return p

    def version_json(self, v):
        result = {k: v.get(k) for k in ['id','parent','created','label','base_version','prompt','object_name',
                 'model','seed','rig_type','source','source_object','approved','warning','metrics','faces','texture','state','batch_id','candidate_index','candidate_count']}
        result['prompt_trace'] = self.prompt_trace(v)
        result['media'] = {key: f"/api/versions/{v['id']}/files/{key}" for key in
                           ('original','reference','cutout','prepared','glb') if v.get(key) and Path(v[key]).is_file()}
        return result

    def prompt_trace(self, version):
        v, seen = version, set()
        while v['id'] not in seen:
            seen.add(v['id'])
            metric=v.get('metrics',{}).get('reference')
            if metric:
                return {'source_version':v['id'],'model':v.get('model'),'seed':v.get('seed'),
                        'raw_prompt':metric.get('raw_prompt',v.get('prompt')),
                        'cleaned_prompt':metric.get('cleaned_prompt'),
                        'template_prompt':metric.get('prompt_used'),
                        'enhanced_prompt':metric.get('revised_prompt'),
                        'enhancer':metric.get('prompt_enhancer'),'pe_seed':metric.get('pe_seed'),
                        'shared_prompt':metric.get('shared_prompt',False)}
            if v.get('model') in ('ernie','qwen_int8') and not v.get('reference'):
                shared={}
                if v.get('batch_id'):
                    try:
                        shared=self.store.get(v['batch_id'],'batch').get('shared',{})
                    except KeyError:
                        pass  # Historical independent candidate sets have no batch record.
                if shared.get('prompt_used'):
                    return {'source_version':v['id'],'model':v.get('model'),'seed':v.get('seed'),
                            'raw_prompt':v.get('prompt'),'cleaned_prompt':shared['caption']['cleaned_prompt'],
                            'template_prompt':shared['prompt_used'],'enhanced_prompt':shared.get('revised_prompt'),
                            'enhancer':v['model']=='ernie','pe_seed':shared.get('pe_seed'),'shared_prompt':True}
                return {'source_version':v['id'],'model':v.get('model'),'seed':v.get('seed'),
                        'raw_prompt':v.get('prompt'),
                        'cleaned_prompt':v.get('metrics',{}).get('caption',{}).get('cleaned_prompt'),
                        'enhancer':v.get('model')=='ernie',
                        'note':'이 후보의 보강문은 아직 생성되지 않았습니다.'}
            if v.get('model')=='업로드':
                return {'source_version':v['id'],'model':'업로드','raw_prompt':v.get('prompt'),
                        'enhancer':False,'note':'업로드 이미지에는 자동 보강문이 없습니다.'}
            parent=v.get('base_version')
            if not parent:
                meta=v.get('metrics',{}).get('source_metadata',{})
                return {'source_version':v['id'],'model':v.get('model'),'raw_prompt':v.get('prompt'),
                        'template_prompt':meta.get('prompt_used'),'enhanced_prompt':None,
                        'note':'이 과거 결과에는 보강문 기록이 없습니다.'}
            v=self.store.get(parent,'version')
        return {}

    def import_asset(self, payload):
        glb = self.allowed(payload['glb_path'], {'.glb'})
        glb_chunks(glb)
        meta_path = Path(str(glb) + '.meta.json')
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        reference = payload.get('reference_path') or meta.get('reference_image')
        ref = self.allowed(reference, {'.png','.jpg','.jpeg','.webp'}) if reference and Path(reference).is_file() else None
        fingerprint = hashlib.sha256((str(glb)+str(glb.stat().st_mtime_ns)+str(glb.stat().st_size)).encode()).hexdigest()
        with self.lock:
            with self.store.connect() as c:
                old = c.execute('SELECT version FROM imports WHERE fingerprint=?', (fingerprint,)).fetchone()
            if old:
                return self.store.get(old['version']), False
            source_key = payload.get('source_key') or str(glb.parent)
            asset = next((a for a in self.store.list('asset') if a.get('source_key') == source_key
                          or a.get('import_directory') == str(glb.parent)), None)
            raw_name = re.sub(r'^scene_\d+_', '', glb.stem)
            raw_name = re.sub(r'_\d+$', '', raw_name)
            display = {'seagull':'갈매기', 'car':'자동차', 'airplane':'비행기'}.get(raw_name, raw_name.replace('_',' '))
            if not asset:
                asset = self.store.create('asset', {'name': payload.get('name') or display,
                    'object_name': raw_name, 'source_key': source_key, 'import_directory':str(glb.parent), 'source': 'USD 연계',
                    'project': payload.get('project') or (' / '.join(glb.relative_to(self.import_roots[0]).parts[:2])
                        if glb.is_relative_to(self.import_roots[0]) else '기존 생성 결과'), 'favorite': False, 'notes': ''})
            v = self.store.create('version', {'label':'USD 생성 결과', 'source':'USD 연계',
                'prompt': meta.get('description_en') or meta.get('prompt_used') or '',
                'object_name': asset.get('object_name', raw_name), 'model':'기존 생성', 'seed':meta.get('seed'),
                'source_object': payload.get('object_path') or meta.get('object_path'),
                'metrics': {'source_metadata':meta}, 'state':'complete', 'approved':False}, asset['id'])
            folder = self.store.folder(v['id'])
            shutil.copy2(glb, folder / 'result.glb')
            fields = {'glb':str(folder / 'result.glb')}
            if ref:
                original = folder / ('original'+ref.suffix.lower())
                shutil.copy2(ref,original)
                fields['original'] = str(original)
                with Image.open(ref) as image:
                    ImageOps.exif_transpose(image).convert('RGBA').save(folder / 'reference.png')
                fields['reference'] = str(folder / 'reference.png')
            v = self.store.update(v['id'], **fields)
            with self.store.connect() as c:
                c.execute('INSERT INTO imports VALUES (?,?)', (fingerprint, v['id']))
            return v, True

    def sync(self):
        # Import the latest result for each USD object, not every old experiment.
        latest = {}
        for root in self.import_roots[:1]:
            if not root.exists():
                continue
            for p in root.rglob('*.glb'):
                if 'assets' not in p.parts or not Path(str(p)+'.meta.json').is_file():
                    continue
                old = latest.get(p.parent)
                if old is None or p.stat().st_mtime_ns > old.stat().st_mtime_ns:
                    latest[p.parent] = p
        count, errors = 0, []
        for p in sorted(latest.values(), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            try:
                _, added = self.import_asset({'glb_path':str(p)})
                count += int(added)
            except (ValueError, OSError, KeyError) as exc:
                errors.append({'file':p.name,'error':str(exc)})
        return {'added':count, 'errors':errors, 'found_objects':len(latest)}

    def submit(self, asset_id, source_id, action, data):
        with self.lock:
            asset = self.store.get(asset_id, 'asset')
            source = self.store.get(source_id, 'version') if source_id else {}
            if source and source['parent'] != asset_id:
                raise ValueError('다른 에셋의 버전입니다.')
            if action not in ('reference','prepare','mesh'):
                raise ValueError('지원하지 않는 작업입니다.')
            model = data.get('model', 'ernie')
            if model not in ('ernie','qwen_int8'):
                raise ValueError('지원하지 않는 이미지 모델입니다.')
            prompt = str(data.get('prompt',source.get('prompt',''))).strip()
            if len(prompt) > 4000 or (action == 'reference' and not prompt):
                raise ValueError('외형 설명을 1~4,000자로 입력하세요.')
            if action != 'reference' and not source.get('reference'):
                raise ValueError('먼저 참조 이미지를 선택하거나 업로드하세요.')
            if sum(j['state'] in ACTIVE for j in self.store.list('job')) >= 30:
                raise ValueError('대기열이 가득 찼습니다. 완료 후 다시 요청하세요.')
            seed = data.get('seed')
            seed = int.from_bytes(os.urandom(4),'little') % 1000000 if seed in (None,'') else int(seed)
            faces, texture = int(data.get('faces',300000)), int(data.get('texture',2048))
            if not 0 <= seed <= 2147470000 or faces not in (100000,300000,1000000) or texture not in (1024,2048):
                raise ValueError('생성 설정 범위를 확인하세요.')
            rig = data.get('rig_type',source.get('rig_type','')) or ''
            if rig not in ('','static_object','bird','biped','quadruped','insect'):
                raise ValueError('객체 유형을 확인하세요.')
            v = self.store.create('version', {'label':{'reference':'새 참조 이미지','prepare':'배경 제거 확인','mesh':'3D 재생성'}[action],
                'base_version':source_id,'prompt':prompt,'object_name':source.get('object_name') or asset.get('object_name') or asset['name'],
                'model':model if action == 'reference' else source.get('model','업로드'),
                'seed':seed,'rig_type':rig,'faces':faces,'texture':texture,'state':'queued',
                'source':'디자이너 작업','approved':False,'metrics':{}}, asset_id)
            folder = self.store.folder(v['id'])
            if action != 'reference':
                updates = {}
                for key in ('original','reference','cutout','prepared'):
                    if source.get(key):
                        dest = folder / (key+Path(source[key]).suffix)
                        shutil.copy2(self.media(source,key),dest)
                        updates[key] = str(dest)
                self.store.update(v['id'], **updates)
            j = self.store.create('job', {'version':v['id'],'action':action,'state':'queued',
                                  'stage':None,'message':'대기 중','completed':[]},asset_id)
            self.wake.set()
            return j

    def submit_candidates(self, data):
        with self.lock:
            count=data.get('count',4)
            if isinstance(count,bool) or not isinstance(count,int) or count not in (2,3,4):
                raise ValueError('후보 수는 2·3·4장 중 선택하세요.')
            if sum(j['state'] in ACTIVE for j in self.store.list('job')) + count > 30:
                raise ValueError(f'후보 {count}장을 추가할 대기 공간이 부족합니다.')
            seed=data.get('seed')
            mode=data.get('seed_mode','random' if seed in (None,'') else 'manual')
            if mode not in ('random','manual'):
                raise ValueError('시드 설정을 확인하세요.')
            if mode=='random':
                seed=int.from_bytes(os.urandom(4),'little')%2147470001
            else:
                if isinstance(seed,bool) or seed in (None,'') or not str(seed).isdigit():
                    raise ValueError('시드는 0 이상의 정수로 입력하세요.')
                seed=int(seed)
            if not 0<=seed<=2147470000:
                raise ValueError('시드 범위를 확인하세요.')
            jobs=[]
            for index in range(count):
                jobs.append(self.submit(data['asset_id'],data.get('version_id'),'reference',
                              {**data,'seed':(seed+index)%2147470001}))
            batch=self.store.create('batch',dict(count=count,pe_seed=seed,seed_mode=mode,
                seeds=[(seed+i)%2147470001 for i in range(count)],state='queued',attempt=0,
                jobs=[j['id'] for j in jobs]),data['asset_id'])
            for index,j in enumerate(jobs):
                self.store.update(j['version'],batch_id=batch['id'],candidate_index=index+1,
                                  candidate_count=count,label=f'이미지 후보 {index+1}/{count}')
                jobs[index]=self.store.update(j['id'],batch_id=batch['id'],candidate_index=index+1,
                                               candidate_count=count,batch_execution=True)
            return {'batch_id':batch['id'],'jobs':jobs,'count':count,'seeds':batch['seeds']}

    def cancel(self, jid):
        with self.lock:
            j = self.store.get(jid,'job')
            if j.get('batch_execution'):
                from candidate_jobs import control
                control(self,j['batch_id'],'cancel')
                return self.store.get(jid)
            if j['state'] not in ACTIVE:
                return j
            if j['state'] in ('queued','waiting'):
                self.store.update(j['version'],state='cancelled')
                return self.store.update(jid,state='cancelled',ended=time.time(),message='대기 취소됨')
            self.store.update(jid,state='cancelling',message='현재 작업을 중지하고 있습니다.')
            if self.current == jid and self.proc and self.proc.poll() is None:
                os.killpg(self.proc.pid,signal.SIGTERM)
            return self.store.get(jid)

    def available(self):
        try:
            p = subprocess.run(['nvidia-smi',f'--id={self.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                               capture_output=True,text=True,timeout=10)
            return int(p.stdout.strip()) >= 22000
        except (ValueError, OSError, subprocess.TimeoutExpired):
            return False

    def run_job(self, job):
        with self.lock:
            job=self.store.get(job['id'],'job')
        if job.get('batch_execution'):
            from candidate_jobs import run_batch
            return run_batch(self,job)
        jid, vid = job['id'], job['version']
        folder = self.store.folder(vid)
        self.current = jid
        with self.lock:
            if self.store.get(jid)['state'] not in ('queued','waiting'):
                return
            self.store.update(jid,state='running',started=time.time(),message='작업 준비 중')
            self.store.update(vid,state='running')
        try:
            stages = ['caption','reference'] if job['action'] == 'reference' else ['prepare'] + ([] if job['action']=='prepare' else ['mesh','export'])
            for stage in stages:
                if self.stop.is_set() or self.store.get(jid)['state'] == 'cancelling':
                    raise InterruptedError()
                v = self.store.get(vid)
                # Only reuse complete atomic outputs from this exact version/spec.
                output = {'caption':'caption.metrics.json','reference':'reference.png','prepare':'prepared.png','mesh':'mesh.npz','export':'result.glb'}[stage]
                marker = folder / f'{stage}.metrics.json'
                inherited = stage == 'prepare' and v.get('prepared')
                if not inherited and not ((folder/output).is_file() and marker.is_file()):
                    self.store.update(jid,stage=stage,message=STAGES[stage],stage_started=time.time())
                    spec = {**v,'folder':str(folder)}
                    (folder/'spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
                    env = {**os.environ,'CUDA_VISIBLE_DEVICES':self.gpu,'PYTHONUNBUFFERED':'1',
                           'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'8'}
                    with (folder/'worker.log').open('ab') as log:
                        with self.lock:
                            if self.store.get(jid)['state']=='cancelling':
                                raise InterruptedError()
                            self.proc = subprocess.Popen([self.python,str(HERE/'worker.py'),stage,str(folder/'spec.json')],
                                stdout=log,stderr=subprocess.STDOUT,env=env,cwd=REPO,start_new_session=True)
                        deadline = time.monotonic()+1800
                        cancel_started = None
                        while self.proc.poll() is None:
                            if self.stop.is_set() or self.store.get(jid)['state']=='cancelling':
                                if cancel_started is None:
                                    cancel_started = time.monotonic()
                                    os.killpg(self.proc.pid,signal.SIGTERM)
                                elif time.monotonic()-cancel_started > 5:
                                    os.killpg(self.proc.pid,signal.SIGKILL)
                            if time.monotonic() > deadline:
                                os.killpg(self.proc.pid,signal.SIGKILL)
                                self.proc.wait()
                                raise RuntimeError('단계 실행 제한 시간(30분)을 초과했습니다.')
                            time.sleep(.5)
                        if self.stop.is_set() or self.store.get(jid)['state']=='cancelling':
                            raise InterruptedError()
                        if self.proc.returncode:
                            tail = (folder/'worker.log').read_text(errors='replace')[-3000:]
                            message = '생성 단계가 실패했습니다. 같은 작업을 재시도하거나 참조 이미지를 바꿔 주세요.'
                            if 'out of memory' in tail.lower():
                                message = 'GPU 메모리가 부족합니다. 다른 작업이 끝난 뒤 이어서 재시도하세요.'
                            elif '배경 제거 후 객체가 남지' in tail:
                                message = '배경 제거 후 객체가 남지 않았습니다. 다른 참조 이미지를 선택하세요.'
                            raise RuntimeError(message)
                updates = {}
                if stage == 'export':
                    glb_chunks(folder / 'result.glb')
                for key, name in [('reference','reference.png'),('cutout','cutout.png'),('prepared','prepared.png'),('glb','result.glb')]:
                    if (folder/name).is_file():
                        updates[key] = str(folder/name)
                metrics = self.store.get(vid).get('metrics',{})
                if marker.is_file():
                    metrics[stage] = json.loads(marker.read_text())
                updates.update(metrics=metrics,warning=metrics.get('mesh',{}).get('warning',''))
                self.store.update(vid,**updates)
                completed = self.store.get(jid)['completed']
                if stage not in completed:
                    self.store.update(jid,completed=completed+[stage])
            with self.lock:
                if self.store.get(jid)['state']=='cancelling':
                    raise InterruptedError()
                self.store.update(jid,state='complete',ended=time.time(),message='완료')
                self.store.update(vid,state='complete')
        except InterruptedError:
            state = 'interrupted' if self.stop.is_set() else 'cancelled'
            self.store.update(jid,state=state,ended=time.time(),message='작업 중지됨 · 완료된 단계는 보존했습니다.')
            self.store.update(vid,state=state)
        except Exception as exc:
            self.store.update(jid,state='failed',ended=time.time(),message=str(exc))
            self.store.update(vid,state='failed')
        finally:
            self.proc = None
            self.current = None

    def dispatch(self):
        while not self.stop.is_set():
            jobs = [j for j in reversed(self.store.list('job')) if j['state'] in ('queued','waiting')]
            if jobs:
                lockpath = self.store.root.parent / f'generation-studio-gpu-{self.gpu}.lock'
                with lockpath.open('a') as gpu_lock:
                    try:
                        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        if self.available():
                            self.run_job(jobs[0])
                            continue
                    except BlockingIOError:
                        pass
                with self.lock:
                    if self.store.get(jobs[0]['id'])['state'] in ('queued','waiting'):
                        self.store.update(jobs[0]['id'],state='waiting',message='다른 작업이 GPU를 사용 중입니다. 여유 공간을 기다립니다.')
            self.wake.wait(3)
            self.wake.clear()

    def watch_imports(self):
        while not self.stop.wait(30):
            try:
                self.sync()
            except Exception as exc:
                print(f'USD result sync: {type(exc).__name__}',flush=True)


def create_app(studio):
    app = Flask(__name__,static_folder=str(HERE/'static'))
    app.config['MAX_CONTENT_LENGTH'] = 250*1024**2

    @app.before_request
    def guard():
        if request.method not in ('GET','HEAD','OPTIONS'):
            if request.headers.get('X-Requested-With') != 'GenerationStudio':
                abort(403,description='요청 출처를 확인할 수 없습니다.')
            origin = request.headers.get('Origin')
            if origin and urlsplit(origin).netloc != request.host:
                abort(403,description='다른 사이트의 요청은 허용하지 않습니다.')

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['X-Frame-Options']='DENY'
        response.headers['Cache-Control']='no-store' if request.path.startswith('/api/') else 'no-cache'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' blob:; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc,HTTPException):
            return jsonify(error=exc.description),exc.code
        if isinstance(exc,KeyError):
            return jsonify(error='작업 또는 파일을 찾을 수 없습니다.'),404
        if isinstance(exc,(ValueError,TypeError,UnidentifiedImageError,Image.DecompressionBombError)):
            return jsonify(error=str(exc)),400
        app.logger.exception('Studio request failed')
        return jsonify(error='처리 중 문제가 발생했습니다. 잠시 후 다시 시도하세요.'),500

    @app.get('/')
    def index():
        return send_file(HERE/'static/index.html')

    @app.get('/api/assets')
    def assets():
        vs = studio.store.list('version')
        data = []
        for a in studio.store.list('asset'):
            versions = [v for v in vs if v['parent']==a['id']]
            thumbnail = next((v for v in versions if v.get('reference') and v.get('glb')),None)
            thumbnail = thumbnail or next((v for v in versions if v.get('reference')),None)
            data.append({k:a.get(k) for k in ['id','name','project','source','created','favorite','notes']})
            data[-1].update(versions=len(versions),latest=versions[0]['id'] if versions else None,
                updated=versions[0]['created'] if versions else a['created'],
                approved=any(v.get('approved') for v in versions),
                thumbnail=f"/api/versions/{thumbnail['id']}/files/reference" if thumbnail else None)
        return jsonify(sorted(data,key=lambda a:a['updated'],reverse=True))

    @app.post('/api/assets')
    def new_asset():
        data = request.get_json()
        name = str(data.get('name','')).strip()[:120]
        if not name:
            raise ValueError('에셋 이름을 입력하세요.')
        return jsonify(studio.store.create('asset',{'name':name,'object_name':name,'project':'직접 추가',
                             'source':'직접 추가','favorite':False,'notes':''})),201

    @app.get('/api/assets/<aid>')
    def detail(aid):
        a = studio.store.get(aid,'asset')
        a.pop('source_key',None)
        return jsonify(asset=a,versions=[studio.version_json(v) for v in studio.store.list('version',aid)])

    @app.patch('/api/assets/<aid>')
    def edit_asset(aid):
        studio.store.get(aid,'asset')
        data = request.get_json()
        changes = {}
        if 'name' in data:
            changes['name'] = str(data['name']).strip()[:120]
            if not changes['name']:
                raise ValueError('이름을 입력하세요.')
        if 'notes' in data:
            changes['notes'] = str(data['notes'])[:4000]
        if 'favorite' in data:
            changes['favorite'] = bool(data['favorite'])
        studio.store.update(aid,**changes)
        return jsonify(ok=True)

    @app.post('/api/import')
    def register():
        v, added = studio.import_asset(request.get_json())
        return jsonify(asset_id=v['parent'],version_id=v['id'],added=added,
                       review_url=f"/?asset={v['parent']}&version={v['id']}")

    @app.post('/api/sync')
    def sync():
        return jsonify(studio.sync())

    @app.post('/api/assets/<aid>/upload')
    def upload(aid):
        a = studio.store.get(aid,'asset')
        file = request.files.get('file')
        if not file or not file.filename:
            raise ValueError('이미지 또는 GLB 파일을 선택하세요.')
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ('.png','.jpg','.jpeg','.webp','.glb'):
            raise ValueError('PNG, JPG, WebP, GLB 파일을 사용할 수 있습니다.')
        content = file.read()
        image = None
        if suffix != '.glb':
            if len(content)>25*1024**2:
                raise ValueError('참조 이미지는 25MB 이하로 올려 주세요.')
            image = Image.open(io.BytesIO(content))
            if image.width*image.height>25000000:
                raise ValueError('이미지 해상도는 2,500만 픽셀 이하로 올려 주세요.')
            image = ImageOps.exif_transpose(image).convert('RGBA')
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.glb',dir=studio.store.root) as temp:
                temp.write(content)
                temp.flush()
                glb_chunks(temp.name)
        source_id = request.form.get('base_version') or None
        source = studio.store.get(source_id,'version') if source_id else {}
        if source and source['parent']!=aid:
            raise ValueError('다른 에셋의 버전입니다.')
        v = studio.store.create('version',{'label':'GLB 업로드' if suffix=='.glb' else '참조 이미지 업로드',
              'base_version':source_id,'source':'업로드','object_name':source.get('object_name') or a.get('object_name') or a['name'],
              'prompt':request.form.get('prompt',source.get('prompt',''))[:4000],
              'model':'업로드','seed':None,'state':'complete','approved':False,'metrics':{}},aid)
        folder = studio.store.folder(v['id'])
        if image:
            (folder/('original'+suffix)).write_bytes(content)
            image.save(folder/'reference.png')
            studio.store.update(v['id'],reference=str(folder/'reference.png'),original=str(folder/('original'+suffix)))
        else:
            (folder/'result.glb').write_bytes(content)
            studio.store.update(v['id'],glb=str(folder/'result.glb'))
        return jsonify(studio.version_json(studio.store.get(v['id']))),201

    @app.patch('/api/versions/<vid>')
    def approve(vid):
        v = studio.store.get(vid,'version')
        if not v.get('glb'):
            raise ValueError('3D 결과가 있는 버전을 선택하세요.')
        studio.store.update(vid,approved=bool(request.get_json().get('approved')))
        return jsonify(ok=True)

    @app.get('/api/versions/<vid>/files/<key>')
    def files(vid,key):
        if key not in ('original','reference','cutout','prepared','glb'):
            abort(404)
        v = studio.store.get(vid,'version')
        return send_file(studio.media(v,key),conditional=True)

    @app.get('/api/versions/<vid>/download')
    def download(vid):
        v = studio.store.get(vid,'version')
        a = studio.store.get(v['parent'],'asset')
        height = request.args.get('height')
        height = float(height) if height else None
        pivot, yaw = request.args.get('pivot','keep'),int(request.args.get('yaw',0))
        if (height is not None and (not math.isfinite(height) or not .001<=height<=10000)) or pivot not in ('keep','center','floor') or yaw not in (0,90,180,270):
            raise ValueError('내보내기 설정 범위를 확인하세요.')
        data = export_glb(studio.media(v,'glb'),height,pivot,yaw)
        name = safe_name(request.args.get('name') or a['name'])+'_'+vid[:6]
        if request.args.get('bundle')=='1':
            output = io.BytesIO()
            with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr(name+'.glb',data)
                for key in ('original','reference','cutout'):
                    if v.get(key):
                        z.write(studio.media(v,key),key+Path(v[key]).suffix)
                manifest = studio.version_json(v)
                manifest.pop('media')
                manifest['export']={'height_m':height,'pivot':pivot,'yaw_degrees':yaw,'up_axis':'+Y',
                                    'scale_note':'원본 스케일 유지' if height is None else '지정한 높이를 미터로 적용'}
                z.writestr('generation.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            output.seek(0)
            return send_file(output,mimetype='application/zip',as_attachment=True,download_name=name+'.zip')
        return send_file(io.BytesIO(data),mimetype='model/gltf-binary',as_attachment=True,download_name=name+'.glb')

    @app.get('/api/jobs')
    def jobs():
        return jsonify(studio.store.list('job')[:200])

    @app.post('/api/jobs')
    def submit():
        data = request.get_json()
        return jsonify(studio.submit(data['asset_id'],data.get('version_id'),data['action'],data)),202

    @app.post('/api/candidates')
    def candidates():
        return jsonify(studio.submit_candidates(request.get_json())),202

    @app.get('/api/candidates/<bid>')
    def candidate_status(bid):
        return jsonify({**studio.store.get(bid,'batch'),'server_time':time.time()})

    @app.post('/api/candidates/<bid>/<action>')
    def candidate_control(bid,action):
        if action not in ('cancel','retry'):
            abort(404)
        from candidate_jobs import control
        return jsonify(control(studio,bid,action))

    @app.post('/api/jobs/<jid>/cancel')
    def cancel(jid):
        return jsonify(studio.cancel(jid))

    @app.post('/api/jobs/<jid>/retry')
    def retry(jid):
        with studio.lock:
            j = studio.store.get(jid,'job')
            if j.get('batch_execution'):
                from candidate_jobs import control
                return jsonify(control(studio,j['batch_id'],'retry'))
            if j['state'] not in ('failed','cancelled','interrupted'):
                raise ValueError('중단되거나 실패한 작업만 이어서 실행할 수 있습니다.')
            studio.store.update(jid,state='queued',message='완료된 단계부터 이어서 실행합니다.',ended=None)
            studio.store.update(j['version'],state='queued')
            studio.wake.set()
        return jsonify(ok=True)

    @app.get('/api/jobs/<jid>/report')
    def report(jid):
        j = studio.store.get(jid,'job')
        v = studio.store.get(j['version'])
        payload={'job':j,'version':studio.version_json(v)}
        if j.get('batch_execution'):
            payload['batch']=studio.store.get(j['batch_id'],'batch')
        return send_file(io.BytesIO(json.dumps(payload,ensure_ascii=False,indent=2).encode()),
                         mimetype='application/json',as_attachment=True,download_name=f'작업보고_{jid[:8]}.json')

    @app.get('/api/health')
    def health():
        return jsonify(ok=True,gpu=studio.gpu,queue=sum(j['state'] in ACTIVE for j in studio.store.list('job')),
                       models=['ernie','qwen_int8'],defaults={'prompt_enhancer':True,'rig_pose':True,'view':'three_quarter'},
                       disk_free_gib=round(shutil.disk_usage(studio.store.root).free/1024**3,1))
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host',default='10.254.62.200')
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--gpu',default=os.environ.get('STUDIO_GPU','1'))
    p.add_argument('--data',default=str(PROJECT/'.runtime/generation-studio'))
    p.add_argument('--import-root',action='append')
    args = p.parse_args()
    studio = Studio(args.data,args.gpu,import_roots=args.import_root)
    # Do not start a second dispatcher on the same database.
    service_lock = (studio.store.root/'service.lock').open('a')
    fcntl.flock(service_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    studio.store.recover()
    studio.sync()
    dispatcher = threading.Thread(target=studio.dispatch,daemon=True)
    dispatcher.start()
    threading.Thread(target=studio.watch_imports,daemon=True).start()
    def shutdown(*_):
        studio.stop.set()
        studio.wake.set()
        if studio.proc and studio.proc.poll() is None:
            os.killpg(studio.proc.pid,signal.SIGTERM)
        dispatcher.join(timeout=8)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    app = create_app(studio)
    from werkzeug.serving import make_server
    server = make_server(args.host,args.port,app,threaded=True)
    print(f'Generation Studio: http://{args.host}:{args.port}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
