"""Candidate batching contracts, shared PE, partial recovery and memory fallback."""
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch
import unittest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_studio import StudioTests
from candidate_worker import run, atomic_json


class CandidateTests(StudioTests):
    def request_batch(self, **extra):
        return self.post('/api/candidates',dict(asset_id=self.a,version_id=self.v['id'],
            prompt='A red car',model='ernie',seed=101,count=2,**extra))

    def test_counts_and_seed_modes_reject_invalid_without_orphans(self):
        for count in (2,3,4):
            r=self.post('/api/candidates',dict(asset_id=self.a,prompt='A box',count=count,seed=2147470000))
            self.assertEqual(r.status_code,202)
            self.assertEqual(r.json['seeds'],[(2147470000+i)%2147470001 for i in range(count)])
            self.assertEqual(len(r.json['jobs']),count)
        before=len(self.studio.store.list('job'))
        for count in (1,5,2.5,True,'3',None):
            self.assertEqual(self.post('/api/candidates',dict(asset_id=self.a,prompt='A box',count=count)).status_code,400)
        for seed in ('',-1,1.5,True,2147470001,'NaN'):
            self.assertEqual(self.post('/api/candidates',dict(asset_id=self.a,prompt='A box',count=2,seed_mode='manual',seed=seed)).status_code,400)
        self.assertEqual(len(self.studio.store.list('job')),before)
        import os
        original_random=os.urandom
        starts=iter([150,250])
        with patch('app.os.urandom',side_effect=lambda n:next(starts).to_bytes(4,'little') if n==4 else original_random(n)):
            results=[self.post('/api/candidates',dict(asset_id=self.a,prompt='A box',count=2,seed_mode='random',seed=99)).json for _ in range(2)]
        self.assertEqual([r['seeds'] for r in results],[[150,151],[250,251]])

    def test_batch_cancel_retry_preserves_complete_candidate(self):
        r=self.request_batch().json; first,second=r['jobs']
        self.studio.store.update(first['id'],state='complete')
        self.studio.store.update(first['version'],state='complete')
        self.assertEqual(self.post(f"/api/candidates/{r['batch_id']}/cancel",{}).status_code,200)
        self.assertEqual(self.studio.store.get(first['id'])['state'],'complete')
        self.assertEqual(self.studio.store.get(second['id'])['state'],'cancelled')
        self.assertEqual(self.post(f"/api/jobs/{second['id']}/retry",{}).status_code,200)
        self.assertEqual(self.studio.store.get(first['id'])['state'],'complete')
        self.assertEqual(self.studio.store.get(second['id'])['state'],'queued')
        self.assertEqual(self.post(f"/api/candidates/{r['batch_id']}/retry",{}).status_code,400)

    def test_batch_recovery_freezes_progress_and_pending_prompt_trace(self):
        from store import Store
        r=self.request_batch().json; bid=r['batch_id']
        now=time.time()
        self.studio.store.update(bid,state='running',progress={'updated':now,'phases':[
            {'key':'generate','state':'running','started':now-3}]},shared={
            'caption':{'cleaned_prompt':'A car'},'prompt_used':'A car, three-quarter view',
            'revised_prompt':'共用原文','pe_seed':101})
        first=r['jobs'][0]
        self.studio.store.update(first['id'],state='running')
        trace=self.studio.version_json(self.studio.store.get(first['version']))['prompt_trace']
        self.assertEqual(trace['enhanced_prompt'],'共用原文')
        self.assertEqual(trace['pe_seed'],101)
        recovered=Store(self.root/'data'); recovered.recover()
        result=recovered.get(bid)
        self.assertEqual(result['state'],'interrupted')
        self.assertEqual(result['progress']['phases'][0]['seconds'],3)
        self.assertEqual(recovered.get(first['id'])['state'],'interrupted')
        self.assertEqual(result['shared']['revised_prompt'],'共用原文')

    def spec(self,count=4):
        r=self.post('/api/candidates',dict(asset_id=self.a,prompt='A car',seed=101,count=count)).json
        b=self.studio.store.get(r['batch_id'])
        folder=self.studio.store.folder(b['id'])
        candidates=[{**self.studio.store.get(j['version']),'folder':str(self.studio.store.folder(j['version']))} for j in r['jobs']]
        atomic_json(folder/'shared.json',{'caption':{'cleaned_prompt':'A car','cached':False}})
        return dict(folder=str(folder),batch_id=b['id'],count=count,pe_seed=101,candidates=candidates,profile_path='/missing')

    def test_shared_prompt_oom_fallback_and_retry(self):
        import torch
        calls=[]; enhanced=[]
        class Fake:
            config={'steps':8}
            def __init__(self,name):calls.append('load')
            def sync(self):pass
            def enhance(self,prompt,seed):enhanced.append(seed);return '共同原文'
            def encode(self,prompt):calls.append(('encode',prompt))
            def clear_oom(self):calls.append('clear')
            def peaks(self):return {}
            def generate(self,seeds,progress):
                calls.append(tuple(seeds))
                if len(seeds)>2:raise torch.OutOfMemoryError('test')
                progress(8,8)
                return [Image.new('RGB',(16,16),'white') for _ in seeds]
        spec=self.spec()
        with patch('candidate_worker.choose_chunk',return_value=4):run(spec,Fake)
        self.assertEqual(enhanced,[101])
        self.assertEqual([x for x in calls if isinstance(x,tuple) and isinstance(x[0],int)],[(101,102,103,104),(101,102),(103,104)])
        self.assertEqual(calls.count('load'),1)
        before=(Path(spec['candidates'][0]['folder'])/'reference.png').read_bytes()
        (Path(spec['candidates'][-1]['folder'])/'reference.metrics.json').unlink()
        with patch('candidate_worker.choose_chunk',return_value=4):run({**spec,'attempt':2},Fake)
        self.assertEqual(enhanced,[101]);self.assertEqual(calls[-1],(104,))
        self.assertEqual(before,(Path(spec['candidates'][0]['folder'])/'reference.png').read_bytes())
        for c in spec['candidates']:
            metric=json.loads((Path(c['folder'])/'reference.metrics.json').read_text())
            self.assertEqual(metric['revised_prompt'],'共同原文');self.assertEqual(metric['seed'],c['seed'])
        progress=json.loads((Path(spec['folder'])/'progress.json').read_text())
        self.assertEqual(progress['state'],'complete')
        self.assertTrue(all('seconds' in p for p in progress['phases']))
        self.assertTrue(next(p for p in progress['phases'] if p['key']=='enhance')['cached'])

    def test_qwen_no_enhancer_and_real_failures_not_retried(self):
        spec=self.spec(2)
        for c in spec['candidates']:c['model']='qwen_int8'
        class Fake:
            config={}
            def __init__(self,name):pass
            def sync(self):pass
            def encode(self,p):pass
            def generate(self,seeds,progress):raise ValueError('ordinary error')
        with self.assertRaisesRegex(ValueError,'ordinary error'):
            run(spec,Fake)
        progress=json.loads((Path(spec['folder'])/'progress.json').read_text())
        self.assertTrue(next(p for p in progress['phases'] if p['key']=='enhance')['skipped'])
        self.assertEqual(len([p for p in progress['phases'] if p['key']=='generate']),1)
        self.assertEqual(progress['state'],'failed')

    def test_batch_process_live_partial_and_cancel(self):
        worker=self.root/'fake_batch_worker.py'
        worker.write_text('''#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
s=json.loads(Path(sys.argv[2]).read_text());folder=Path(s['folder'])
c=s['candidates'][0];p=Path(c['folder'])
from PIL import Image
Image.new('RGB',(16,16),'white').save(p/'reference.png')
(p/'caption.metrics.json').write_text('{}')
(p/'reference.metrics.json').write_text(json.dumps({'seed':c['seed'],'revised_prompt':'same'}))
(folder/'progress.json').write_text(json.dumps({'phases':[],'message':'partial','completed':1}))
time.sleep(30)
''')
        worker.chmod(0o755);self.studio.python=str(worker)
        r=self.request_batch().json
        thread=threading.Thread(target=self.studio.run_job,args=(r['jobs'][0],));thread.start()
        deadline=time.monotonic()+8
        while self.studio.store.get(r['jobs'][0]['id'])['state']!='complete' and time.monotonic()<deadline:time.sleep(.05)
        self.assertEqual(self.studio.store.get(r['jobs'][0]['id'])['state'],'complete')
        child=self.studio.proc
        self.post(f"/api/candidates/{r['batch_id']}/cancel",{})
        thread.join(8);self.assertFalse(thread.is_alive());self.assertIsNotNone(child.poll())
        self.assertEqual(self.studio.store.get(r['jobs'][0]['id'])['state'],'complete')
        self.assertEqual(self.studio.store.get(r['jobs'][1]['id'])['state'],'cancelled')
        self.assertEqual(self.studio.store.get(r['batch_id'])['state'],'cancelled')
        v=self.studio.version_json(self.studio.store.get(r['jobs'][0]['version']))
        self.assertIn('reference',v['media']);self.assertEqual(v['prompt_trace']['enhanced_prompt'],'same')
        mesh=self.studio.submit(self.a,v['id'],'mesh',{})
        self.assertEqual(self.studio.version_json(self.studio.store.get(mesh['version']))['prompt_trace']['enhanced_prompt'],'same')


if __name__=='__main__':unittest.main()
