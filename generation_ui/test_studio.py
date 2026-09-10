"""CPU checks for persistence, immutable intake, exports and queue boundaries."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import Studio, create_app, export_glb, glb_chunks
from store import Store
from caption import clean_caption


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/'incoming'
        self.source.mkdir()
        self.studio=Studio(self.root/'data',import_roots=[self.source])
        self.app=create_app(self.studio)
        self.client=self.app.test_client()
        self.headers={'X-Requested-With':'GenerationStudio'}
        mesh=trimesh.creation.box(extents=[1,2,3])
        mesh.apply_translation([4,5,6])
        self.glb=self.source/'box.glb'
        mesh.export(self.glb)
        self.ref=self.source/'ref.png'
        Image.new('RGB',(32,32),'white').save(self.ref)
        self.v,_=self.studio.import_asset({'glb_path':str(self.glb),'reference_path':str(self.ref),'name':'검증 상자'})
        self.a=self.v['parent']

    def tearDown(self):
        self.tmp.cleanup()

    def post(self,url,data):
        return self.client.post(url,json=data,headers=self.headers)

    def test_import_is_idempotent_and_does_not_overwrite_source(self):
        before=self.glb.read_bytes()
        again,added=self.studio.import_asset({'glb_path':str(self.glb)})
        self.assertFalse(added)
        self.assertEqual(again['id'],self.v['id'])
        self.assertEqual(before,Path(self.v['glb']).read_bytes())
        self.assertEqual(before,self.glb.read_bytes())

    def test_csrf_and_path_boundaries(self):
        self.assertEqual(self.client.post('/api/assets',json={'name':'x'}).status_code,403)
        self.assertEqual(self.client.post('/api/assets',json={'name':'x'},headers={**self.headers,'Origin':'https://elsewhere.test'}).status_code,403)
        outside=self.root/'outside.glb'
        outside.write_bytes(self.glb.read_bytes())
        self.assertEqual(self.post('/api/import',{'glb_path':str(outside)}).status_code,400)
        link=self.source/'escape.glb'
        link.symlink_to(outside)
        self.assertEqual(self.post('/api/import',{'glb_path':str(link)}).status_code,400)
        self.assertEqual(self.client.get(f"/api/versions/{self.v['id']}/files/worker.log").status_code,404)

    def test_versions_preserve_reference_and_asset_ownership(self):
        response=self.post('/api/jobs',{'asset_id':self.a,'version_id':self.v['id'],'action':'mesh'})
        self.assertEqual(response.status_code,202)
        job=response.json
        new=self.studio.store.get(job['version'])
        self.assertNotEqual(new['id'],self.v['id'])
        self.assertEqual(new['base_version'],self.v['id'])
        self.assertEqual(Path(new['reference']).read_bytes(),Path(self.v['reference']).read_bytes())
        self.assertEqual(Path(new['original']).read_bytes(),self.ref.read_bytes())
        self.assertNotIn('glb',new)
        other=self.post('/api/assets',{'name':'other'}).json
        bad=self.post('/api/jobs',{'asset_id':other['id'],'version_id':self.v['id'],'action':'mesh'})
        self.assertEqual(bad.status_code,400)

    def test_cancel_retry_and_restart_persist(self):
        j=self.studio.submit(self.a,self.v['id'],'mesh',{})
        self.studio.cancel(j['id'])
        self.assertEqual(self.studio.store.get(j['id'])['state'],'cancelled')
        self.assertEqual(self.post(f"/api/jobs/{j['id']}/retry",{}).status_code,200)
        self.studio.store.update(j['id'],state='running',completed=['prepare'])
        restarted=Store(self.root/'data')
        restarted.recover()
        result=restarted.get(j['id'])
        self.assertEqual(result['state'],'interrupted')
        self.assertEqual(result['completed'],['prepare'])

    def test_download_transform_preserves_original(self):
        before=self.glb.read_bytes()
        transformed=export_glb(self.glb,height=4,pivot='floor',yaw=90)
        mesh=trimesh.load(io.BytesIO(transformed),file_type='glb',force='scene',process=False)
        self.assertAlmostEqual(mesh.bounds[0,1],0,places=5)
        self.assertAlmostEqual(mesh.extents[1],4,places=5)
        self.assertAlmostEqual(mesh.bounds[:,0].mean(),0,places=5)
        self.assertEqual(before,self.glb.read_bytes())
        self.assertEqual(before,export_glb(self.glb))

    def test_bundle_has_exact_reference_and_generation_record(self):
        r=self.client.get(f"/api/versions/{self.v['id']}/download?bundle=1&name=상자")
        self.assertEqual(r.status_code,200)
        with zipfile.ZipFile(io.BytesIO(r.data)) as z:
            self.assertEqual(z.read('reference.png'),Path(self.v['reference']).read_bytes())
            self.assertEqual(z.read('original.png'),self.ref.read_bytes())
            self.assertEqual(json.loads(z.read('generation.json'))['id'],self.v['id'])
            self.assertTrue(any(x.endswith('.glb') for x in z.namelist()))

    def test_upload_validation_and_new_version(self):
        bad=self.client.post(f'/api/assets/{self.a}/upload',data={'file':(io.BytesIO(b'not image'),'bad.png')},headers=self.headers)
        self.assertEqual(bad.status_code,400)
        good=self.client.post(f'/api/assets/{self.a}/upload',data={'file':(io.BytesIO(self.ref.read_bytes()),'ref.png'),'base_version':self.v['id']},headers=self.headers)
        self.assertEqual(good.status_code,201)
        self.assertIn('reference',good.json['media'])
        self.assertNotIn('glb',good.json['media'])
        self.assertEqual(len(self.studio.store.list('version',self.a)),2)

    def test_export_rejects_nan_and_external_resources(self):
        self.assertEqual(self.client.get(f"/api/versions/{self.v['id']}/download?height=nan").status_code,400)
        doc,chunks=glb_chunks(self.glb)
        doc['images']=[{'uri':'https://elsewhere.test/image.png'}]
        data=json.dumps(doc).encode();data+=b' '*(-len(data)%4)
        chunks[0]=(0x4E4F534A,data)
        body=b''.join(struct.pack('<II',len(d),k)+d for k,d in chunks)
        bad=self.source/'external.glb';bad.write_bytes(b'glTF'+struct.pack('<II',2,len(body)+12)+body)
        with self.assertRaises(ValueError):glb_chunks(bad)

    def test_worker_retry_reuses_completed_stages(self):
        # Simulate a worker that fails its first export. No GPU/model required.
        worker=self.root/'fake_worker.py'
        worker.write_text('''#!/usr/bin/env python3
import json,sys,shutil
from pathlib import Path
stage=sys.argv[2];spec=json.loads(Path(sys.argv[3]).read_text());p=Path(spec['folder'])
with (p/'calls.txt').open('a') as log: log.write(stage+'\\n')
if stage=='prepare':
 for name in ['prepared.png','cutout.png']:shutil.copy2(spec['reference'],p/name)
elif stage=='mesh': (p/'mesh.npz').write_bytes(b'private-test-checkpoint')
elif not (p/'failed-once').exists():
 (p/'failed-once').touch();sys.exit(1)
else:shutil.copy2(spec['test_glb'],p/'result.glb')
(p/(stage+'.metrics.json')).write_text(json.dumps({'stage':stage,'seconds':0.1}))
''')
        worker.chmod(0o755)
        self.studio.python=str(worker)
        j=self.studio.submit(self.a,self.v['id'],'mesh',{})
        self.studio.store.update(j['version'],test_glb=str(self.glb))
        self.studio.run_job(j)
        first=self.studio.store.get(j['id'])
        self.assertEqual(first['state'],'failed')
        self.assertEqual(first['completed'],['prepare','mesh'])
        self.post(f"/api/jobs/{j['id']}/retry",{})
        self.studio.run_job(self.studio.store.get(j['id']))
        self.assertEqual(self.studio.store.get(j['id'])['state'],'complete')
        self.assertEqual((self.studio.store.folder(j['version'])/'calls.txt').read_text().splitlines(),
                         ['prepare','mesh','export','export'])

    def test_running_cancel_terminates_only_own_worker(self):
        worker=self.root/'slow_worker.py'
        worker.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n')
        worker.chmod(0o755)
        self.studio.python=str(worker)
        job=self.studio.submit(self.a,self.v['id'],'reference',{'prompt':'A box'})
        thread=threading.Thread(target=self.studio.run_job,args=(job,))
        thread.start()
        deadline=time.monotonic()+5
        while self.studio.proc is None and time.monotonic()<deadline:
            time.sleep(.02)
        self.assertIsNotNone(self.studio.proc)
        child=self.studio.proc
        self.studio.cancel(job['id'])
        thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.studio.store.get(job['id'])['state'],'cancelled')
        self.assertTrue(Path(self.v['glb']).is_file())

    def test_four_candidates_share_config_but_distinct_seeds(self):
        data={'asset_id':self.a,'version_id':self.v['id'],'prompt':'A red box','model':'ernie','seed':101}
        r=self.post('/api/candidates',data)
        self.assertEqual(r.status_code,202)
        self.assertEqual(len(r.json['jobs']),4)
        versions=[self.studio.store.get(j['version']) for j in r.json['jobs']]
        self.assertEqual([v['seed'] for v in versions],[101,102,103,104])
        self.assertEqual({v['batch_id'] for v in versions},{r.json['batch_id']})
        self.assertEqual({v['prompt'] for v in versions},{'A red box'})
        self.assertEqual({v['base_version'] for v in versions},{self.v['id']})
        self.assertTrue(all(not v.get('glb') for v in versions))
        bad=self.post('/api/candidates',{**data,'model':'unknown'})
        self.assertEqual(bad.status_code,400)
        self.assertEqual(len(self.studio.store.list('job')),4)

    def test_reference_prompt_trace_follows_image_not_mesh_seed(self):
        ref=self.studio.store.create('version',{'model':'ernie','seed':101,'prompt':'raw scene',
            'reference':self.v['reference'],'metrics':{'reference':{'raw_prompt':'raw scene','cleaned_prompt':'A red box',
            'prompt_used':'A red box, neutral background','revised_prompt':'中文原文','prompt_enhancer':True}}},self.a)
        mesh=self.studio.submit(self.a,ref['id'],'mesh',{'seed':999,'prompt':'new mesh note'})
        trace=self.studio.version_json(self.studio.store.get(mesh['version']))['prompt_trace']
        self.assertEqual(trace['seed'],101)
        self.assertEqual(trace['enhanced_prompt'],'中文原文')
        self.assertEqual(trace['raw_prompt'],'raw scene')
        pending=self.studio.submit(self.a,ref['id'],'reference',{'prompt':'another color'})
        trace=self.studio.version_json(self.studio.store.get(pending['version']))['prompt_trace']
        self.assertFalse(trace.get('enhanced_prompt'))
        self.assertEqual(trace['raw_prompt'],'another color')

    def test_caption_uses_main_rules_cpu_and_reuses_seed_independent_cache(self):
        spec={'object_name':'car','prompt':'a red car on a ferry','folder':str(self.studio.store.folder(self.v['id']))}
        reply=json.dumps({'message':{'content':json.dumps({'studio/object':'A red car'})}}).encode()
        with patch('caption.urlopen',return_value=io.BytesIO(reply)) as network:
            first=clean_caption(spec)
            payload=json.loads(network.call_args.args[0].data)
            self.assertEqual(payload['options']['num_gpu'],0)
            self.assertIn('3D model caption specialist',payload['messages'][0]['content'])
            second=clean_caption({**spec,'seed':202})
            self.assertEqual(network.call_count,1)
            self.assertFalse(first['cached'])
            self.assertTrue(second['cached'])
            self.assertEqual(second['cleaned_prompt'],'A red car')
        with patch('caption.urlopen',return_value=io.BytesIO(b'{"message":{"content":""}}')):
            with self.assertRaises(ValueError):
                clean_caption({**spec,'prompt':'another scene'})


if __name__=='__main__':
    unittest.main()
