"""Use the main caption rules before the production T2I template.

Local Ollama runs this short step on CPU so it cannot occupy the T2I GPU.
The same caption is cached across a four-seed comparison. No raw fallback.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

RULES = Path(__file__).resolve().parents[1]/'previz_pipeline/step2_filter_prompt.txt'


def clean_caption(spec):
    model = os.environ.get('STUDIO_CAPTION_MODEL','gpt-oss:20b')
    rules = RULES.read_text()
    source = {'target':'object','object_path':'studio/object','name':spec['object_name'],
              'category':spec.get('category',''),'description':spec['prompt']}
    identity = json.dumps({'source':source,'rules':rules,'model':model,'revision':2},sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    cache = Path(spec['folder']).parents[1]/'caption_cache'
    cache.mkdir(exist_ok=True)
    path = cache/(digest+'.json')
    if path.is_file():
        return {**json.loads(path.read_text()),'cached':True,'seconds':0}
    payload = {'model':model,'messages':[{'role':'system','content':rules},
               {'role':'user','content':'Return the JSON caption for this single input.\n'+json.dumps([source],ensure_ascii=False)}],
               'stream':False,'think':'low','keep_alive':0,
               'format':{'type':'object','properties':{'studio/object':{'type':'string'}},
                         'required':['studio/object'],'additionalProperties':False},
               'options':{'num_gpu':0,'num_thread':8,'num_ctx':8192,'num_predict':2048,'temperature':0.2,'seed':42}}
    start = time.monotonic()
    url = os.environ.get('STUDIO_CAPTION_URL','http://127.0.0.1:11434').rstrip('/')+'/api/chat'
    with urlopen(Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=300) as response:
        result = json.load(response)
    output=result.get('message',{}).get('content','')
    if not output.strip():
        raise ValueError('외형 정제 모델이 답변을 완료하지 못했습니다. 작업을 재시도하세요.')
    caption = json.loads(output).get('studio/object')
    if not isinstance(caption,str) or not caption.strip() or len(caption)>4000:
        raise ValueError('외형 설명 정제 결과가 유효하지 않습니다. 작업을 재시도하세요.')
    record = {'stage':'caption','model':model,'device':'CPU','raw_prompt':spec['prompt'],
              'cleaned_prompt':caption.strip(),'rules_sha256':hashlib.sha256(rules.encode()).hexdigest(),
              'cache_key':digest,'cached':False,'seconds':round(time.monotonic()-start,3)}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,ensure_ascii=False,indent=2));temp.replace(path)
    return record
