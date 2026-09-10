import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';

const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const date = value => new Date(value*1000).toLocaleString('ko-KR',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const stateNames={queued:'대기',waiting:'GPU 대기',running:'진행 중',cancelling:'중지 중',complete:'완료',failed:'실패',cancelled:'중지됨',interrupted:'중단됨'};
let toastTimer;
function toast(message,error=false){$('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6500);}
async function api(path,options={}){
  const headers={'X-Requested-With':'GenerationStudio',...options.headers};
  if(options.body && !(options.body instanceof FormData)){headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
  const response=await fetch(path,{...options,headers});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'요청을 처리할 수 없습니다.');
  return data;
}
function act(id,fn){$(id).addEventListener('click',async()=>{const b=$(id);b.disabled=true;try{await fn();}catch(e){toast(e.message,true);}finally{b.disabled=false;updateControls();}});}

class AssetViewer{
  constructor(container,onStats=()=>{}){
    this.container=container;this.onStats=onStats;this.ticket=0;this.model=null;this.mode='material';this.url=null;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#edf0e9');
    this.camera=new THREE.PerspectiveCamera(34,1,.01,1000);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.2;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute('aria-label','3D 에셋: 마우스로 회전·확대');
    this.renderer.domElement.tabIndex=0;
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    const pmrem=new THREE.PMREMGenerator(this.renderer);const room=new RoomEnvironment();
    this.environment=pmrem.fromScene(room,.04);this.scene.environment=this.environment.texture;room.dispose();pmrem.dispose();
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x8a977f,2));
    const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(3,5,4);this.scene.add(light);
    this.grid=new THREE.GridHelper(5,20,0xb9c5ae,0xd5ddcd);this.scene.add(this.grid);
    this.clay=new THREE.MeshStandardMaterial({color:0xd3d9cc,roughness:.8,metalness:0,side:THREE.DoubleSide});
    this.wire=new THREE.MeshBasicMaterial({color:0x587050,wireframe:true});
    this.resize=new ResizeObserver(()=>this.fitCanvas());this.resize.observe(container);
    this.view('three');
    this.renderer.setAnimationLoop(()=>{if(container.clientWidth && container.clientHeight && !document.hidden){this.controls.update();this.renderer.render(this.scene,this.camera);}});
  }
  fitCanvas(){const w=this.container.clientWidth,h=this.container.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
  disposeModel(model){if(!model)return;model.traverse(o=>{o.geometry?.dispose();const mats=o.userData.originalMaterials|| (o.material?[o.material].flat():[]);for(const m of mats){for(const value of Object.values(m))if(value?.isTexture)value.dispose();m.dispose();}});}
  clear(){this.ticket++;this.url=null;if(this.model){this.scene.remove(this.model);this.disposeModel(this.model);this.model=null;}this.onStats('실제 GLB를 표시합니다');}
  async load(url){
    if(this.url===url)return;
    this.clear();this.url=url;const ticket=this.ticket;
    if(!url)return;
    this.onStats('3D 파일을 불러오는 중…');
    try{
      const response=await fetch(url);if(!response.ok)throw new Error('3D 파일을 불러오지 못했습니다.');
      const buffer=await response.arrayBuffer();if(ticket!==this.ticket)return;
      const result=await new GLTFLoader().parseAsync(buffer,'');
      if(ticket!==this.ticket){this.disposeModel(result.scene);return;}
      const original=result.scene;let triangles=0;const textures=new Set();
      original.traverse(o=>{if(!o.isMesh)return;triangles+=(o.geometry.index?.count??o.geometry.attributes.position.count)/3;
        o.userData.originalMaterials=[o.material].flat();for(const m of o.userData.originalMaterials)for(const x of Object.values(m))if(x?.isTexture)textures.add(x.uuid);});
      const box=new THREE.Box3().setFromObject(original),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
      const length=Math.max(size.x,size.y,size.z);if(!Number.isFinite(length)||length<=0){this.disposeModel(original);throw new Error('표시할 형상이 없는 에셋입니다.');}
      const group=new THREE.Group();group.add(original);original.position.sub(center);group.scale.setScalar(2/length);this.scene.add(group);this.model=group;
      this.grid.position.y=-size.y/length-.005;this.setMode(this.mode);this.view('three');
      this.onStats(`${Math.round(triangles).toLocaleString()}면 · 텍스처 ${textures.size}개 · 원본 ${size.x.toFixed(2)} × ${size.y.toFixed(2)} × ${size.z.toFixed(2)}`);
    }catch(e){if(ticket===this.ticket){this.url=null;this.onStats(e.message);toast(e.message,true);}}
  }
  setMode(mode){this.mode=mode;this.model?.traverse(o=>{if(o.isMesh){o.material=mode==='clay'?this.clay:mode==='wire'?this.wire:(o.userData.originalMaterials.length===1?o.userData.originalMaterials[0]:o.userData.originalMaterials);}});}
  view(kind){const positions={three:[3,2,3],front:[0,0,4.6],side:[4.6,0,0],back:[0,0,-4.6],bottom:[0,-4.6,.001],top:[0,4.6,.001]};this.camera.position.set(...(positions[kind]||positions.three));this.camera.up.set(0,1,0);this.controls.target.set(0,0,0);this.controls.update();}
}

let viewer,compareViewer;
try{viewer=new AssetViewer($('viewer'),text=>$('mesh-stats').textContent=text);}catch(e){$('viewer-empty').innerHTML='<strong>3D 보기를 시작하지 못했습니다.</strong><p>브라우저의 하드웨어 가속을 확인하세요. GLB 다운로드는 사용할 수 있습니다.</p>';}
const state={assets:[],asset:null,versions:[],selected:null,jobs:[],filter:'all',query:'',imageMode:'reference',detailTicket:0,tab:'i23d',batch:null,batchInfo:null,batchClock:0};
const selected=()=>state.versions.find(v=>v.id===state.selected);
const number=v=>state.versions.length-state.versions.findIndex(x=>x.id===v.id);
function renderAssets(){
  const list=state.assets.filter(a=>(`${a.name} ${a.project}`.toLowerCase().includes(state.query.toLowerCase()))&&(state.filter!=='favorite'||a.favorite)&&(state.filter!=='approved'||a.approved));
  $('asset-count').textContent=`${state.assets.length}개 에셋`;
  $('asset-list').innerHTML=list.map(a=>`<button class="asset-card ${state.asset?.id===a.id?'selected':''}" data-asset="${a.id}">${a.thumbnail?`<img src="${a.thumbnail}" loading="lazy" alt="">`:'<span class="placeholder">3D</span>'}<span class="info"><strong>${escape(a.name)} ${a.favorite?'☆':''}</strong><small>${escape(a.project||a.source)}</small><small>${a.versions}개 버전 ${a.approved?'<span class="approved-dot">· 선택 완료</span>':''}</small></span></button>`).join('')||'<p class="empty">표시할 에셋이 없습니다.<br>USD 결과를 가져오거나 직접 추가하세요.</p>';
}
function updateControls(){const v=selected(),asset=!!state.asset;for(const id of ['favorite','rename','generate-reference','save-notes','upload-button'])$(id).disabled=!asset;
  $('generate-mesh').disabled=!v?.media.reference;$('prepare-button').disabled=!v?.media.reference;
  $('use-candidate').disabled=!v?.media.reference;
  $('approve').disabled=!v?.media.glb;$('download').disabled=!v?.media.glb;
  $('approve').classList.toggle('chosen',!!v?.approved);$('approve').textContent=v?.approved?'✓ 선택 완료':'✓ 이 버전 선택 완료';
  $('favorite').textContent=state.asset?.favorite?'★':'☆';$('favorite').setAttribute('aria-pressed',String(!!state.asset?.favorite));
}
function renderVersions(){
  $('version-count').textContent=state.versions.length;
  $('versions').innerHTML=state.versions.map(v=>`<button class="version-card ${v.id===state.selected?'selected':''} ${v.approved?'approved':''}" data-version="${v.id}"><span class="version-chip">${v.approved?'✓ 선택 완료':v.media.glb?'3D 결과':v.media.reference?'참조 이미지':stateNames[v.state]||'준비 중'}</span><div class="version-preview">${v.media.reference?`<img src="${v.media.reference}" loading="lazy" alt="">`:'<span class="placeholder">3D</span>'}</div><div class="version-info"><strong>v${number(v)} · ${escape(v.label)}</strong><small>${date(v.created)}</small></div></button>`).join('')||'<p class="empty">이미지를 올리거나 첫 후보를 생성하세요.</p>';
  const previous=$('compare-version').value;
  $('compare-version').innerHTML='<option value="">비교 안 함</option>'+state.versions.filter(v=>v.media.glb&&v.id!==state.selected).map(v=>`<option value="${v.id}">v${number(v)} · ${escape(v.label)}</option>`).join('');
  $('compare-version').value=previous;renderComparison();
  renderCandidates();
}
function candidateVersions(){return state.versions.filter(v=>v.batch_id||v.metrics?.reference||v.model==='업로드'&&v.media.reference);}
const candidateRender={options:null,html:null};
function candidateCount(){return Number($('candidate-count').value);}
function updateCandidateSettings(){
  const n=candidateCount(),manual=$('image-seed-mode').value==='manual';
  $('image-seed').hidden=!manual;
  $('generate-reference').textContent=`＋ 서로 다른 seed로 ${n}장 생성`;
  $('seed-hint').textContent=`${manual?'입력한':'매번 새로 뽑은'} 시작값 R에서 ${Array.from({length:n},(_,i)=>i?`R+${i}`:'R').join(', ')}을 사용합니다.`;
  try{localStorage.setItem('studio.candidateCount',String(n));}catch(e){}
  renderCandidates();
}
try{const saved=localStorage.getItem('studio.candidateCount');if(['2','3','4'].includes(saved))$('candidate-count').value=saved;}catch(e){}
$('candidate-count').addEventListener('change',updateCandidateSettings);
$('image-seed-mode').addEventListener('change',updateCandidateSettings);

function renderCandidates(){
  const all=candidateVersions(),groups=[...new Set(all.filter(v=>v.batch_id).map(v=>v.batch_id))];
  if(state.batch&&state.batch!=='legacy'&&!groups.includes(state.batch))state.batch=null;
  if(!state.batch&&groups.length)state.batch=groups[0];
  const options=groups.map((id,i)=>{const v=all.find(v=>v.batch_id===id);return `<option value="${id}">${date(v.created)} · ${v.model==='ernie'?'ERNIE':'Qwen'} · 후보 ${v.candidate_count||all.filter(x=>x.batch_id===id).length}장</option>`;}).join('')+'<option value="legacy">이전 단일 후보·업로드</option>';
  if(options!==candidateRender.options){$('candidate-batch').innerHTML=options;candidateRender.options=options;}
  $('candidate-batch').value=state.batch||'legacy';
  const candidates=state.batch&&state.batch!=='legacy'?all.filter(v=>v.batch_id===state.batch).sort((a,b)=>a.candidate_index-b.candidate_index):all.filter(v=>!v.batch_id).slice(0,4).reverse();
  const count=state.batch&&state.batch!=='legacy'?(candidates[0]?.candidate_count||candidates.length||candidateCount()):Math.max(candidates.length,candidateCount());
  $('candidate-title').textContent=`서로 다른 seed, 이미지 후보 ${count}장`;
  const slots=Array.from({length:count},(_,i)=>candidates[i]);
  const cards=slots.map((v,i)=>v?`<button class="candidate-card ${v.id===state.selected?'selected':''}" data-candidate="${v.id}" aria-pressed="${v.id===state.selected}"><div class="candidate-image">${v.media.reference?`<img src="${v.media.reference}" alt="후보 ${i+1}, seed ${escape(v.seed)}">`:`<div class="candidate-wait"><strong>${stateNames[v.state]||'대기'}</strong><span>${escape(state.jobs.find(j=>j.version===v.id)?.message||'생성을 준비합니다.')}</span></div>`}<span class="candidate-number">${String(i+1).padStart(2,'0')}</span><span class="candidate-model">${v.model==='ernie'?'ERNIE · 보강 ON':v.model==='qwen_int8'?'Qwen INT8':escape(v.model)}</span>${v.id===state.selected?'<span class="candidate-check">✓ 선택</span>':''}</div><div class="candidate-meta"><strong>seed ${escape(v.seed??'업로드')}</strong><span>${stateNames[v.state]||''}</span></div></button>`:`<div class="candidate-card candidate-placeholder"><span>${String(i+1).padStart(2,'0')}</span><p>이미지 후보 ${i+1}</p><small>오른쪽에서 후보 ${candidateCount()}장을 생성하세요.</small></div>`).join('');
  if(cards!==candidateRender.html){$('candidate-grid').innerHTML=cards;candidateRender.html=cards;}
  const v=selected();$('candidate-selection').textContent=v?.media.reference?`선택: v${number(v)} · seed ${v.prompt_trace?.seed??v.seed??'업로드'}`:'후보를 선택하면 생성 입력과 보강문을 확인할 수 있습니다.';
}
function renderPrompts(){
  const v=selected(),trace=v?.prompt_trace||{};
  $('prompt-original').value=trace.raw_prompt??v?.prompt??'';
  $('prompt-cleaned').value=trace.cleaned_prompt||'이 결과에는 외형 정제 기록이 없습니다.';
  $('prompt-template').value=trace.template_prompt||'';
  $('prompt-enhanced').value=trace.enhanced_prompt||'';
  $('prompt-provenance').textContent=v?`v${number(v)}의 참조 이미지 · ${trace.model||v.model||''} · seed ${trace.seed??v.seed??'기록 없음'}`:'';
  $('prompt-note').textContent=trace.enhanced_prompt?(trace.shared_prompt?`이 묶음은 보강문을 공유합니다 · 보강 seed ${trace.pe_seed}. 원래 언어 그대로 표시합니다.`:'중국어 등 원래 언어 그대로 표시합니다. 번역·요약하지 않았습니다.'):trace.enhancer===false?'이 이미지에는 별도 자동 보강문이 없습니다. 템플릿 입력을 확인하세요.':trace.note||'생성 완료 후 기록된 보강문을 표시합니다.';
  for(const b of document.querySelectorAll('[data-copy-prompt]'))b.disabled=!$(b.dataset.copyPrompt).value;
}
function setTab(tab,choose=true){
  state.tab=tab;
  if(tab==='t2i'&&choose&&!candidateVersions().some(v=>v.id===state.selected)){
    const first=candidateVersions().find(v=>v.id===selected()?.prompt_trace?.source_version)||candidateVersions()[0];if(first){state.selected=first.id;state.batch=first.batch_id||'legacy';renderVersions();renderSelected(true);}
  }
  for(const key of ['t2i','i23d']){const on=key===tab;$('tab-'+key).setAttribute('aria-selected',String(on));$('tab-'+key).tabIndex=on?0:-1;$('panel-'+key).hidden=!on;}
  if(tab==='t2i')renderCandidates();
  setURL();
}
function renderImage(){const v=selected();if(!v?.media[state.imageMode])state.imageMode='reference';const url=v?.media[state.imageMode];
  $('reference-image').hidden=!url;$('reference-empty').hidden=!!url;
  if(url&&$('reference-image').getAttribute('src')!==url)$('reference-image').src=url;
  for(const b of $('image-modes').querySelectorAll('button')){b.disabled=!v?.media[b.dataset.image];b.classList.toggle('active',b.dataset.image===state.imageMode);}
  $('reference-badge').textContent={reference:'원본',cutout:'배경 제거',prepared:'3D 입력'}[state.imageMode];
}
function renderSelected(resetForm=false){
  const v=selected();updateControls();renderImage();renderPrompts();renderCandidates();
  $('current-version-label').textContent=v?`v${number(v)}`:'';
  if(viewer){$('viewer-empty').hidden=!!v?.media.glb;viewer.load(v?.media.glb||null);}
  const problem=state.jobs.find(j=>j.version===v?.id&&['failed','interrupted','cancelled'].includes(j.state));
  const warning=v?.warning|| (problem?`${stateNames[problem.state]} · ${problem.message} 대기열에서 이어서 재시도할 수 있습니다.`:'');
  $('asset-warning').hidden=!warning;$('asset-warning').textContent=warning;
  if(resetForm){$('prompt').value=v?.prompt||'';$('rig-type').value=v?.rig_type||'';$('seed').value='';}
  $('record').textContent=v?JSON.stringify({버전:`v${number(v)}`,작업번호:v.id,원본버전:v.base_version,
    이미지모델:v.model,시드:v.seed,입력설명:v.prompt,USD객체:v.source_object,측정기록:v.metrics},null,2):'아직 생성 기록이 없습니다.';
  const job=state.jobs.find(j=>j.version===v?.id&&['queued','waiting','running','cancelling'].includes(j.state));
  $('generation-hint').textContent=job?`${stateNames[job.state]} · ${job.message}`:'결과는 새 버전으로 저장됩니다.';
}
async function selectAsset(id,version=null){
  const ticket=++state.detailTicket;const data=await api('/api/assets/'+id);if(ticket!==state.detailTicket)return;
  state.asset=data.asset;state.versions=data.versions;state.selected=version&&data.versions.some(v=>v.id===version)?version:(data.versions.find(v=>v.approved&&v.media.glb)||data.versions.find(v=>v.media.glb)||data.versions[0])?.id||null;
  $('asset-title').textContent=data.asset.name;$('asset-project').textContent=data.asset.project||data.asset.source;
  $('asset-source').textContent=data.asset.source==='USD 연계'?'USD → ASSET REVIEW':'ASSET REVIEW';
  $('notes').value=data.asset.notes||'';state.imageMode='reference';state.batch=data.versions.find(v=>v.id===state.selected)?.batch_id||(candidateVersions().some(v=>v.id===state.selected)?'legacy':null);$('compare-version').value='';
  renderAssets();renderVersions();renderSelected(true);setURL();
}
function setURL(){const u=new URL(location.href);if(state.asset)u.searchParams.set('asset',state.asset.id);if(state.selected)u.searchParams.set('version',state.selected);else u.searchParams.delete('version');u.searchParams.set('tab',state.tab);history.replaceState(null,'',u);}
function selectVersion(id){if(!state.versions.some(v=>v.id===id))return;state.selected=id;state.batch=selected()?.batch_id||(candidateVersions().some(v=>v.id===id)?'legacy':state.batch);state.imageMode='reference';renderVersions();renderSelected(true);setURL();refreshBatch();}
function renderComparison(){const v=state.versions.find(v=>v.id===$('compare-version').value);$('compare-surface').hidden=!v;
  document.querySelector('.viewer-group').classList.toggle('comparing',!!v);
  if(v){if(!compareViewer)try{compareViewer=new AssetViewer($('compare-viewer'));}catch(e){toast('비교용 3D 보기를 열 수 없습니다.',true);}
    $('compare-label').textContent=`v${number(v)} · ${v.label}`;compareViewer?.load(v.media.glb);
  }else compareViewer?.clear();
}
function renderJobs(){
  const grouped=state.jobs.filter((j,i,all)=>!j.batch_execution||all.findIndex(x=>x.batch_id===j.batch_id)===i);
  const active=state.jobs.filter(j=>['queued','waiting','running','cancelling'].includes(j.state));$('queue-count').textContent=active.length;
  $('job-list').innerHTML=grouped.map(j=>{const a=state.assets.find(a=>a.id===j.parent);const stages=j.action==='reference'?['caption','reference']:j.action==='prepare'?['prepare']:['prepare','mesh','export'];
    return `<article class="job-row"><div class="job-top"><strong>${escape(a?.name||'에셋')} · ${j.action==='reference'?(j.batch_execution?`이미지 후보 ${j.candidate_count}장`:'이미지'):j.action==='prepare'?'배경 제거':'3D 생성'}</strong><span class="job-status ${j.state}">${stateNames[j.state]}</span><time>${j.started?`${Math.floor(((j.ended||Date.now()/1000)-j.started)/60)}분 ${Math.floor(((j.ended||Date.now()/1000)-j.started)%60)}초`:date(j.created)}</time></div><p class="job-message">${escape(j.message)}${['queued','waiting'].includes(j.state)?` · 대기 순서 ${[...active].reverse().findIndex(x=>x.id===j.id)+1}`:''}</p><div class="job-steps">${stages.map(s=>`<span class="${j.completed.includes(s)?'done':j.stage===s&&j.state==='running'?'current':''}">${{caption:'외형 정제',reference:'참조 이미지',prepare:'배경 제거',mesh:'3D 형상',export:'텍스처·저장'}[s]}</span>`).join('')}</div><div class="job-actions"><a href="/api/jobs/${j.id}/report" download>작업 보고서</a><button data-job-view="${j.id}">결과 보기</button>${['failed','cancelled','interrupted'].includes(j.state)?`<button data-job-retry="${j.id}">${j.batch_execution?'묶음 이어서 재시도':'이어서 재시도'}</button>`:''}${['queued','waiting','running'].includes(j.state)?`<button data-job-cancel="${j.id}">${j.batch_execution?'묶음 중지':'중지'}</button>`:''}</div></article>`;}).join('')||'<p class="empty">아직 실행한 작업이 없습니다.</p>';
}
async function refreshBatch(){
  const bid=state.batch;
  if(!bid||!state.jobs.some(j=>j.batch_id===bid&&j.batch_execution)){state.batchInfo=null;renderBatchProgress();return;}
  try{const info=await api('/api/candidates/'+bid);if(state.batch!==bid)return;state.batchInfo=info;state.batchClock=performance.now();renderBatchProgress();}
  catch(e){if(state.batch===bid){state.batchInfo=null;renderBatchProgress();}}
}
let batchProgressHTML='';
function tickBatchClocks(){
  const b=state.batchInfo;if(!b||b.id!==state.batch)return;
  const now=(b.server_time||Date.now()/1000)+(performance.now()-state.batchClock)/1000;
  for(const el of $('candidate-progress').querySelectorAll('[data-live-start]'))el.textContent=`${Math.max(0,now-Number(el.dataset.liveStart)).toFixed(1)}초`;
}
function renderBatchProgress(){
  const box=$('candidate-progress'),b=state.batchInfo;
  box.hidden=!b||b.id!==state.batch;if(box.hidden)return;
  const now=(b.server_time||Date.now()/1000)+(performance.now()-state.batchClock)/1000,p=b.progress;
  const active=['queued','waiting','running','cancelling'].includes(b.state);
  const waiting=!b.started||b.state==='queued'||b.state==='waiting';
  const waitEnd=waiting?(active?now:b.ended):b.started;
  const waitStart=b.queued_at||b.created;
  const seconds=x=>`${Math.max(0,x).toFixed(1)}초`;
  const live=start=>`<span data-live-start="${Number(start)}">계산 중</span>`;
  const phases=((waiting?null:p)?.phases||[]).map(x=>`<span class="phase ${x.state}"><span>${escape(x.label)}${x.shared?' · 공통':''}</span><strong>${x.skipped?'해당 없음':x.cached?'캐시 사용 · ':''}${x.skipped?'':(x.state==='running'&&active?live(x.started):seconds(x.seconds||0))}${x.step?` · ${x.step}/${x.steps}`:''}</strong></span>`).join('');
  const html=`<div class="batch-status"><strong>${escape(waiting&&active?'GPU·작업 대기 중':p?.message||stateNames[b.state])}</strong><span>${b.attempt>1?`재시도 ${b.attempt-1}회 · `:''}${p?.completed||0}/${b.count}장</span></div><div class="phase-list"><span class="phase"><span>대기</span><strong>${waiting&&active?live(waitStart):seconds((waitEnd||now)-waitStart)}</strong></span>${phases}</div>${p?.notice?`<p class="muted">${escape(p.notice)}</p>`:''}<div class="batch-actions">${active?'<button data-batch-action="cancel">후보 묶음 중지</button>':''}${['failed','cancelled','interrupted'].includes(b.state)?'<button data-batch-action="retry">미완료 후보 재시도</button>':''}<span class="muted">준비 시간은 공통이며, 이미지 생성 시간은 함께 처리한 묶음 기준입니다.</span></div>`;
  if(html!==batchProgressHTML){box.innerHTML=html;batchProgressHTML=html;}
  tickBatchClocks();
}
$('candidate-progress').addEventListener('click',async e=>{const button=e.target.closest('[data-batch-action]');if(!button)return;button.disabled=true;try{await api(`/api/candidates/${state.batch}/${button.dataset.batchAction}`,{method:'POST'});await refresh();}catch(e){toast(e.message,true);}});
setInterval(tickBatchClocks,1000);
let refreshing=false;
async function refresh(){if(refreshing)return;refreshing=true;try{
  const [assets,jobs]=await Promise.all([api('/api/assets'),api('/api/jobs')]);
  const before=JSON.stringify(state.assets);state.assets=assets;state.jobs=jobs;
  if(before!==JSON.stringify(assets))renderAssets();
  if($('jobs-dialog').open)renderJobs();else $('queue-count').textContent=jobs.filter(j=>['queued','waiting','running','cancelling'].includes(j.state)).length;
  if(state.asset){const ticket=state.detailTicket;const id=state.asset.id;const data=await api('/api/assets/'+id);if(ticket===state.detailTicket){
    if(JSON.stringify(data.versions)!==JSON.stringify(state.versions)){state.versions=data.versions;renderVersions();}
    state.asset=data.asset;renderSelected();
    await refreshBatch();
  }}
  $('connection').textContent='● 내부망 연결됨';
}catch(e){$('connection').textContent='연결 확인 중 · 작업은 서버에 보존됩니다';}finally{refreshing=false;}}

for(const tab of ['t2i','i23d']){
  $('tab-'+tab).addEventListener('click',()=>setTab(tab));
  $('tab-'+tab).addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?'t2i':e.key==='End'?'i23d':tab==='t2i'?'i23d':'t2i';setTab(next);$('tab-'+next).focus();}});
}
$('candidate-grid').addEventListener('click',e=>{const b=e.target.closest('[data-candidate]');if(b)selectVersion(b.dataset.candidate);});
$('candidate-batch').addEventListener('change',()=>{state.batch=$('candidate-batch').value;const v=state.batch==='legacy'?candidateVersions().find(v=>!v.batch_id):candidateVersions().filter(v=>v.batch_id===state.batch).sort((a,b)=>a.candidate_index-b.candidate_index)[0];if(v)selectVersion(v.id);else renderCandidates();refreshBatch();});
$('use-candidate').addEventListener('click',()=>setTab('i23d',false));
for(const b of document.querySelectorAll('[data-copy-prompt]'))b.addEventListener('click',async()=>{const field=$(b.dataset.copyPrompt);try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(field.value);else{field.focus();field.select();if(!document.execCommand('copy'))throw new Error('copy');}toast('원문을 복사했습니다.');}catch(e){field.focus();field.select();toast('선택된 원문을 Ctrl+C로 복사하세요.');}});

$('asset-list').addEventListener('click',e=>{const b=e.target.closest('[data-asset]');if(b)selectAsset(b.dataset.asset).catch(e=>toast(e.message,true));});
$('versions').addEventListener('click',e=>{const b=e.target.closest('[data-version]');if(b)selectVersion(b.dataset.version);});
$('search').addEventListener('input',e=>{state.query=e.target.value;renderAssets();});
$('filters').addEventListener('click',e=>{const b=e.target.closest('[data-filter]');if(!b)return;state.filter=b.dataset.filter;for(const x of $('filters').children)x.classList.toggle('active',x===b);renderAssets();});
$('material-mode').addEventListener('click',e=>{const b=e.target.closest('[data-mode]');if(!b)return;viewer?.setMode(b.dataset.mode);compareViewer?.setMode(b.dataset.mode);for(const x of $('material-mode').children)x.classList.toggle('active',x===b);});
$('view-buttons').addEventListener('click',e=>{const b=e.target.closest('[data-view]');if(b){viewer?.view(b.dataset.view);compareViewer?.view(b.dataset.view);}});
$('image-modes').addEventListener('click',e=>{const b=e.target.closest('[data-image]');if(b){state.imageMode=b.dataset.image;renderImage();}});
$('compare-version').addEventListener('change',renderComparison);
$('fit-view').addEventListener('click',()=>{viewer?.view('three');compareViewer?.view('three');});
$('grid-toggle').addEventListener('click',()=>{const on=$('grid-toggle').getAttribute('aria-pressed')!=='true';$('grid-toggle').setAttribute('aria-pressed',String(on));if(viewer)viewer.grid.visible=on;if(compareViewer)compareViewer.grid.visible=on;});
$('reference-image').addEventListener('click',()=>{const url=selected()?.media[state.imageMode];if(url)window.open(url,'_blank','noopener');});
for(const b of document.querySelectorAll('[data-close]'))b.addEventListener('click',()=>$(b.dataset.close).close());
$('jobs-toggle').addEventListener('click',()=>{renderJobs();$('jobs-dialog').showModal();});
$('help-open').addEventListener('click',()=>$('help-dialog').showModal());
$('new-asset').addEventListener('click',()=>$('new-dialog').showModal());
$('new-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;try{const a=await api('/api/assets',{method:'POST',body:{name:$('new-name').value}});$('new-dialog').close();$('new-form').reset();await refresh();await selectAsset(a.id);toast('에셋을 추가했습니다. 이미지를 올리거나 설명을 입력하세요.');}catch(e){toast(e.message,true);}finally{b.disabled=false;}});
act('sync',async()=>{const result=await api('/api/sync',{method:'POST'});await refresh();if(!state.asset&&state.assets.length)await selectAsset(state.assets[0].id);toast(`${result.added}개 새 결과를 가져왔습니다.${result.errors.length?` ${result.errors.length}개 파일은 연결하지 못했습니다.`:''}`);});
act('favorite',async()=>{await api('/api/assets/'+state.asset.id,{method:'PATCH',body:{favorite:!state.asset.favorite}});await refresh();});
act('approve',async()=>{const v=selected();await api('/api/versions/'+v.id,{method:'PATCH',body:{approved:!v.approved}});await refresh();toast(v.approved?'선택 완료 표시를 해제했습니다.':'이 버전을 선택 완료로 표시했습니다.');});
act('save-notes',async()=>{await api('/api/assets/'+state.asset.id,{method:'PATCH',body:{notes:$('notes').value}});toast('작업 메모를 저장했습니다.');});
act('rename',async()=>{const name=window.prompt('에셋 이름',state.asset.name);if(name?.trim()){await api('/api/assets/'+state.asset.id,{method:'PATCH',body:{name}});$('asset-title').textContent=name;await refresh();}});
$('upload-button').addEventListener('click',()=>$('upload-file').click());
$('upload-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file||!state.asset)return;const aid=state.asset.id;
  const data=new FormData();data.append('file',file);data.append('prompt',$('prompt').value);if(state.selected)data.append('base_version',state.selected);
  $('upload-button').disabled=true;try{const v=await api(`/api/assets/${aid}/upload`,{method:'POST',body:data});await refresh();await selectAsset(aid,v.id);toast('새 버전으로 업로드했습니다.');}catch(e){toast(e.message,true);}finally{e.target.value='';updateControls();}});
async function submit(action){
  const aid=state.asset.id,isImage=action==='reference',manual=$('image-seed-mode').value==='manual';
  const imageSeed=$('image-seed').value;
  if(isImage&&manual&&(!/^\d+$/.test(imageSeed)||Number(imageSeed)>2147470000))throw new Error('이미지 시작 seed를 0~2147470000 사이 정수로 입력하세요.');
  const result=await api(isImage?'/api/candidates':'/api/jobs',{method:'POST',body:{asset_id:aid,version_id:state.selected,action,
    prompt:$('prompt').value,model:$('model').value,rig_type:$('rig-type').value,
    seed:isImage?(manual?imageSeed:null):$('seed').value,
    ...(isImage?{count:candidateCount(),seed_mode:manual?'manual':'random'}:{}),
    faces:Number($('faces').value),texture:Number($('texture').value)}});
  await refresh();await selectAsset(aid,isImage?result.jobs[0].version:result.version);
  setTab(isImage?'t2i':'i23d',false);await refreshBatch();
  toast(isImage?`서로 다른 seed의 후보 ${result.count||result.jobs.length}장을 준비합니다.`:'작업을 대기열에 추가했습니다.');
}
act('generate-reference',()=>submit('reference'));act('generate-mesh',()=>submit('mesh'));act('prepare-button',()=>submit('prepare'));
$('job-list').addEventListener('click',async e=>{const button=e.target.closest('button');if(!button)return;button.disabled=true;try{
  if(button.dataset.jobView){const j=state.jobs.find(j=>j.id===button.dataset.jobView);await selectAsset(j.parent,j.version);$('jobs-dialog').close();}
  if(button.dataset.jobCancel){await api(`/api/jobs/${button.dataset.jobCancel}/cancel`,{method:'POST'});await refresh();}
  if(button.dataset.jobRetry){await api(`/api/jobs/${button.dataset.jobRetry}/retry`,{method:'POST'});await refresh();}
}catch(e){toast(e.message,true);}finally{button.disabled=false;}});
$('download').addEventListener('click',()=>{$('export-name').value=state.asset.name;$('export-height').value='';$('export-pivot').value='keep';$('export-yaw').value='0';$('download-dialog').showModal();});
$('download-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;try{
  const query=new URLSearchParams({name:$('export-name').value,height:$('export-height').value,pivot:$('export-pivot').value,yaw:$('export-yaw').value,bundle:b.value});
  const response=await fetch(`/api/versions/${state.selected}/download?${query}`);if(!response.ok)throw new Error((await response.json()).error);
  const blob=await response.blob(),url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=$('export-name').value+(b.value==='1'?'.zip':'.glb');link.click();setTimeout(()=>URL.revokeObjectURL(url),60000);$('download-dialog').close();toast('파일을 내려받았습니다. 작업 툴에서 크기와 재질을 확인하세요.');
}catch(e){toast(e.message,true);}finally{b.disabled=false;}});

// Optional structured navigation uses the same live selection as the visible UI.
if(document.modelContext?.registerTool){try{document.modelContext.registerTool({name:'select_review_asset',description:'라이브러리 에셋을 선택하고 검토 화면을 엽니다. 생성 작업을 시작하지 않습니다.',inputSchema:{type:'object',properties:{asset_id:{type:'string'}},required:['asset_id'],additionalProperties:false},annotations:{readOnlyHint:true},async execute(input){if(typeof input.asset_id!=='string'||!state.assets.some(a=>a.id===input.asset_id))throw new Error('Unknown asset');await selectAsset(input.asset_id);return {asset_id:state.asset.id,version_id:state.selected};}});}catch(e){/* Unsupported draft API; the visible UI remains available. */}}

updateCandidateSettings();await refresh();const params=new URLSearchParams(location.search);const initial=state.assets.find(a=>a.id===params.get('asset'))||state.assets[0];
if(initial)await selectAsset(initial.id,params.get('version')).catch(e=>toast(e.message,true));setTab(params.get('tab')==='t2i'?'t2i':'i23d',false);updateControls();setInterval(refresh,3000);
