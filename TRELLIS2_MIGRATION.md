# TRELLIS → TRELLIS.2 마이그레이션 계획

> 대상 프로젝트: `t2o_pipeline` (USD → 3D 사물 에셋 생성)
> 목표: Stage 2의 3D 생성 백엔드를 **TRELLIS v1 (text-to-3D)** → **TRELLIS.2 (image-to-3D)** 로 교체
> 작성 기준: A100 80GB × 4 서버 / CUDA 12.4 툴체인 / Docker 컨테이너 `previs-prep`

---

## 0. 요약 (TL;DR)

- **이건 드롭인 버전업이 아니다.** TRELLIS.2는 **이미지 입력 전용**이고 text-to-3D 파이프라인이 없다.
  현재 파이프라인은 text-to-object 구조이므로, 텍스트와 TRELLIS.2 사이에 **Text→Image(T2I) 다리**를 새로 넣어야 한다.
- 채택 전략: **T2I 다리 추가** (아래 §2). 기존 USD 파싱·번역·필터 단계는 대부분 재사용한다.
- 환경은 **기존 `trellis` conda env는 보존**하고, **새 `trellis2` conda env**를 별도로 만든다 (torch 2.6 / cu124). 롤백·A/B 비교를 위해 병존.
- 팀 병합 단위는 **Docker 이미지 + `environment.yml`/락파일 + Dockerfile**로 코드화한다 (현재 의존성 명세가 git에 전혀 없음 — 최우선 보완 대상).

---

## 1. 현황과 변경 격차 (Gap Analysis)

### 1.1 현재 데이터 흐름
```
USD 파싱 (usd_parse_and_augment.py)
  → 객체 설명(한글) 추출
  → ollama 번역/필터 (step1_translate / step2_filter)
  → TrellisTextTo3DPipeline.run(prompt)      ← Stage 2 (교체 대상)
  → outputs['gaussian'/'radiance_field'/'mesh']
  → render_utils.render_video / postprocessing_utils.to_glb
  → GLB → USD (usd_from_gltf) → Reference 주입
```

### 1.2 API·환경 격차표

| 항목 | TRELLIS v1 (현재) | TRELLIS.2 (목표) |
|---|---|---|
| 입력 | 텍스트 프롬프트 | **PIL 이미지** |
| import | `from trellis.pipelines import TrellisTextTo3DPipeline` | `from trellis2.pipelines import Trellis2ImageTo3DPipeline` |
| util | `from trellis.utils import render_utils, postprocessing_utils` | `from trellis2.utils import render_utils`, `from trellis2.renderers import EnvMap`, `import o_voxel` |
| 모델 | `microsoft/TRELLIS-text-*` | `microsoft/TRELLIS.2-4B` (4B) |
| run() | `pipeline.run(prompt, seed=…, sparse_structure_sampler_params=…, slat_sampler_params=…)` | `pipeline.run(image)` → `mesh = [0]` |
| 출력 | dict(`gaussian`,`radiance_field`,`mesh`) | O-Voxel **mesh 객체** (`.vertices/.faces/.attrs/.coords/.layout/.voxel_size`, `.simplify()`) |
| GLB | `postprocessing_utils.to_glb(gaussian, mesh, simplify, texture_size)` | `o_voxel.postprocess.to_glb(vertices, faces, attr_volume, coords, attr_layout, voxel_size, aabb, decimation_target, texture_size, remesh…)` |
| 렌더 | `render_utils.render_video(outputs['gaussian'][0])['color']` | `render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))` (**HDRI EnvMap 필요**) |
| 재질 | 단일 텍스처 | **PBR** (BaseColor/Roughness/Metallic/Opacity), webp 텍스처 |
| torch/CUDA | 2.4.0 / cu121 | **2.6.0 / cu124** |
| 네이티브 확장 | spconv-cu120, kaolin, nvdiffrast, flexicubes | + **cumesh, o-voxel, flexgemm, flash-attn** |
| 주 영향 파일 | `previz_pipeline/trellis_inference_core.py` | 동일 파일 전면 개편 |

### 1.3 하드웨어 적합성
- TRELLIS.2 요구: Linux, NVIDIA GPU ≥24GB (A100/H100 검증). → **A100 80GB로 충분**, 여유 큼.
- 4B 모델 + 1536³ 해상도 옵션은 VRAM 사용이 큼 → 80GB에서 안전.

---

## 2. 채택 아키텍처 — T2I 다리 추가

```
USD 파싱 → 객체 설명(한글) → ollama 번역/필터
   → [신규] Text→Image (SDXL 또는 FLUX.1)     ← 참조 이미지 1~N장 생성
   → Trellis2ImageTo3DPipeline.run(image)      ← Stage 2 (신규 백엔드)
   → O-Voxel mesh (PBR)
   → o_voxel.postprocess.to_glb(...)           → PBR GLB (webp)
   → GLB → USD (usd_from_gltf) → Reference 주입
```

### 2.1 T2I 백엔드 선택지 (별도 결정 필요)
| 후보 | 장점 | 단점 |
|---|---|---|
| **FLUX.1-dev** (권장) | 프롬프트 충실도·품질 높음, 단일 객체/무배경 튜닝 용이 | VRAM 크고 라이선스(비상업 dev) 확인 필요 |
| **SDXL + 무배경 LoRA** | 가볍고 생태계 성숙, 상업 사용 용이 | 프롬프트 충실도 FLUX보다 낮음 |
| ollama 멀티모달 경유 | 이미 ollama 사용 중 | 3D용 깨끗한 단일객체 이미지엔 부적합 |

> **권장 산출 규격:** 정사각(예: 1024²), 배경 흰색/투명, 객체 1개 중앙 배치, 그림자 최소. TRELLIS.2 입력 품질이 최종 3D 품질을 좌우하므로 배경 제거(rembg)·중앙 정렬 전처리 단계를 T2I 뒤에 둔다.

---

## 3. 환경 구성 계획

### 3.1 새 conda env (기존과 병존)
```bash
# 컨테이너 내부
source /root/miniconda3/etc/profile.d/conda.sh
cd /data/previs_object/t2o_pipeline
git clone https://github.com/microsoft/TRELLIS.2.git trellis2_src   # 신규 소스
cd trellis2_src
# 공식 설치 (새 env 생성). 이름을 trellis2로 고정
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
# ↑ 생성되는 env 이름 확인 후 `conda rename` 또는 setup 옵션으로 trellis2 지정
```
- 기존 `trellis` env는 **건드리지 않는다** (롤백/A-B 비교용).
- torch 2.6/cu124 기준이므로 spconv·kaolin·nvdiffrast를 이 env에서 **재빌드**해야 함 (v1 env 재사용 불가).

### 3.2 모델 다운로드
```bash
huggingface-cli download microsoft/TRELLIS.2-4B --local-dir hf_models/TRELLIS.2-4B
```
- 게이트/라이선스 여부 확인. 기존 `hf_models`는 `/data/t2o_pipeline_backup/hf_models` 심볼릭 링크 → 동일 규칙으로 배치.
- HDRI EnvMap 파일(`assets/hdri/*.exr`)도 확보 (렌더용).

### 3.3 팀 병합용 코드화 (필수 신규)
- `environment-trellis2.yml` ← `conda env export --no-builds`
- `requirements-trellis2.lock` ← `pip freeze` (컴파일 확장 버전 고정)
- `Dockerfile.trellis2` ← 베이스 `nvidia/cuda:12.4.*` → conda env 생성 → TRELLIS.2 setup → 모델 마운트 규약
- 실행 스크립트에 **명시적 `conda activate trellis2`** 추가 (현재는 암묵 전제라 타인 환경에서 실패).

---

## 4. 코드 변경 맵

### 4.1 `previz_pipeline/trellis_inference_core.py` (핵심 개편)
| 위치 | 현재 | 변경 |
|---|---|---|
| L13~14 | `SPCONV_ALGO`, `ATTN_BACKEND=xformers` | `ATTN_BACKEND`는 flash-attn 기준으로. `OPENCV_IO_ENABLE_OPENEXR=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 추가 |
| L22~23 import | `TrellisTextTo3DPipeline`, `postprocessing_utils` | `Trellis2ImageTo3DPipeline`, `trellis2.utils.render_utils`, `trellis2.renderers.EnvMap`, `o_voxel` |
| `load_pipeline()` (L135~) | `TrellisTextTo3DPipeline.from_pretrained` | `Trellis2ImageTo3DPipeline.from_pretrained("…/TRELLIS.2-4B")` + EnvMap 로드 |
| `run()` 시그니처 | `run(prompt, …)` | `run(image, …)` — 호출부에서 T2I 결과 이미지를 전달 |
| `pipeline.run(...)` (L542) | `run(prompt, seed, …sampler_params)` | `mesh = pipeline.run(image)[0]; mesh.simplify(…)` |
| 렌더 (L556~) | `render_video(outputs['gaussian'][0])` | `make_pbr_vis_frames(render_video(mesh, envmap=envmap))` |
| GLB export (L584~) | `postprocessing_utils.to_glb(gaussian, mesh, …)` | `o_voxel.postprocess.to_glb(vertices=…, faces=…, attr_volume=mesh.attrs, coords=…, voxel_size=…, aabb=[[-.5]*3,[.5]*3], texture_size, remesh=True); glb.export(path, extension_webp=True)` |
| PLY (L610) | `outputs['gaussian'][0].save_ply` | 제거 또는 대체 (v2엔 gaussian 없음) |
| MP4 3종 (gs/rf/mesh) | gaussian/rf/mesh 각각 | 단일 PBR mesh 비디오로 축소 |
| 썸네일 (L648~) | `video_gs` 프레임 | PBR 렌더 프레임 기준으로 인덱스 재조정 |

### 4.2 신규 파일
- `previz_pipeline/text_to_image.py` — 프롬프트 → 참조 이미지 + 배경제거/정렬 전처리
- 호출 오케스트레이션은 `json_parse_and_inference.py`에서 T2I → TrellisInferenceCore 순서로 연결

### 4.3 스크립트/문서
- `object_generate.sh`, `run_usd_to_3D_object.sh`: `TRELLIS_MODEL_PATH` 기본값 `microsoft/TRELLIS.2-4B`로, T2I 관련 env 추가, `conda activate trellis2` 삽입
- `OBJECT_GENERATION_MIGRATION.md` §4(Stage 2) 갱신

### 4.4 다운스트림 (Stage 3) 검증 포인트
- `usd_from_gltf`가 **PBR 멀티맵 + webp 텍스처 GLB**를 정상 변환하는지 확인. 미지원 시 `extension_webp=False`(png)로 export하거나 텍스처 변환 단계 추가.

---

## 5. 단계별 실행 계획 (체크리스트)

- [ ] **Phase 0 — 준비/브랜치**
  - [ ] `t2o_pipeline`에 `feature/trellis2` 브랜치 생성
  - [ ] T2I 백엔드 확정 (FLUX vs SDXL) — §2.1
  - [ ] 현재 `trellis` env 스냅샷 (`conda env export`) 백업
- [ ] **Phase 1 — 환경 구축**
  - [ ] TRELLIS.2 clone + `setup.sh`로 `trellis2` env 생성
  - [ ] `TRELLIS.2-4B` 모델 + HDRI 다운로드
  - [ ] 공식 `example.py`로 스모크 테스트 (샘플 이미지 → GLB 성공 확인)
- [ ] **Phase 2 — T2I 다리**
  - [ ] `text_to_image.py` 구현 (프롬프트→이미지, rembg, 중앙정렬)
  - [ ] 번역/필터 출력 프롬프트로 이미지 품질 튜닝
- [ ] **Phase 3 — Stage 2 개편**
  - [ ] `trellis_inference_core.py` import/load/run/export/render 교체 (§4.1)
  - [ ] `run(image)` 인터페이스로 호출부 연결
- [ ] **Phase 4 — 다운스트림 검증**
  - [ ] PBR/webp GLB → USD 변환 확인, 필요 시 export 옵션 조정
- [ ] **Phase 5 — 스크립트/E2E**
  - [ ] `object_generate.sh` 등 모델·env·T2I 반영
  - [ ] USD 1건으로 전체 파이프라인 E2E 통과
- [ ] **Phase 6 — 품질 A/B**
  - [ ] 동일 프롬프트로 v1 vs v2 결과 비교 (형상/재질/시간/VRAM)
- [ ] **Phase 7 — 병합 패키징**
  - [ ] `environment-trellis2.yml` / 락파일 / `Dockerfile.trellis2` 커밋
  - [ ] 문서 갱신 후 PR

---

## 6. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| **text-to-3D 상실** (핵심) | 텍스트→3D 직접성 저하, T2I 품질에 결과 종속 | T2I 프롬프트/전처리 튜닝, 참조이미지 N장 후보 중 선택 |
| PBR/webp GLB의 USD 변환 비호환 | Stage 3 실패 | png export 폴백, usd_from_gltf 사전 검증 |
| torch 2.6/cu124 재빌드 실패 (flash-attn/spconv/cumesh) | 환경 구축 지연 | 새 env 격리, 공식 setup.sh 옵션 준수, 빌드 로그 보존 |
| 4B 모델 라이선스/게이트 | 팀 배포 제약 | HF 라이선스 확인 후 모델 배포 규약 문서화 |
| 신규 네이티브 확장(o-voxel/flexgemm) API 변동성 | 유지보수 부담 | 커밋 SHA 고정, `trellis2_src`를 서브모듈/버전 태그로 고정 |
| 병행 중 env 오염 | v1 결과 재현 불가 | `trellis`/`trellis2` env 완전 분리, 스크립트에 activate 명시 |

---

## 7. 열린 결정 사항
1. ~~T2I 백엔드~~ → **FLUX.1-dev 확정** (2026-07). 비상업 라이선스이므로 팀 배포 시 사용 범위 확인 필요.
2. 참조 이미지 장수(1장 고정 vs N장 생성 후 선별)
3. ~~TRELLIS.2 소스 관리 방식~~ → **`.gitignore` + SHA 고정 별도 clone 확정** (2026-07-21).
   `trellis2_src/`(6.5GB)는 repo에 포함하지 않는다. 재현 시 SHA `75fbf0183001ed9876c8dbb35de6b68552ee08bd`로
   `git clone --recursive` 후 checkout. 절차는 `NEW_SERVER_SETUP.md` §3.1 참조.
4. 렌더 산출물 범위: PBR 프리뷰 mp4/썸네일을 어디까지 유지할지

---

## 부록 A. Phase 1 실행 결과 (2026-07-16)

### A.1 완료 항목
- **소스**: `trellis2_src/` 에 `microsoft/TRELLIS.2` clone (SHA `75fbf0183001ed9876c8dbb35de6b68552ee08bd`), 서브모듈 eigen 포함.
- **conda env `trellis2`** 생성 (`/root/miniconda3/envs/trellis2`, python 3.10):
  - `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`
  - 네이티브 확장 빌드 성공(arch `sm_80` / A100): `nvdiffrast 0.3.3`, `nvdiffrec_render`, `cumesh 0.0.1`, `o_voxel 0.0.1`, `flex_gemm 1.0.0`, `flash-attn 2.7.3`
  - `--basic` 스택: transformers 5.x, trimesh, kornia, timm, gradio 6.0.1 등
- **모델**: `microsoft/TRELLIS.2-4B` (16GB, 22 files) → `/data/t2o_pipeline_backup/hf_models/TRELLIS.2-4B` (repo 심볼릭 `hf_models/TRELLIS.2-4B`). MIT, 게이트 없음.
- **HDRI**: `trellis2_src/assets/hdri/*.exr` (forest/city/studio 등) repo에 포함 — 렌더용.
- **락파일**: `environment-trellis2.yml`, `requirements-trellis2.lock` 추출 완료 (팀 병합용).

### A.2 setup.sh 함정 (재현 시 반드시 반영 — Dockerfile에도)
- **flash-attn**: 공식 `setup.sh`의 flash-attn 라인은 `--no-build-isolation`이 **누락**되어 있어, 빌드 격리 환경에서 `ModuleNotFoundError: No module named 'torch'`로 실패한다.
  → 우회: 나머지 확장 설치 후 별도로
  `pip install flash-attn==2.7.3 --no-build-isolation --no-deps` (arch 한정 `TORCH_CUDA_ARCH_LIST=8.0`, `MAX_JOBS=16`).
- **sudo/libjpeg**: `--basic`의 `sudo apt install libjpeg-dev`는 컨테이너에 sudo가 없어 실패 → 사전에 `apt-get install -y libjpeg-dev` 선설치, setup.sh에서 `sudo ` 제거.
- **Pillow-SIMD 충돌**: `--basic`이 `pillow-simd`(구버전 9.5 포크)를 설치하는데, 이게 표준 Pillow를 가로채 `PIL._webp`에 `HAVE_WEBPANIM`이 없어 imageio·webp 처리가 깨진다(GLB의 `extension_webp=True`에 영향).
  → 수정: `pip uninstall -y Pillow-SIMD pillow && pip install "pillow>=11"` (현재 12.3.0).
- **OpenCV EXR 미지원**: 설치되는 `opencv-python-headless 5.0.0.93`은 `OpenEXR: NO`로 빌드되어 HDRI(.exr)를 못 읽는다(`cv2.imread`→None). example.py의 envmap 로딩이 실패.
  → 수정: `pip install OpenEXR`(3.4.x) 후 `OpenEXR.File(path).channels()["RGB"].pixels`로 RGB float32 직접 로드(BGR2RGB 불필요). freeimage 3.16 백엔드는 이 EXR 압축을 못 읽으니 사용 불가.
- 빌드는 GPU 불필요(nvcc 컴파일만). 캐시된 flash-attn 휠: `/root/.cache/pip/wheels/.../flash_attn-2.7.3-cp310-...whl` (재설치 시 재사용).

### A.3 미해결 블로커 — 컨테이너 GPU 접근 (스모크 테스트 전 필수)
- 실행 중 컨테이너 `previs-prep` 내부에서 `nvidia-smi` → **"Failed to initialize NVML: Unknown Error"**, `torch.cuda.is_available()` → False.
- 원인: 호스트 드라이버(580)/`daemon-reload` 이후 실행 중 컨테이너의 GPU device cgroup 접근이 무효화됨. `/dev/nvidia*`·라이브러리는 정상, 새 컨테이너 `--gpus all`은 A100×4 정상.
- 영향: `import o_voxel` / `import flex_gemm` 이 import 시 CUDA 드라이버 초기화로 실패(`RuntimeError: 0 active drivers`). 따라서 **import 검증·스모크 테스트 모두 GPU 필요**.
- 해결: `docker restart previs-prep` 1회 (파일시스템·conda env·`/data` 보존, 실행 중 프로세스만 재기동). 현재 사용자 결정에 따라 **재시작 보류** → 재시작 후 아래 검증 재개.
- GPU 없이 import 성공 확인: `torch`, `flash_attn`, `nvdiffrast`, `cumesh`.

### A.4 재시작 후 Phase 1 검증 진행 (2026-07-16 재시작 완료)
- [x] `docker restart previs-prep` → GPU 복구: `nvidia-smi` A100×4, `torch.cuda.is_available()=True`
  - ⚠️ 이 컨테이너는 PID1이 bare `/bin/bash`라 재시작 시 **sshd가 자동 기동되지 않음** → 재시작 직후 `docker exec previs-prep service ssh start` 필요(host key·/etc/init.d/ssh 존재). 향후: entrypoint에 sshd 기동 추가 권장.
- [x] `import o_voxel`, `import flex_gemm`, `cumesh`, `nvdiffrast`, `flash_attn` 전부 정상
- [x] EXR(HDRI) 로딩 정상(OpenEXR 패키지), 파이프라인 backend 확인: `flex_gemm` + `flash_attn`
- [ ] **BLOCKER — DINOv3 게이트 접근**: TRELLIS.2-4B의 `image_cond_model`이 `facebook/dinov3-vitl16-pretrain-lvd1689m`(게이트 레포). 현재 로그인 계정 `nayeon2da`가 미승인 → 403 GatedRepoError.
  - 해결(사용자 액션): https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m 에서 `nayeon2da`로 접근 요청/약관 수락 → 승인 후 기존 토큰으로 자동 동작.
  - `pipeline.json`의 `image_cond_model.args.model_name`에 하드코딩됨. (대안: 승인 후 로컬 캐시/미러 경로로 교체 가능하나 비권장.)
- [ ] 승인 후 `python smoke_test.py` 재실행 → `smoke_out/sample.glb`(+선택적 `sample.mp4`) 생성 확인 → Phase 1 완료

---

## 부록 B. DINOv3 승인 대기 중 진행 작업 (2026-07-16)

### B.1 GLB→USD 호환성 점검 (Stage 3) — webp 미지원 확정
- 번들된 `usd_from_gltf`(`usd_from_gltf_build/bin/`) 바이너리 분석 결과, 지원 glTF 확장은
  `KHR_draco_mesh_compression`, `KHR_materials_pbrSpecularGlossiness`, `KHR_materials_unlit`, `KHR_texture_transform`뿐 — **`EXT_texture_webp` 없음**, 바이너리에 `webp` 문자열 자체가 없음(텍스처는 `.png`/`.jpg`만).
- **결론:** TRELLIS.2 `to_glb(..., extension_webp=True)`(webp)는 이 도구로 텍스처 변환 실패.
  → **Phase 3에서 `extension_webp=False`(PNG)로 export** (코드에 반영 완료). PBR metallic/roughness는 glTF 코어라 정상 변환됨.
- `merge_glb_to_usd.py`의 `_TEXTURE_EXTENSIONS`에는 `.webp`가 이미 포함(후처리 rename은 webp도 처리) — 즉 변환기만 PNG면 downstream 정규화는 문제없음.

### B.2 Phase 2 코드 — `previz_pipeline/text_to_image.py` (FLUX Text→Image)
- FLUX.1-dev 래퍼(`TextToImage`): prompt → 단일객체·흰배경·중앙 정렬 참조 이미지. 문법검증 완료.
- FLUX.1-dev는 계정 `nayeon2da`로 **접근 가능(게이트 승인됨)** → Phase 2는 승인 대기 없음. 모델(~24GB) 다운로드는 HF 504 재시도 루프로 진행 중.
- 의존성 추가: `diffusers 0.39`, `accelerate`, `sentencepiece`(transformers는 **5.14 유지** — TRELLIS.2 안전). `FluxPipeline` import 확인.
- 단독 스모크(`python text_to_image.py`)는 FLUX 다운로드 완료 시 **DINOv3 없이 실행 가능**.

### B.3 Phase 3 코드 — `previz_pipeline/trellis2_inference_core.py`
- `Trellis2InferenceCore(TrellisInferenceCore)`: 기존 경로/네이밍/manifest/CSV/generation-json 로직 **전부 상속 재사용**, `load_pipeline()`·`_generate_single()`만 TRELLIS.2용으로 오버라이드.
- 흐름: prompt →(FLUX)→ 이미지 →(`pipeline.run(image, preprocess_image=True)`)→ `MeshWithVoxel` → `mesh.simplify()` → `o_voxel.postprocess.to_glb(..., extension_webp=False)` → PNG GLB. 렌더(PBR+HDRI)는 best-effort.
- 처리한 함정: (a) base `__init__`이 v1 부재 시 raise → `_base_mod.TRELLIS_AVAILABLE=True`로 우회; (b) base가 강제하는 `ATTN_BACKEND=xformers` → trellis2 import 전 `flash_attn`으로 재설정(xformers 미설치).
- 검증: import·인스턴스화·헬퍼 상속·오버라이드·백엔드(`flex_gemm`+`flash_attn`) 확인. **E2E(run) 테스트는 DINOv3 승인 후.**

### B.4 승인 후 즉시 할 일
- [ ] DINOv3 승인 → `smoke_test.py`로 Phase 1 마감
- [ ] FLUX 다운로드 완료 → `text_to_image.py` 단독 스모크(참조 이미지 생성 확인)
- [ ] `trellis2_inference_core.py` E2E: 샘플 USD 1건 → prompt → 이미지 → GLB(PNG) → usd_from_gltf 변환까지 관통

---

## 부록 C. Phase 1 완료 + 추가 게이트/버전 이슈 (2026-07-20)

### C.1 ✅ Phase 1 스모크 통과 (TRELLIS.2 image→3D 완전 동작)
- `smoke_test.py` (assets/example_image/T.png → TRELLIS.2-4B) 성공:
  - pipeline load → inference **36.4s** → PBR 렌더 → GLB export **41.7s**, **TOTAL 174s**
  - 산출: `smoke_out/sample.glb` **42.8MB**(PBR, remesh 10.5M→969K faces), `sample.mp4` 4.2MB
- 전체 스택 검증: flex_gemm+flash_attn, o_voxel/cumesh/nvdiffrast, DINOv3, RMBG-2.0, PNG GLB export.

### C.2 🔧 결정적 수정: transformers 5.14 → **4.57.6**
- `--basic`이 최신 `transformers 5.14`를 설치 → TRELLIS.2의 `image_feature_extractor.py`가 접근하는 `DINOv3ViTModel.layer`가 5.14엔 **없음** → `AttributeError: ... has no attribute 'layer'`.
- 검증: `.layer` 속성은 **transformers 4.57.6에 존재**, 5.14엔 없음. → `pip install transformers==4.57.6` (huggingface_hub도 0.36.2로 동반 하향). 락파일 반영 완료.
- **Dockerfile/재현 시 transformers는 반드시 `==4.57.6` 핀** (setup.sh의 unpinned `transformers`가 5.x를 끌어옴).

### C.3 추가 게이트 의존성 2개 (DINOv3 외)
TRELLIS.2-4B `pipeline.json`이 요구하는 게이트 HF 레포:
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (image_cond_model) — 승인 완료(계정 raengs).
- `briaai/RMBG-2.0` (rembg_model=BiRefNet) — 승인 완료. **단, 비상업 라이선스.**

### C.4 ⚠️ 상업화 라이선스 감사 (사업화 전제 시 필수 검토)
| 구성요소 | 라이선스 | 상업 가능 | 조치 |
|---|---|---|---|
| TRELLIS.2-4B | MIT | ✅ | - |
| FLUX.1-dev (T2I 초기안) | 비상업 | ❌ | → **FLUX.1-schnell(Apache-2.0)로 교체 완료**(코드 반영). schnell 레포는 게이트 → 약관 수락 필요 |
| **briaai/RMBG-2.0** (TRELLIS.2 내장 rembg) | **비상업** | ❌ | **미해결** — 상업 시 BRIA 상업 라이선스 계약 또는 pipeline.json의 rembg를 상업가능 모델로 교체 필요 |
| DINOv3 | DINOv3 License | ⚠️ | 라이선스 조건 확인 필요 |
- T2I 이미지가 이미 흰 배경이면 rembg가 불필요할 수 있음 → `preprocess_image=False` + rembg_model 우회 가능성 조사 가치 있음(상업 라이선스 회피 경로).

### C.5 ✅ 전체 파이프라인 E2E 검증 완료 (2026-07-20)
- `trellis2_inference_core.py` E2E: prompt "a worn leather armchair" → FLUX(subprocess/t2i) → TRELLIS.2 → PNG GLB **41.8MB** + 프리뷰/provenance. `success=True`, 162s.
- **Stage 3 (GLB→USD) 검증**: 위 GLB를 `usd_from_gltf`로 변환 → `e2e_chair.usdz` **25.5MB**. 내부에 PBR 3맵 포함: `image0_rgb`(베이스컬러)·`image1_metal`(메탈릭)·`image1_rough`(러프니스). webp 경고는 무해, **PNG export 결정이 정확했음** 확정.
- 즉 **prompt → FLUX → TRELLIS.2 → PNG GLB → USDZ(PBR)** 전 구간 관통 성공. 기술 미지수 0.

### C.6 오케스트레이션 통합 (json_parse_and_inference.py)
- `--backend {trellis,trellis2}` 추가, **기본값 `trellis2`**. `--backend trellis`로 v1 롤백 가능.
- `--t2i_model_path`(기본 FLUX.1-schnell), `--pipeline_type` 추가.
- `Trellis2InferenceCore`가 base의 `process_batch_from_records`→`_process_file_batch`를 상속하고 `_generate_single`만 v2로 오버라이드 → 기존 오케스트레이션/레코드 로딩/출력구조 그대로 재사용.
- import-guard로 v2 미설치 환경에서도 v1 경로 정상.

### C.7 결정/이슈 로그
- **RMBG-2.0**: 사용자 결정 — **그대로 사용**(비상업 라이선스 인지함, 추후 재검토).
- **FLUX.1-schnell**(Apache-2.0): 게이트 승인 완료. 다운로드 시 **`HF_HUB_DISABLE_XET=1` 필수** — hf-xet 백엔드가 이 레포에서 "Unable to parse string as hex hash value"로 실패함.
- FLUX.1-dev(54GB)는 기능검증용으로 보유. 상업본은 schnell.

### C.8 남은 것 (제품화)
- [ ] FLUX.1-schnell 다운로드 완료 후 T2I를 schnell로 최종 스모크
- [ ] 실제 `movie_usd/` USD 1건으로 `--backend trellis2` 관통 (acceptance)
- [ ] `Dockerfile.trellis2` 팀 패키징 (락파일·transformers==4.57.6·xet-off·게이트 모델 규약 반영)

---

## 부록 D. 새 서버 이관 (2026-07-21)

### D.1 배경
기존 컨테이너 `previs-prep`의 GPU cgroup이 재차 끊겨(§A.3과 동일 증상) 재시작이 필요하나,
다른 사용자의 장시간 작업(codex 세션 2개, `python3 server.py`, 각 ~18시간)이 돌고 있어 재시작 불가.
→ **남은 acceptance 작업(§C.8)을 새 서버에서 진행하기로 결정.**

- 재시작 전 점검용 스크립트를 호스트에 작성: `check_container_users.sh`
  (SSH 세션 / VS Code 원격 / 대화형 셸 / 장시간 작업 / GPU 점유를 확인, exit 0=안전 1=사용중.
  컨테이너 내부 `nvidia-smi`는 cgroup 차단 시 실패하므로 **호스트** nvidia-smi를 `docker top`의 호스트 PID와 교차 대조한다.)

### D.2 이관 방식 — 이미지 이동 대신 재구축
현황 실측:

| 위치 | 크기 | 비고 |
|---|---|---|
| 이미지 `previs-prep-dev:latest` | 318GB | 7개월 전 빌드 — **trellis2 작업 미포함** |
| 컨테이너 쓰기 레이어 | 120GB | conda env(`/root/miniconda3` 46GB)가 여기 |
| 바인드마운트 `/dev/data/PrevisPrep` → `/data` | 242GB | 코드·모델. 대부분은 백업 잔여물 |
| 볼륨 `previz_nas` → `/nas` | 비어 있음 | - |

`docker save`로는 의미가 없다(이미지에 trellis2가 없음). **새 서버에 conda env를 새로 구축**하고,
코드는 git으로, 모델(~70GB)은 별도 다운로드/rsync한다.

### D.3 되돌리기 원칙
새 서버는 "환경을 만들어 가져오는 곳"이 아니라 **"방해받지 않고 실행하는 곳"**이다.
기존 서버의 conda env는 이미 검증 완료(§C.1, §C.5)이므로 **env 바이너리는 되돌리지 않는다.**
회수 대상은 **코드·문서·실행결과(git)뿐.**

상세 절차는 **`NEW_SERVER_SETUP.md`** 참조 (환경 구축 함정 5개, 게이트 레포, 실행 인자, 미검증 지점).
