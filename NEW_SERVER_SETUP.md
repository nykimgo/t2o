# 새 서버 TRELLIS.2 환경 구축 & 파이프라인 실행 인수인계

> 작성: 2026-07-21 / 기존 서버(A100 80GB×4, 컨테이너 `previs-prep`)에서 작성
> 대상 독자: **새 서버에서 이 작업을 이어받는 사람(또는 Claude)**
> 상세 배경·설계 근거는 같은 디렉토리의 `TRELLIS2_MIGRATION.md` 참조 (특히 부록 A/B/C)

---

## 0. 왜 새 서버로 가나 (컨텍스트)

- 기존 서버의 TRELLIS.2 환경은 **이미 완성·검증 완료**다. 스모크 테스트와 E2E(prompt→FLUX→TRELLIS.2→GLB→USDZ) 모두 통과했다 (`TRELLIS2_MIGRATION.md` §C.1, §C.5).
- 남은 건 **실제 `movie_usd` USD 1건으로 전체 파이프라인 관통(acceptance)** 뿐이다.
- 그런데 기존 컨테이너는 (a) GPU cgroup이 끊겨 재시작이 필요하고, (b) 재시작하면 다른 사용자의 장시간 작업(codex 세션 2개, `python3 server.py`)이 죽는다. 그래서 **새 서버에서 돌린다.**

### 이 작업의 성격 — 오해 주의
**"새 서버에서 환경을 만들어 기존 서버로 가져오는 것"이 아니다.**
기존 서버의 conda env는 멀쩡하므로 **env 바이너리를 되돌릴 필요가 없다.**
새 서버는 **"방해받지 않고 파이프라인을 돌려보는 곳"**이고,
되돌릴 것은 **git으로 관리되는 코드/문서/실행결과뿐**이다.

```
[기존 서버] 코드 push
      ↓ git clone
[새 서버]  conda env 구축 → 파이프라인 실행 → 코드 수정/결과 기록
      ↓ git push  (코드·문서만. conda env는 절대 복사하지 않는다)
[기존 서버] git pull → 반영 끝
```

---

## 1. 목표 (Definition of Done)

- [ ] 새 서버에 conda env `trellis2` + `t2i` 구축, `import o_voxel` / `flex_gemm` 정상
- [ ] 실제 `movie_usd/` USD 1건으로 `--backend trellis2` 전 구간 관통 (USD 파싱 → FLUX → TRELLIS.2 → GLB → USD 병합)
- [ ] **FLUX.1-schnell로 T2I 스모크** (기존 검증은 FLUX.1-dev로만 돌았다 — 아래 §6 참조)
- [ ] 실행 중 발견한 함정·수정사항을 `TRELLIS2_MIGRATION.md`에 부록으로 추가하고 push

---

## 2. 새 서버 요구사항

| 항목 | 기존 서버 | 새 서버 조건 |
|---|---|---|
| OS | Ubuntu 22.04.4 LTS | 동일 권장 |
| CUDA | 12.4 (`nvcc 12.4.131`) | **12.4 필수** (torch 2.6.0+cu124 기준) |
| GPU | A100 80GB ×4 (sm_80) | **VRAM 24GB 이상.** 아키텍처가 다르면 §3의 `TORCH_CUDA_ARCH_LIST`를 바꿔야 한다 (A100=8.0, H100=9.0) |
| 디스크 | - | **최소 100GB 여유** (env 12GB + 모델 70GB + 산출물) |

---

## 3. conda env 구축

### 3.1 소스 clone (⚠️ trellis2_src는 git 추적 대상이 아니다)

`trellis2_src/`는 6.5GB짜리 외부 소스라 **`.gitignore` 처리했다.** repo를 clone해도 딸려오지 않으므로 **반드시 아래 SHA로 별도 clone**한다. (API 변동성이 커서 SHA 고정이 필수 — `TRELLIS2_MIGRATION.md` §6 리스크 참조)

```bash
cd <repo>/t2o_pipeline
git clone --recursive https://github.com/microsoft/TRELLIS.2.git trellis2_src
cd trellis2_src && git checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd && cd ..
```

`--recursive` 필수 (서브모듈 eigen 포함).

### 3.2 env 생성

공식 `setup.sh`를 쓰되, **아래 함정 5개는 반드시 반영한다.** 전부 기존 서버에서 실제로 겪은 것들이다.

```bash
source <conda>/etc/profile.d/conda.sh
cd <repo>/t2o_pipeline/trellis2_src

# [함정 2] setup.sh의 `sudo apt install libjpeg-dev`는 컨테이너에 sudo가 없어 실패
apt-get install -y libjpeg-dev      # 선설치
sed -i 's/sudo //g' setup.sh        # setup.sh에서 sudo 제거

# flash-attn은 빼고 실행 (아래 [함정 1])
. ./setup.sh --new-env --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
conda activate trellis2   # 생성된 env 이름이 다르면 conda rename
```

#### 함정 1 — flash-attn 빌드 격리
공식 `setup.sh`의 flash-attn 라인에 `--no-build-isolation`이 **누락**돼 있어 `ModuleNotFoundError: No module named 'torch'`로 실패한다. 나머지 확장 설치 후 따로 깐다:
```bash
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=16 \
  pip install flash-attn==2.7.3 --no-build-isolation --no-deps
```
> `TORCH_CUDA_ARCH_LIST`는 GPU에 맞춰라 (A100=8.0, H100=9.0). 빌드에 GPU는 필요 없고 nvcc만 있으면 된다.

#### 함정 2 — sudo/libjpeg
위 스크립트에 반영됨.

#### 함정 3 — Pillow-SIMD 충돌
`--basic`이 `pillow-simd`(구버전 9.5 포크)를 깔아 표준 Pillow를 가로챈다. `PIL._webp`에 `HAVE_WEBPANIM`이 없어 imageio·webp 처리가 깨진다.
```bash
pip uninstall -y Pillow-SIMD pillow && pip install "pillow>=11"
```

#### 함정 4 — OpenCV가 EXR을 못 읽음
설치되는 `opencv-python-headless 5.0.0.93`은 `OpenEXR: NO`로 빌드돼 HDRI(.exr)를 못 읽는다(`cv2.imread`→None). 코드는 이미 `OpenEXR` 패키지로 우회하도록 작성돼 있으니 설치만 하면 된다:
```bash
pip install OpenEXR   # 3.4.x
```
> freeimage 3.16 백엔드는 이 EXR 압축을 못 읽으니 쓰지 말 것.

#### 함정 5 — transformers 버전 (⚠️ 가장 치명적)
`--basic`이 unpinned `transformers`를 깔아 5.x를 끌어온다. TRELLIS.2의 `image_feature_extractor.py`가 접근하는 `DINOv3ViTModel.layer`가 5.x엔 **없어서** `AttributeError`로 죽는다.
```bash
pip install transformers==4.57.6    # huggingface_hub도 0.36.2로 동반 하향됨
```

### 3.3 검증

```bash
conda activate trellis2
export PYTHONPATH=<repo>/t2o_pipeline/trellis2_src:$PYTHONPATH
python -c "import torch,o_voxel,flex_gemm,cumesh,nvdiffrast,flash_attn;print('CUDA',torch.cuda.is_available())"
```
`CUDA True`가 떠야 한다.

> **`RuntimeError: 0 active drivers ([])`가 뜨면** env 문제가 아니라 **GPU가 안 잡힌 것**이다. triton이 CUDA 드라이버를 못 찾은 것. 컨테이너라면 `--gpus all`로 띄웠는지 확인하고, 실행 중 컨테이너에서 갑자기 이 증상이 났다면 재시작 외엔 복구법이 없다 (`TRELLIS2_MIGRATION.md` §A.3).

### 3.4 T2I용 별도 env (`t2i`)

FLUX는 **별도 env에서 subprocess로 실행된다** (`trellis2_inference_core.py`의 `_T2I_PYTHON` 상수). 경로가 하드코딩돼 있으니 새 서버 경로가 다르면 **수정 필요**:

```python
_T2I_PYTHON = "/root/miniconda3/envs/t2i/bin/python"
_T2I_SCRIPT = "/data/previs_object/t2o_pipeline/previz_pipeline/text_to_image.py"
```

env 구성 (기존 서버 검증본):
```bash
conda create -n t2i python=3.10 -y && conda activate t2i
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.39.0 transformers==4.57.6 accelerate sentencepiece
```

---

## 4. 모델 다운로드 (~70GB)

`hf_models/`는 `.gitignore` 대상이다. 새 서버에서 직접 받는다.

```bash
export HF_HUB_DISABLE_XET=1   # ⚠️ FLUX.1-schnell 필수 (아래 참조)
huggingface-cli download microsoft/TRELLIS.2-4B      --local-dir hf_models/TRELLIS.2-4B       # 16GB
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir hf_models/FLUX.1-schnell # 54GB
```

- **`HF_HUB_DISABLE_XET=1` 없으면 schnell이 실패한다** — hf-xet 백엔드가 이 레포에서 `Unable to parse string as hex hash value` 에러를 낸다.
- 기존 서버에서 rsync로 받아오는 것도 가능하다 (다운로드가 더 빠를 수 있음).

### 게이트 레포 3개 — HF 계정 승인 필요
TRELLIS.2-4B의 `pipeline.json`이 런타임에 아래를 자동으로 당겨온다. **미승인 계정이면 403 GatedRepoError로 죽는다.**

| 레포 | 용도 | 라이선스 |
|---|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | image_cond_model | DINOv3 License (⚠️ 조건 확인 필요) |
| `briaai/RMBG-2.0` | rembg (배경제거) | **비상업** (⚠️ 아래 참조) |
| `black-forest-labs/FLUX.1-schnell` | T2I | Apache-2.0 ✅ |

기존 서버는 계정 `raengs`로 3개 모두 승인 완료. 새 서버에서 다른 계정을 쓴다면 각 레포에서 약관 수락이 필요하다. `huggingface-cli login`으로 토큰 설정.

> **라이선스 주의:** `briaai/RMBG-2.0`은 비상업 라이선스다. 사용자 결정으로 "일단 그대로 사용, 추후 재검토" 상태 (`TRELLIS2_MIGRATION.md` §C.7). 상업화 시 BRIA 상업 라이선스 계약 또는 rembg 모델 교체가 필요하다.

---

## 5. 파이프라인 실행

### 5.1 준비

```bash
source <conda>/etc/profile.d/conda.sh
conda activate trellis2
cd <repo>/t2o_pipeline

# ⚠️ 러너가 SCRIPT_DIR만 잡아줘서 이것 없으면 import trellis2 실패
export PYTHONPATH=<repo>/t2o_pipeline/trellis2_src:$PYTHONPATH
export TRELLIS_BASE_OUTPUT=<repo>/t2o_pipeline/t2o_results
```

> `run_usd_to_3D_object.sh`에는 **`conda activate`가 없다.** 직접 활성화해야 한다. (개선 여지: 스크립트에 activate 삽입 — `TRELLIS2_MIGRATION.md` §4.3의 미반영 항목)

### 5.2 실행

```bash
./run_usd_to_3D_object.sh \
  <repo>/movie_usd/hidden_time/scene_1/shot_1/objects/object_1.usda \
  <repo>/t2o_pipeline/t2o_results \
  --model <repo>/t2o_pipeline/hf_models/TRELLIS.2-4B \
  -- --backend trellis2 --max_items 1 --formats glb mp4 jpg
```

3단계로 돈다: USD 파싱 → (FLUX→TRELLIS.2) 추론 → GLB→USD 병합.
출력: `t2o_results/{model}/{YYYYMMDD}/run_{HHMMSS}_{flags}/`

### 5.3 반드시 지켜야 할 인자

| 인자 | 이유 |
|---|---|
| `--model .../TRELLIS.2-4B` | **생략하면 실패한다.** 러너 기본값이 v1 `microsoft/TRELLIS-text-base`인데, `json_parse_and_inference.py`의 자동 치환은 `TRELLIS-text-xlarge`일 때만 동작해서 base는 그대로 v2 로더로 넘어간다 |
| `-- --backend trellis2` | argparse 기본값이 이미 `trellis2`라 생략 가능하지만 명시 권장. `--` 뒤 인자는 2단계로 pass-through된다 |
| `--formats glb mp4 jpg` | 기본값의 `ply`는 v2에 gaussian이 없어 무의미 |
| `TRELLIS_BASE_OUTPUT` | 기본 경로 `/mnt/sdb_1TB/previz/text_to_3d`, `/mnt/nas/tmp/nayeon` 둘 다 실존하지 않는다 |

### 5.4 기타 주의점

- **`--filter` 는 ollama가 필요하다.** (`--translate` 는 2026-08 제거됨 — USD 가 `en` 필드를 동봉하는 것으로 계약 확정.) 기본 실행은 USD `customData`의 `en` 필드를 프롬프트로 직접 쓴다. 새 서버에 ollama가 있으면 활성화해도 된다 (기본 모델 `gemma3:4b`).
- **소요 시간**: 객체 1개당 TRELLIS.2 추론 ~36s + GLB export ~42s ≈ 3분. 첫 실행은 FLUX/파이프라인 로딩까지 더 걸린다.
- **HDRI 렌더는 best-effort** — 실패해도 GLB 생성은 계속된다. HDRI는 `trellis2_src/assets/hdri/*.exr` (repo 포함, 기본 `forest.exr`).
- **GLB는 PNG 텍스처로 export된다** (`extension_webp=False`). 번들된 `usd_from_gltf`가 `EXT_texture_webp`를 지원하지 않기 때문이다. 이 결정은 E2E에서 검증됐다 (§C.5) — **바꾸지 말 것.**

---

## 6. 알려진 미검증 지점

- **FLUX.1-schnell 실행 이력 없음.** 다운로드는 완료됐지만 E2E 검증은 **FLUX.1-dev로만** 돌았다 (`previz_pipeline/e2e_test.py`에 dev 경로 하드코딩). schnell은 dev와 파라미터가 다르다 (`text_to_image.py` 상단 주석 참조 — guidance_scale, step 수 등).
  - 문제가 생기면 `--t2i_model_path .../FLUX.1-dev`로 폴백해 원인을 분리하라.
  - **단, 상업용 최종본은 schnell(Apache-2.0)이어야 한다.** dev는 비상업 라이선스라 기능검증용으로만 보유 중이다.
- **실제 movie_usd acceptance 미실행.** 이게 이번 작업의 본체다.

---

## 7. 작업 결과 되돌리기

**되돌릴 것 (git):**
- 수정한 코드 (`previz_pipeline/*.py`, `run_usd_to_3D_object.sh` 등)
- `TRELLIS2_MIGRATION.md`에 추가한 실행 결과 부록 (새로 발견한 함정 포함 — 다음 사람을 위해 꼭 기록)
- 갱신된 `environment-trellis2.yml` / `requirements-trellis2.lock` (새 서버에서 버전이 달라졌다면)
- 이 문서(`NEW_SERVER_SETUP.md`)의 수정사항

**절대 되돌리지 말 것:**
- conda env 바이너리 (`envs/trellis2`, `envs/t2i`) — 기존 서버에 이미 검증된 env가 있다. 덮어쓰면 손해만 본다.
- `hf_models/`, `trellis2_src/`, `t2o_results/` — 전부 `.gitignore` 대상이다.

> ⚠️ **`.gitignore`에 `*.md`가 있다.** 문서를 커밋하려면 `git add -f`를 쓰거나, `.gitignore`의 예외 목록(`!NEW_SERVER_SETUP.md` 등)에 파일명을 추가해야 한다. 이 때문에 `TRELLIS2_MIGRATION.md`가 오랫동안 git에 안 들어가 있었다.

---

## 부록. 기존 서버 환경 스냅샷 (재현 기준값)

```
Python 3.10.20
torch 2.6.0+cu124 / torchvision 0.21.0+cu124
transformers 4.57.6  ← 핀 필수
huggingface_hub 0.36.2
diffusers 0.39.0
flash-attn 2.7.3     ← --no-build-isolation
o_voxel 0.0.1 / flex_gemm 1.0.0 / cumesh 0.0.1
nvdiffrast 0.3.3 / nvdiffrec_render 0.0.0
pillow 12.3.0        ← Pillow-SIMD 제거 후
OpenEXR 3.4.13
opencv-python-headless 5.0.0.93 (OpenEXR:NO — EXR은 OpenEXR 패키지로 우회)
attn backend: flash_attn / spconv algo: native
빌드 arch: sm_80 (A100)
```

전체 목록은 `requirements-trellis2.lock`, `environment-trellis2.yml` 참조.
