# TRELLIS.2 환경 재구축 가이드

> **2026-09-08 변경:** 기본 T2I가 ERNIE-Image-Turbo로 전환되었고 `trellis2` 환경에 통합했다. 현재 설치는 [ERNIE_DEFAULT_SETUP.md](docs/ERNIE_DEFAULT_SETUP.md)를 먼저 적용한다. Transformers **5.16.1** 및 TRELLIS 호환 패치 두 개가 필수다. 아래 본문은 기존 FLUX/Transformers 4.56.2 구성의 기록이며, 그중 5.x 금지·T2I 모델/의존성 설정은 새 문서로 대체된다.

> 작성: 2026-07-24 / 대상: conda env `trellis2` 를 **이 서버(RTX 4090 ×4, sm_89)** 또는
> 동급 서버에 처음부터 다시 만드는 사람(또는 Claude).
> 이 문서는 **실제로 동작 중인 env 에서 값을 뽑아** 작성했다. 추정값이 아니다.
>
> ⚠️ **`NEW_SERVER_SETUP.md` 는 이 문서보다 오래됐고 A100 구서버 기준이라 여러 값이 틀리다.**
> (transformers 4.57.6 / arch 8.0 / opencv 5 / 별도 t2i env / usd_from_gltf 등)
> 재구축할 때는 **이 문서를 우선**하고, `NEW_SERVER_SETUP.md` 는 배경 참고로만 봐라.
> 어긋나는 지점은 이 문서 §9 에 총정리해 뒀다.

---

## 0. 핵심 원칙 — 왜 이 문서가 필요한가

이 env 의 실패는 대부분 **설치 시점이 아니라 추론 실행 중에, 그것도 조용히** 드러난다.
버전 하나 어긋나면 에러가 아니라 "결과가 조용히 줄거나 렌더가 skip" 되는 식이라
"성공"으로 오인하기 쉽다. 그래서 이 문서는 각 핀마다 **틀렸을 때 나는 증상**을 같이 적었다.
값을 바꿀 때는 반드시 증상 컬럼을 먼저 읽어라.

핵심 두 가지:
1. **핀을 지켜라.** transformers / opencv 두 개는 특히 치명적이다 (§5).
2. **FLUX 는 이제 별도 env 가 아니다.** trellis2 하나로 통합됐다 (§6). 구서버 문서의 `t2i` env 설명은 폐기됐다.

---

## 1. 기준 환경 스냅샷 (재현 목표값)

동작 검증된 `trellis2` env 의 실제 값이다. 재구축 후 이 표와 대조해라.

| 항목 | 값 | 비고 |
|---|---|---|
| Python | 3.10.20 | |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 | CUDA 12.4 필수 |
| CUDA toolkit | 12.4 (`/usr/local/cuda-12.4`) | `CUDA_HOME` 이 여기를 가리켜야 확장 빌드됨 |
| GPU | RTX 4090, capability **(8, 9) = sm_89** | **빌드 arch = 8.9** |
| **transformers** | **4.56.2** | 🔴 핀. 틀리면 §5 참조 |
| huggingface-hub | 0.36.2 | transformers 4.56.2 와 동반 |
| tokenizers / safetensors | 0.22.2 / 0.8.0 | |
| **opencv-python-headless** | **4.11.0.86** | 🔴 핀. 5.x 는 EXR 을 못 읽음 (§5) |
| OpenEXR / (Imath) | 3.4.13 | HDRI 읽기 경로 (§7) |
| flash-attn | 2.7.3 | prebuilt wheel 권장 (§4) |
| diffusers | 0.39.0 | FLUX 용 (§6) |
| accelerate | 1.14.0 | FLUX CPU offload 용 (§6) |
| sentencepiece | 0.2.2 | FLUX T5 토크나이저 (§6) |
| pillow | 12.2.0 | 표준 pillow. **Pillow-SIMD 쓰지 마라** (§4) |
| numpy / protobuf | 2.2.6 / 7.35.1 | |
| o_voxel / flex_gemm / cumesh | 0.0.1 / 1.0.0 / 0.0.1 | 커스텀 CUDA 확장, arch 8.9 로 빌드 |
| triton | 3.2.0 | |
| usd-core | 25.8 | 🟠 `--no-deps` 로 추가 (§7) |
| trimesh | 4.12.2 | GLB→USD 네이티브 변환기 (§6, `glb_to_usd_native.py`) |
| utils3d | 0.0.2 | |

전체 목록: `requirements-trellis2.lock`, `environment-trellis2.yml`.
**단, lock 파일은 A100 구서버 시점 값이라 transformers/opencv/arch 가 이 문서와 다를 수 있다.
충돌하면 이 문서가 정답이다.**

---

## 2. 사전 요구

| 항목 | 조건 |
|---|---|
| OS | Ubuntu 22.04 계열 권장 |
| CUDA toolkit | **12.4 필수** (torch 2.6.0+cu124 와 맞춤). `nvcc --version` 으로 확인 |
| GPU | VRAM **24GB 이상**. arch 가 4090(8.9)과 다르면 §4 의 `TORCH_CUDA_ARCH_LIST` 를 바꿔라 (A100=8.0, H100=9.0) |
| 디스크 | **100GB+** 여유 (env ~12GB + 모델 ~70GB + 산출물) |
| 빌드 | nvcc 만 있으면 확장 빌드 가능. 빌드 자체엔 GPU 불필요(단 실행엔 필요) |

`CUDA_HOME` 을 반드시 잡아라. 안 잡으면 확장이 엉뚱한 CUDA 로 빌드되거나 실패한다:
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
```

---

## 3. TRELLIS.2 소스 clone (SHA 고정 필수)

`trellis2_src/` 는 6.5GB 외부 소스라 `.gitignore` 대상이다. repo 를 clone 해도 안 딸려온다.
API 변동성이 커서 **SHA 고정이 필수**다.

```bash
cd <repo>/t2o_pipeline
git clone --recursive https://github.com/microsoft/TRELLIS.2.git trellis2_src
cd trellis2_src && git checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd && cd ..
```

`--recursive` 필수 (서브모듈 eigen 포함). 이 서버의 `/home/sr/TRELLIS.2` 가 이 SHA 와 일치한다.

---

## 4. conda env 생성 + 커스텀 확장 빌드

공식 `setup.sh` 를 쓰되 **아래 함정 5개를 반드시 반영**한다. 전부 실제로 겪은 것들이다.

```bash
source <conda>/etc/profile.d/conda.sh
export CUDA_HOME=/usr/local/cuda-12.4
cd <repo>/t2o_pipeline/trellis2_src

# [함정 2] setup.sh 의 `sudo apt install ...` 는 sudo 없는 환경에서 실패
apt-get install -y libjpeg-dev      # 가능하면 선설치
sed -i 's/sudo //g' setup.sh        # setup.sh 에서 sudo 제거

# flash-attn 은 빼고 실행 (아래 [함정 1]); GLB 생성에 필요한 확장만 설치
. ./setup.sh --new-env --basic --cumesh --o-voxel --flexgemm
conda activate trellis2             # 이름이 다르면 conda rename

# [함정 6] o_voxel 상업 배포용 최종 파일 적용 — 아래 참고
cd <repo>/t2o_pipeline && ./patches/install_o_voxel_commercial.sh
```

### 함정 1 — flash-attn: 소스 빌드 대신 prebuilt wheel
공식 setup.sh 의 flash-attn 라인은 `--no-build-isolation` 누락으로 `No module named 'torch'` 로 죽는다.
소스 빌드는 오래 걸리니 **GitHub 릴리스의 prebuilt wheel** 을 써라.
`github.com/Dao-AILab/flash-attention/releases` 의 `v2.7.3` 에서 **태그가 정확히 맞는** wheel 을 골라라:
`cu12` + `torch2.6` + `cxx11abiFALSE` + `cp310`(python 3.10). 파일명 예:
```
flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
(정확한 파일명은 릴리스 페이지에서 확인 — 태그 하나라도 어긋나면 import 시 심볼 에러가 난다)
```bash
pip install --no-deps <위 wheel URL 또는 내려받은 파일 경로>
```
> 굳이 소스로 빌드해야 한다면: `TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16 pip install flash-attn==2.7.3 --no-build-isolation --no-deps`
> **arch 는 8.9 (4090)** 다. 구서버 문서의 8.0(A100)이 아니다.

### 함정 2 — sudo/libjpeg
위 스크립트에 반영됨.

### 함정 3 — Pillow-SIMD 쓰지 마라
`--basic` 이 `pillow-simd`(구버전 9.5 포크)를 깔면 표준 Pillow 를 가로채 `PIL._webp` 가 깨진다.
이 서버는 **pillow-simd 단계를 건너뛰고 표준 pillow 12.x 로 문제없이 돌았다.**
setup.sh 가 pillow-simd 를 깔았다면 되돌려라:
```bash
pip uninstall -y pillow-simd pillow && pip install "pillow>=11"
```

### 함정 4 — OpenCV 는 4.x 로 핀 (5.x 금지)
`--basic` 이 unpinned opencv 를 깔면 5.x 가 들어오는데, **5.x 는 DWAB 압축 OpenEXR(`assets/hdri/*.exr`)
을 못 읽어 `cv2.imread` 가 조용히 `None` 을 반환**한다. 4.x 로 핀해라:
```bash
pip install --no-deps opencv-python-headless==4.11.0.86
```

### 함정 5 — transformers 는 4.56.2 (5.x 금지)
`--basic` 이 unpinned transformers 를 깔면 5.x 가 들어오는데, 5.x 의 `DINOv3ViTModel` 은
`.layer` 속성이 없어 `trellis2/modules/image_feature_extractor.py` 에서 `AttributeError` 로 죽는다.
**이 서버의 검증값은 4.56.2** 다 (구서버 문서의 4.57.6 이 아니다 — §9 참조):
```bash
pip install transformers==4.56.2    # huggingface-hub 0.36.2 동반
```

### 함정 6 — o_voxel 상업 배포용 최종 파일 적용
검증·배포 경로는 `patches/o_voxel/`에 보관된 순수 torch UV 래스터라이저와
최종 `postprocess.py`를 사용한다. 환경이나 `o_voxel`을 다시 설치하면 아래 스크립트를
다시 실행한다.

```bash
cd <repo>/t2o_pipeline
./patches/install_o_voxel_commercial.sh
```

`trellis2_src/` 가 `.gitignore` 대상이라 수정분이 버전관리에 안 남는다. 그래서 패치를
완료한 최종 파일을 `patches/` 에 두고 스크립트로 설치본과 벤더본에 복사한다.

`trellis2_inference_core.load_pipeline()`은 로드 시점에 적용 여부를 검사한다.
적용되지 않은 환경에서는 자동으로 옛 파일을 사용하지 않고 RuntimeError로 중단한다.

대체 구현은 순수 torch(의존성 0 추가)이며 실제 TRELLIS.2 출력으로 검증했다:
지오메트리에서 구워지는 텍셀 99.7%+ 비트동일, PSNR 69~85 dB, 불일치 지점은 fp64 재계산 결과
대체 구현이 더 정확하다. 근거·재현 방법은 `experiments/uv_raster_swap/FINDINGS.md`.

> 배포 파이프라인에서는 프리뷰 렌더 경로를 제거했으며 GLB/USD 출력만 지원한다.

---

## 5. 🔴 절대 어기면 안 되는 핀 (요약)

| 패키지 | 핀 | 틀리면 나는 증상 |
|---|---|---|
| **transformers** | `4.56.2` | 5.x → `DINOv3ViTModel.layer` AttributeError (추론 중 죽음). 4.56.2 는 FLUX 도 정상 구동(diffusers 하한 4.41.2 만족) |
| **opencv-python-headless** | `4.11.0.86` | 5.x → HDRI EXR `cv2.imread` 가 `None`. 렌더가 조용히 깨짐 |

이 둘은 pip 업그레이드나 다른 패키지 설치 시 **의존성으로 딸려 올라갈 수 있다.**
그래서 아래 §6, §7 의 추가 설치는 전부 `--no-deps` 로 한다.

---

## 6. FLUX 통합 — 별도 env 없음 (단일 env)

> ⚠️ **구서버 문서(`NEW_SERVER_SETUP.md` §3.4)의 별도 `t2i` env 는 폐기됐다.**
> FLUX.1-schnell 은 trellis2 의 transformers 4.56.2 에서 정상 동작함이 **실측 검증**됐다
> (로컬 `hf_models/FLUX.1-schnell` 로 1024² 이미지 생성 확인, 2026-07-24).
> 두 env 를 나눴던 원래 근거("trellis2=transformers 5.x vs FLUX=<5")는 무효 —
> trellis2 가 5.x 가 아니라 4.56.2 로 핀됐기 때문이다.

### 6.1 trellis2 에 6개 패키지 추가 (`--no-deps` 로 핀 보호)
```bash
pip install --no-deps \
  diffusers==0.39.0 accelerate==1.14.0 sentencepiece==0.2.2 \
  importlib_metadata==9.0.0 psutil==7.2.2 zipp==4.1.0
```
설치 후 반드시 핀 무결성 확인 — `transformers 4.56.2` / `opencv 4.11.0.86` 가 그대로여야 한다.

### 6.2 FLUX subprocess 를 trellis2 python 으로 배선
FLUX 를 **별도 프로세스로 띄우는 구조는 유지**한다. 이유가 transformers 버전이 아니라 **VRAM** 이다:
schnell bf16 이 4090 에서 피크 ~23.4GB → TRELLIS.2 가 이미 GPU 에 상주하면 같은 카드에서 OOM.
subprocess 가 PNG 뽑고 종료하며 VRAM 반납(또는 `T2I_GPU` 로 다른 카드).

`object_generate.sh` 는 stage 2 직전에 이미 배선돼 있다:
```bash
export T2I_PYTHON="${T2I_PYTHON:-$(command -v python)}"
```
즉 활성화된 trellis2 python 을 FLUX subprocess 가 그대로 쓴다.
`trellis2_inference_core.py` 의 `_T2I_PYTHON` 기본값(sibling `t2i` env 경로)은 이 env 변수로 오버라이드된다.
**별도 t2i env 를 만들지 마라.**

### 6.3 GLB→USD 는 네이티브 변환기 (usd_from_gltf 불필요)
구서버 문서의 `usd_from_gltf` 바이너리 빌드는 **더 이상 필요 없다.**
`previz_pipeline/glb_to_usd_native.py` (trimesh + pxr 순수 파이썬)가 대체했다.
외부 바이너리 0개. `merge_glb_to_usd.py` 가 이 모듈에 위임한다.

---

## 7. 🟠 없으면 결과가 "조용히" 줄어드는 --no-deps 패키지

에러가 아니라 무증상 결과 축소로 나타나서 제일 위험하다.

```bash
pip install --no-deps usd-core==25.8
pip install --no-deps OpenEXR        # 3.4.x
```

| 패키지 | 없으면 | 증상 (에러 안 남) |
|---|---|---|
| `usd-core` (pxr) | `usd_parser.py` 가 regex 폴백으로 떨어짐 | 파싱 항목이 **11개 → 3개로 조용히 감소** (canonical 3 + shot override 8 이 사라짐). `[DEBUG USD]` 로그도 사라짐 |
| `OpenEXR` | HDRI envmap 로드 실패 | **렌더가 조용히 skip** (GLB 생성은 계속되므로 "성공"으로 오인) |

> `object_generate.sh` 는 시작 시 `import pxr` 체크로 usd-core 누락을 조기에 잡는다.
> 그래도 재구축 직후엔 **원 서버 로그와 파싱 항목 수를 직접 대조**해서 검증해라.

---

## 8. 모델 다운로드 (~70GB)

`hf_models/` 는 `.gitignore` 대상이다.

```bash
export HF_HUB_DISABLE_XET=1   # ⚠️ FLUX.1-schnell 필수 (없으면 hex hash 파싱 에러로 실패)
hf download microsoft/TRELLIS.2-4B          --local-dir hf_models/TRELLIS.2-4B        # ~16GB
hf download black-forest-labs/FLUX.1-schnell --local-dir hf_models/FLUX.1-schnell     # ~54GB
```
> `hf download` 를 써라. `huggingface-cli` 는 신버전 hub 에서 폐기 경로다(이 env 0.36.2 엔 아직 있지만 `hf` 로 통일).

### 게이트 레포 3개 — HF 계정 승인 필요
TRELLIS.2-4B 의 `pipeline.json` 이 런타임에 자동으로 당겨온다. 미승인이면 **403 GatedRepoError** 로 죽는다.

| 레포 | 용도 | 라이선스 |
|---|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | image_cond_model | DINOv3 License |
| `briaai/RMBG-2.0` | rembg(배경제거) | **비상업** ⚠️ 상업화 시 교체/계약 필요 |
| `black-forest-labs/FLUX.1-schnell` | T2I | Apache-2.0 ✅ |

이 서버는 HF 계정 **`raengs`** 로 3개 모두 승인 완료. 다른 계정이면 각 레포에서 약관 수락 + `hf auth login`.

---

## 9. 재구축 후 검증 체크리스트

순서대로 통과해야 "됐다"고 말할 수 있다.

```bash
conda activate trellis2
export CUDA_HOME=/usr/local/cuda-12.4
export PYTHONPATH=<repo>/t2o_pipeline/trellis2_src:$PYTHONPATH

# (1) 확장 + CUDA
python -c "import torch,o_voxel,flex_gemm,cumesh,flash_attn; \
  print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
#   → CUDA True (8, 9) 여야 함. capability 가 (8,9) 가 아니면 arch 빌드가 틀린 것.

# (2) 두 모델이 한 인터프리터에서 공존 (단일 env 성립 확인)
python -c "from transformers import DINOv3ViTModel; from diffusers import FluxPipeline; \
  import transformers; print('transformers', transformers.__version__)"
#   → 4.56.2, import 에러 없어야 함.

# (3) FLUX 실제 1장 생성 (transformers 4.56.2 에서 도는지 실측)
CUDA_VISIBLE_DEVICES=0 python previz_pipeline/text_to_image.py \
  --prompt "a single wooden treasure chest, centered, plain background" \
  --out /tmp/flux_check.png --seed 42 \
  --model hf_models/FLUX.1-schnell
#   → 1024x1024 PNG 생성되면 OK.
```

- (4) **핀 무결성**: `pip list | grep -iE 'transformers|opencv'` → `4.56.2` / `4.11.0.86`.
- (5) **파싱 항목 수 대조**: 실제 movie_usd 1건을 돌려 파싱 항목 수를 원 서버 로그와 맞춰라 (§7 usd-core 무증상 축소 방지).

---

## 10. 구서버 문서(`NEW_SERVER_SETUP.md`)와 어긋나는 지점 총정리

이 서버에서 재구축할 때 **구서버 문서를 그대로 따르면 반복하게 되는 실수들**이다.

| 항목 | 구서버 문서 (A100) | 이 서버 정답 (4090) |
|---|---|---|
| transformers | 4.57.6 | **4.56.2** (검증값) |
| GPU arch | 8.0 (A100) | **8.9** (4090, sm_89) |
| opencv | 5.0.0.93 + OpenEXR 우회 | **4.11.0.86 로 핀** + OpenEXR 병행 |
| FLUX env | **별도 `t2i` env** (transformers 4.57.6) | **trellis2 로 통합** (별도 env 없음) |
| GLB→USD | `usd_from_gltf` 바이너리 빌드 | **`glb_to_usd_native.py`** (바이너리 불필요) |
| flash-attn | 소스 빌드 | **prebuilt wheel** (arch 8.9) |
| HF CLI | `huggingface-cli download` | **`hf download`** |
| FLUX.1-schnell 검증 | "미검증(dev로만)" | **schnell 실측 검증 완료** |

---

## 부록. 되돌리기 원칙

- **git 으로 관리**: `previz_pipeline/*.py`, `object_generate.sh`(주의: previs_proj 루트라 t2o repo 밖),
  이 문서, `TRELLIS2_MIGRATION.md`.
- **절대 복사/되돌리지 말 것**: conda env 바이너리, `hf_models/`, `trellis2_src/`, `t2o_results/` (전부 gitignore).
- ⚠️ `.gitignore` 에 `*.md` 가 있어 문서 커밋은 `git add -f` 가 필요할 수 있다.
