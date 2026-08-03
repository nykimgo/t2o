---
name: trellis2-env-pins
description: TRELLIS.2 conda env (trellis2) needs opencv 4.x and transformers 4.56.x pinned — setup.sh leaves them unpinned and latest versions break it
metadata: 
  node_type: memory
  type: project
  originSessionId: f573150c-78cd-4923-a0d2-eec1d086e4c9
  modified: 2026-07-24T00:48:34.899Z
---

`/home/sr/TRELLIS.2` 의 conda 환경 `trellis2` (python 3.10, torch 2.6.0+cu124, CUDA_HOME=/usr/local/cuda-12.4, TORCH_CUDA_ARCH_LIST=8.9). 2026-07-21 구축.

setup.sh 가 버전을 고정하지 않는 두 패키지를 반드시 고정해야 한다:
- `opencv-python-headless==4.11.0.86` — OpenCV 5.x 는 `assets/hdri/*.exr` (DWAB 압축 OpenEXR) 를 읽지 못하고 `cv2.imread` 가 None 을 반환한다.
- `transformers==4.56.2` — transformers 5.x 의 `DINOv3ViTModel` 은 `.layer` 속성이 없어 `trellis2/modules/image_feature_extractor.py:86` 에서 AttributeError.

**Why:** 두 오류 모두 설치 시점이 아니라 추론 실행 중에야 드러나서 원인 추적에 시간이 걸린다.

**How to apply:** 이 환경을 재구축하거나 pip 업그레이드를 할 때 위 두 핀을 유지할 것. flash-attn 은 소스 빌드 대신 GitHub 릴리스의 `cu12torch2.6cxx11abiFALSE-cp310` prebuilt wheel 을 쓰면 빠르다. setup.sh 의 `pillow-simd` 단계는 sudo apt 가 필요해 건너뛰었고, 기본 pillow 로 문제 없다.

## 재구축 가이드 (권위 문서)
env 를 처음부터 다시 만들 때는 `t2o_pipeline/ENV_REBUILD_GUIDE.md` (2026-07-24 작성, 이 4090 서버 검증본)를 따라라. 라이브 env 실측치 기반이고, 각 핀마다 "틀렸을 때 나는 증상"까지 적혀 있다. 같은 폴더의 `NEW_SERVER_SETUP.md` 는 A100 구서버 기준이라 transformers 4.57.6 / arch 8.0 / opencv 5 / 별도 t2i env / usd_from_gltf 등이 **틀리다** — 배경 참고로만.

## 2026-07-23: FLUX(t2i) 를 이 env 로 통합 — 별도 t2i env 불필요
`previs_proj` 의 T2I(FLUX.1-schnell)를 위해 따로 두던 `t2i` env 는 이제 필요 없다. **FLUX 는 trellis2 의 transformers 4.56.2 에서 정상 동작한다** — 클론 env 로 로컬 `hf_models/FLUX.1-schnell` 1024² 이미지 생성까지 실측 검증(clone→검증→본 env 병합). diffusers 0.39.0 의 transformers 하한이 4.41.2 라 4.56.2 로 충분. 두 env 를 나눴던 원래 근거("trellis2=transformers 5.x vs FLUX=<5")는 무효 — trellis2 가 5.x 가 아니라 4.56.2 로 핀됐기 때문. 코드 주석 `text_to_image.py:22`, `trellis2_inference_core.py:158,165` 는 이 낡은 전제라 실제와 어긋난다.

trellis2 에 `--no-deps` 로 추가한 6개 (핀 보호): `diffusers==0.39.0 accelerate==1.14.0 sentencepiece==0.2.2 importlib_metadata==9.0.0 psutil==7.2.2 zipp==4.1.0`.

**subprocess 경계는 유지한다** — 이유가 transformers 버전이 아니라 **VRAM** 이다. schnell bf16 이 4090 에서 피크 ~23.4GB → TRELLIS.2 가 이미 GPU 에 상주하면 같은 카드에서 OOM. FLUX 를 별도 프로세스로 띄워 PNG 뽑고 종료하며 VRAM 반납(또는 `T2I_GPU` 로 다른 카드). `object_generate.sh` 는 stage 2 직전 `export T2I_PYTHON="${T2I_PYTHON:-$(command -v python)}"` 로 그 subprocess 가 활성 trellis2 python 을 쓰게 배선함. `t2i` env 는 폴백으로 아직 남겨둠(삭제해도 무방).
