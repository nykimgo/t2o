# A100 서버 인수인계 — text→image→3D 환경 셋팅 (+ 프롬프트 최적화 맥락)

> **독자**: A100 공유 도커 서버에서 이 프로젝트를 이어받는 Claude(+동료).
> **목적**: 4090 개발서버에서 검증된 `text→image→3D` 환경을 A100 공유 서버에 재현해
> 동료들이 같이 쓸 기준 환경을 만든다.
> **우선순위**: ① t2o_pipeline(FLUX+TRELLIS.2) 환경 셋팅이 가장 급함(동료 공유). ② prompt_lab
> 최적화는 그 다음. ③ ComfyUI는 **최종 산출물용**이라 지금 우선순위 아님.
> **작성 시점**: 2026-07 (4090 dev 서버 세션 산출). 실험 상세는 `prompt_lab/docs/EXPERIMENT_LOG.md`.

---

## 0. 30초 요약 (TL;DR)

- 파이프라인: `text → image(FLUX.1-schnell) → 3D(TRELLIS.2, image-to-3D)` + auto-rigging(나중).
- **이 문서는 이 4090 개발서버에서 실제로 검증된 env 를 기준으로 쓴다.** 그대로 재현하는 게 목표.
- 환경 핀(§3.2): torch 2.6.0+cu124, diffusers 0.39.0, transformers 4.56.2, opencv-python-headless 4.11.0.86 등 — **라이브 env 실측치.**
- **GPU 가 이 서버(4090, 24GB, sm_89)와 다르면** 확인·판단할 지점이 있다(§2): 컴파일 확장의 arch,
  VRAM 에 따른 offload/상주, 드라이버 정합. 이건 **그쪽 서버 Claude 가 실측으로 확인**할 것(여기선 단정하지 않는다).
- 이번 세션 산출물: **T2I 시스템 프롬프트 2개 확정**(무생물 / 생명체). §7.

---

## 1. 프로젝트 맥락 (무엇을·왜)

**Previs / t2o**: 스토리보드 → 3D 오브젝트 자동 생성 파이프라인.
```
스토리보드 분석(동료 VLM) → 구조화 필드(object/base_description/appearance) + USD
        │
        ▼  text → image      FLUX.1-schnell (t2i)
        ▼  image → 3D        TRELLIS.2-4B (image-to-3D)
        ▼  → GLB → USD merge
        ▼  (나중) auto-rigging
```
- 3D가 **image-to-3D**라 품질의 실질 손잡이는 **t2i 프롬프트**다. 그래서 별도 `prompt_lab`이 t2i 프롬프트를
  최적화한다(§7·§8).
- 진입점: `t2o_pipeline/object_generate.sh` (USD 파싱 → 번역/필터(ollama) → FLUX(별도 프로세스) → TRELLIS.2 → GLB/USD).
- repo: github `nykimgo/t2o`, 브랜치 `feature/trellis2`. 이 dev 서버 경로 `/home/sr/previs_proj/{t2o_pipeline, prompt_lab}`.

---

## 2. GPU 가 이 서버(4090)와 다르면 확인·판단할 지점

이 env 는 **RTX 4090(24GB, sm_89)** 에서 검증됐다. 다른 GPU 에 셋팅할 때 아래는 하드웨어에 따라 달라질 수 있으니,
**맞다/틀리다 단정 대신 그쪽에서 실측으로 확인하고 판단**할 것.

- **컴파일 확장 arch**: flash-attn / o_voxel / flex_gemm / cumesh 는 이 서버에서 **sm_89**로 빌드됨.
  arch-specific 이라 GPU 가 다르면 해당 arch(`TORCH_CUDA_ARCH_LIST`)로 재빌드가 필요할 수 있다. → 확인.
- **VRAM 에 따른 offload/상주**: FLUX schnell bf16 은 ~30-34GB. `text_to_image.py` 가 VRAM 을 감지해
  **≥40GB 면 상주, 미만이면 CPU offload** 로 자동 분기한다(`T2I_OFFLOAD=0/1` 강제 가능). 24GB(4090)에선
  offload 로 장당 25-41초였다. VRAM 이 크면 상주로 빨라진다(코드가 자동 처리). → 그쪽 VRAM 에서 확인.
- **fp8/양자화**: 이 서버에선 4090 24GB 상주용으로 fp8 을 검토했으나 **미완**(ComfyUI 용 Kijai fp8 파일만 받아둠,
  `hf_models/flux-fp8`). VRAM 이 크면 bf16 상주로 충분해 불필요할 수 있다. → 확인.
- **드라이버 정합**: 이 서버는 세션 중 `nvidia-smi` "Driver/library version mismatch" 가 발생해 상주 `.to("cuda")` 가
  assert 로 죽은 적이 있다(offload 는 우회됨). 새 서버/도커에선 처음에 **`nvidia-smi` 정상 출력 + 기본 CUDA 할당**을 확인할 것.

---

## 3. 환경 구축

### 3.1 참고 문서
- **`t2o_pipeline/ENV_REBUILD_GUIDE.md`** — 이 4090 서버 검증본. 각 핀의 "틀렸을 때 증상"까지 있음. 핀의 1차 근거.
- `t2o_pipeline/NEW_SERVER_SETUP.md` — 파이프라인 실행/모델 다운로드/게이트 승인 등 절차 참고.
- 아래 **§3.2 는 이 서버 라이브 env 실측치**다. 이걸 재현 기준으로 삼되, GPU 의존 부분(§2)은 그쪽에서 확인.

### 3.2 검증된 핀 (이 4090 서버 라이브 env 실측)
`conda env trellis2` (python 3.10, CUDA_HOME=/usr/local/cuda-12.4):
```
torch 2.6.0+cu124   diffusers 0.39.0   transformers 4.56.2   numpy 2.2.6
accelerate 1.14.0   sentencepiece 0.2.2   einops 0.8.2   safetensors 0.8.0
opencv-python-headless 4.11.0.86   trimesh 4.12.2   imageio 2.37.4   pxr(usd-core)
```
반드시 추가(없으면 **조용한 결과 축소** — 무증상이라 위험):
- **`usd-core`** — 없으면 `usd_parser.py`가 에러 없이 regex 폴백 → 파싱 항목이 11개(canonical 3 + shot override 8)에서
  3개로 조용히 줆, `[DEBUG USD]` 로그 사라짐. **항목 수를 원 서버와 대조 검증할 것.**
- **`OpenEXR`** — 없으면 HDRI envmap 로드 실패 → 렌더 조용히 skip(GLB 생성은 계속).

### 3.3 컴파일 확장 (arch 의존 — 그쪽 GPU 로 확인)
`o_voxel / flex_gemm / cumesh`(via `t2o_pipeline/trellis2_src/setup.sh`)와 flash-attn 은 이 서버에서 **sm_89**로 빌드됨.
GPU 가 다르면 해당 arch(`TORCH_CUDA_ARCH_LIST`)로 재빌드가 필요할 수 있으니 확인·판단할 것.
- flash-attn: 소스 빌드 대신 GitHub 릴리스 prebuilt wheel(torch2.6 / cu12 / cp310, 해당 arch 호환) 권장.
- ⚠️ `trellis2_src/`는 git 추적 대상 아님 — 별도 clone 필요(SHA 는 NEW_SERVER_SETUP §3.1 참조).

### 3.4 HuggingFace
- `huggingface-cli`는 폐기됨 → **`hf download`** 사용.
- 게이트 레포 승인 필요: **FLUX.1-schnell / RMBG-2.0 / dinov3-vitl16** (dev 서버는 HF 계정 `raengs`로 승인됨).

---

## 4. 모델 가중치 (~70GB)

| 모델 | 크기 | 위치(dev) | 비고 |
|---|---|---|---|
| FLUX.1-schnell | 54GB | `t2o_pipeline/hf_models/FLUX.1-schnell` | Apache-2.0. ⚠️ **dev는 비상업 금지** — schnell만 |
| TRELLIS.2-4B | 16GB | `t2o_pipeline/hf_models/TRELLIS.2-4B` | image-to-3D |
| flux-fp8(Kijai) | 12GB | `t2o_pipeline/hf_models/flux-fp8` | 4090 24GB 상주/ComfyUI 용. VRAM 큰 GPU 면 불필요할 수 있음(확인) |

FLUX schnell은 `xet` 백엔드에서 해시 파싱 에러가 날 수 있음 — `hf download`로 받되 문제 시 `HF_HUB_DISABLE_XET=1`.

---

## 5. ollama (번역/필터 + prompt_lab LLM)
- `object_generate.sh --translate`/필터가 ollama 사용. prompt_lab auto 루프도 사용.
- 모델: **gpt-oss:20b**. A100서: ollama 설치 + `ollama pull gpt-oss:20b`.
- ⚠️ gpt-oss는 reasoning 모델이라 콜당 느림(~99s@4090). prompt_lab에서 쓸 땐 `llm.params.max_tokens`를 넉넉히(4096) —
  기본 1024면 추론 토큰이 예산 먹어 JSON이 잘림.

---

## 6. 파이프라인 실행 & 검증

```bash
conda activate trellis2
export PYTHONPATH=/path/to/t2o_pipeline/trellis2_src:/path/to/t2o_pipeline/previz_pipeline
cd /path/to/t2o_pipeline
# 검증 실행 (dev 서버에서 end-to-end 통과 확인된 명령)
T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda
```
검증 포인트:
- GLB가 `{object_n.usda 디렉토리}/assets/{object_name}/*.glb`에 생성되는지.
- **usd 파싱 항목 수 대조**(usd-core 있으면 canonical 3 + shot override 8 = 11 처리; 없으면 3으로 조용히 축소).
- VRAM 이 크면 FLUX 가 상주로 돌아 이 서버(offload, 장당 25-41s)보다 빠름(코드가 자동 분기 §2).

prompt_lab 회귀(선택): `conda run -n trellis2 python -m pytest -q` (prompt_lab 디렉터리) → **184 passed, 1 skipped** 기대.

---

## 7. ★ 이번 세션 산출물 — 확정 T2I 시스템 프롬프트 2개

`prompt_lab` 최적화로 도출·확정(상세·근거는 `prompt_lab/docs/EXPERIMENT_LOG.md`). **category로 라우팅.**

### ① 무생물 (ablation+LOO 검증 — 신뢰도 높음)
```
{base_description}, {appearance}, {object}, single centered object, neutral background, full object visible in frame, unoccluded
```

### ② 생명체 — LLM 빌더 1콜(분류 + 고정 자세 스캐폴딩 삽입)
LLM이 body-plan을 `biped|quadruped|bird`로 분류 → 아래 고정 문구를 **verbatim 삽입**(창작 금지). 필드는 **object만**.
```
{object}, <형태별 스캐폴딩>
  biped:     in a symmetric A-pose, arms angled slightly down and away from the body,
             legs straight and slightly apart, front view, neutral background, single object, full body
  quadruped: standing naturally on all four legs, all four legs clearly separated and extended,
             not tucked, side view, neutral background, single object, full body in frame
  bird:      with wings fully spread symmetrically, standing, front view, neutral background,
             single object, full body in frame
```
- 자세 기준은 리깅 요구(Tripo rig-type 참고 — 실제 리거로 Tripo를 쓰는 건 아님)에서 나옴.
- **신뢰도 주의**: ①무생물 = 검증됨. ②생명체 자세 3종 = **짧은 테스트로 "형태별 > 단일자세" 확인한 잠정값**
  (단일 A-pose는 말=뒷발서기·개구리=의인화로 붕괴). 정련하려면 클래스별 **사람 O/X** 필요(VQA는 자세품질 판단 불가).
- ⚠️ 프롬프트는 **모델·정밀도별**로 전이 안 됨(CLAUDE.md §9). 위 값은 **FLUX schnell bf16** 기준.
  같은 모델·bf16 이면 유효하고, 정밀도(fp8 등)/모델이 바뀌면 재검증 필요.

---

## 8. prompt_lab에서 이번에 구현·변경된 코드 (A100서 그대로 동작)

- **스텁 A(FLUX backend)** 구현: `previs_lab/adapters/image/diffusers_slots.py` — `FluxSchnellBackend._load/_infer`
  (`text_to_image.py` 미러링, `_POSITIVE_SUFFIX` 미포함, `max_sequence_length=256`). `fp8_transformer` param은 4090용(미완).
  로컬 FLUX 경로를 `image.params.model`에 주면 됨. VRAM≥40GB 면 자동 상주(§2).
- **VQAScore metric**: `vqa_align`(전체 프롬프트 정렬) + `vqa_query`(고정 질문=T-pose 준수). 별도 **`prompt_metrics` env**를
  subprocess 데몬으로 호출(`_vqa_scorer_daemon.py`). ★ **병렬 워커 안에서 데몬을 띄우면 BrokenPipe로 죽음** →
  **2단계 아키텍처**: Phase1 병렬 FLUX 생성+CV → Phase2 메인 프로세스 데몬으로 VQA 사후 채점.
- **ablation에 `isolation_text` 축 추가**(ablation.py + config.py). 회귀 테스트 갱신(184 passed).
- CV 게이트/집계/O/X/resume 등은 기존 그대로.

### prompt_metrics env (VQA 채점용 — prompt_lab 최적화 돌릴 때만 필요, 선택)
격리 env(trellis2 핀 보호). transformers 4.36.1 / diffusers 0.31.0 필요(t2v-metrics가 강제):
```
torch 2.6.0+cu124   transformers 4.36.1   diffusers 0.31.0
image-reward 1.5   t2v-metrics 1.2   clip(git+openai/CLIP)   open-clip-torch 2.32.0
```
설치 함정: image-reward가 OpenAI `clip` 패키지 요구(`pip install git+https://github.com/openai/CLIP.git`);
diffusers는 image-reward가 끌어오지만 채점엔 미사용(ReFL만) — 0.31.0이 transformers 4.36.1+hf_hub 0.36과 호환.
VQAScore VLM은 `clip-flant5-xl`(첫 실행 시 ~4GB 다운).

---

## 9. 이번 세션에서 밟은 함정 (재현 방지)

- **budget 기본값**: `budget.max_objects=200`(넘으면 조기 stop — ablation이 11/24셀만 돌고 멈춤), `max_llm_calls: 0`을
  주면 `0>=0`으로 **즉시 stop**. ablation/대량 실행 시 넉넉히(`max_objects: 1000, max_llm_calls: 100`).
- **VQA 데몬 + 병렬 워커 = BrokenPipe** → 2단계 아키텍처(§8).
- **ablation dedupe**: 동일 조립 텍스트는 공유(재생성 안 함). 원점수 전량 보존 → LOO는 **사후 재집계**(재생성 0).
- **VQA는 자세 품질 판단 불가**: 뒷발 선 말을 자연기립보다 높게 줬음. 생명체 자세는 사람 O/X로 채점.
- **NVML driver mismatch**(이 4090 서버): 상주 `.to("cuda")` assert. A100 도커선 드라이버 정합 먼저 확인(§2).
- **CV single_object 오탐**: 펼친 날개/사지를 multiple_objects로 오판 → 생명체는 `cv.thresholds.max_objects` 상향(soft-gate).

---

## 10. 포인터

- `prompt_lab/docs/EXPERIMENT_LOG.md` — 실험 전체 기록(방법/결과/해석/한계, §1~§9). **실험 맥락은 여기.**
- `t2o_pipeline/ENV_REBUILD_GUIDE.md` — 이 4090 서버 env 검증본(핀 근거).
- `t2o_pipeline/NEW_SERVER_SETUP.md` — 파이프라인 실행/모델/게이트 절차 문서.
- `prompt_lab/README.md`, `prompt_lab/CLAUDE.md` — prompt_lab 설계 결정.

---

## 11. A100서 권장 순서 (Definition of Done)

1. 도커 CUDA/드라이버 정합 확인(`nvidia-smi` 정상).
2. `trellis2` env 구축(§3.2 핀 실측치) + 컴파일 확장은 그쪽 GPU arch 로 확인·빌드(§2·§3.3) + usd-core/OpenEXR.
3. 모델 3종 다운로드(§4, 게이트 승인).
4. ollama + gpt-oss:20b.
5. **검증**: `T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda` → GLB 생성 + usd 항목 수 대조.
6. (선택) prompt_lab pytest 184 passed + `prompt_metrics` env.
7. 동료들과 이 기준 환경 공유. 확정 시스템 프롬프트(§7)를 t2i 단계에 적용.
