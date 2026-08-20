# 파일 전송 매니페스트 — text→image→3D 전체 시스템 (4090 dev → A100)

> **목적**: 이 4090 dev 서버에서 A100 서버로 옮길 때 **필수 파일**을 빠짐없이 정리.
> **사용법 (A100 서버 Claude용)**: 아래 각 항목의 **경로 존재 + 크기**를 그쪽 서버에서 확인해
> `[있음]` / `[없음→전송]` / `[부분/크기불일치→확인]` 으로 필터링하라. **A100 은 이 시스템이 원래 있던
> 동일 서버**이므로, 특히 대용량 모델(§B, ~82GB)은 **이미 존재할 가능성이 높다** — 존재하면 전송 생략.
> 없으면 "전송" 또는 "HF 재다운로드(게이트 승인 필요)" 중 택.
> **기준 경로(dev)**: `/home/sr/previs_proj/`. env 는 전송하지 않고 재구축한다(→ `A100_HANDOFF.md`).

---

## 요약 (무엇을 옮기나)

| 분류 | 크기 | 전송? |
|---|---|---|
| A. 코드 (파이프라인+prompt_lab) | ~825MB (대부분 trellis2_src 821M) | ✅ 필수 (또는 git clone) |
| B. 모델 가중치 (hf_models + HF 캐시) | ~88GB | ⚠️ **A100에 이미 있을 것 — 먼저 확인**. 없으면 전송/HF다운 |
| C. 입력 데이터 (movie_usd) | 1.7GB | ✅ 필수 |
| D. 문서 | <1MB | ✅ 권장 |
| E. env (conda) | — | ❌ 전송 안 함, 재구축 |
| F. 생성물/캐시 | ~7GB+ | ❌ 전송 안 함 |

---

## A. 코드 — 필수 (작음, 전송 또는 git clone)

repo = github `nykimgo/t2o`, 브랜치 `feature/trellis2`. **git clone 가능하면 그게 최선**(trellis2_src 제외).

| 경로 | 크기 | 내용 |
|---|---|---|
| `t2o_pipeline/previz_pipeline/*.py` | 504KB | ★파이프라인 핵심: `object_generate` 로직, `text_to_image.py`, `trellis2_inference_core.py`, `usd_parser.py`, `usd_parse_and_augment.py`, `merge_glb_to_usd.py`, `glb_to_usd_native.py`, `json_parse_and_inference.py`, `bilingual.py`, `e2e_test.py` |
| `t2o_pipeline/trellis/` | 872KB | TRELLIS 유틸(구버전 text-to-3D 경로 등) |
| `t2o_pipeline/trellis2_src/` | **821MB** | ★TRELLIS.2 소스. **git 추적 대상 아님** → 별도 clone(SHA는 `NEW_SERVER_SETUP.md §3.1`) 또는 전송 |
| `t2o_pipeline/*.sh *.md` | <1MB | `run_usd_to_3D_object.sh`, `fix_flexicubes_submodule.sh`, `ENV_REBUILD_GUIDE.md`, `NEW_SERVER_SETUP.md`, `TRELLIS2_MIGRATION.md`, `README.md` |
| `object_generate.sh`, `prompt_lab.sh`, `RUN_GUIDE.md` (previs_proj 루트) | <100KB | ★메인 진입 스크립트 + 실행 가이드 |
| `prompt_lab/previs_lab/` | 1.4MB | ★prompt_lab 코드(어댑터/CV/엔진/ablation 등) |
| `prompt_lab/run.py` `configs/` `data/` `tests/` `docs/` `*.md` `requirements.txt` | ~2MB | ★run.py, config 9종+신규(image_template_ablation, image_creature_ablation, image_seagull_tpose_auto), data(upstream_sample.jsonl, creature_sample.jsonl), 테스트, 문서 |

> prompt_lab 단독 번들 방법은 `prompt_lab/docs/TRANSFER_MANIFEST.md`(기존, prompt_lab 전용) 참고.

---

## B. 모델 가중치 — ⚠️ A100에 이미 있을 가능성 높음 (먼저 존재 확인)

| 경로 | 크기 | 없으면 얻는 법 |
|---|---|---|
| `t2o_pipeline/hf_models/FLUX.1-schnell` | **54GB** | HF `hf download black-forest-labs/FLUX.1-schnell`(게이트 승인 필요). Apache-2.0 |
| `t2o_pipeline/hf_models/TRELLIS.2-4B` | **16GB** | HF 다운로드 |
| `t2o_pipeline/hf_models/flux-fp8` | 12GB | `Kijai/flux-fp8` → `flux1-schnell-fp8-e4m3fn.safetensors`. **4090/ComfyUI용, VRAM 큰 서버면 생략 가능** |
| `~/.cache/huggingface/hub/models--briaai--RMBG-2.0` | (게이트) | 배경제거. HF 로그인+승인 후 자동 다운 |
| `~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m` | (게이트) | TRELLIS.2 특징추출. HF 승인 필요 |
| `prompt_lab/hf_cache/` (CLIP 등) | 5.9GB | prompt_lab VQA/CLIP 채점용. **재다운로드 가능**(clip-vit-large 등) — 전송 불필요 |
| `~/.cache/ImageReward/ImageReward.pt` | ~2GB | prompt_metrics 쓸 때만. 첫 실행 자동 다운 |

**게이트 레포 3종**(FLUX.1-schnell / RMBG-2.0 / dinov3-vitl16): HF 계정 승인 필요(dev는 `raengs`).
A100에 캐시가 이미 있으면 재다운로드 불필요.

---

## C. 입력 데이터 — 필수

| 경로 | 크기 | 내용 |
|---|---|---|
| `movie_usd/` | 1.7GB | ★입력 USD 씬(hidden_time 등). `object_generate.sh` 실행 대상. 검증에 필요 |

---

## D. 문서 — 권장 (맥락)

| 경로 | 내용 |
|---|---|
| `A100_HANDOFF.md` | ★환경 셋팅 인수인계(이번 세션) |
| `prompt_lab/docs/EXPERIMENT_LOG.md` | ★프롬프트 최적화 실험 전체 기록 |
| `t2o_pipeline/ENV_REBUILD_GUIDE.md` | env 핀 근거(4090 검증) |
| `t2o_pipeline/NEW_SERVER_SETUP.md` | 파이프라인 실행/모델/게이트 절차 |

---

## E. 전송하지 않음 — env (재구축)

conda env `trellis2`, `prompt_metrics`, prompt_lab `.venv` 는 **전송하지 말고 재구축**한다
(핀·절차는 `A100_HANDOFF.md §3`). ollama `gpt-oss:20b` 도 `ollama pull` 로 재취득.

---

## F. 전송하지 않음 — 생성물/캐시 (재생성)

| 경로 | 이유 |
|---|---|
| `prompt_lab/runs/` (1.2GB) | 실험 산출물(events/이미지). 재생성됨 |
| `t2o_pipeline/t2o_results/` (11MB) | 파이프라인 출력 |
| `prompt_lab/.venv/`, `**/__pycache__/`, `**/.pytest_cache/` | env/캐시 |
| scratchpad, 로그 | 임시 |

⚠️ 단, `prompt_lab/runs/` 안에는 이번 실험 원점수(무생물/생명체 ablation, phase2_vqa.json)가 있다.
**재현·재분석이 필요하면** 이 두 run 디렉터리는 선택적으로 백업 고려:
`20260727-165903-ablation-*`(무생물), `20260727-192640-ablation-*`(생명체).

---

## 검증 (전송 후)

1. **코드 무결성**: A/C/D 항목 전송 후 파일 수/크기 대조. (repo는 `git status` clean 확인)
2. **모델 크기 대조**: `du -sh hf_models/*` 가 dev(FLUX 54G / TRELLIS 16G)와 일치하는지.
   `stat -c %s` 로 핵심 safetensors 바이트 일치 확인(부분 다운로드 방지).
3. **usd-core 함정**: 파이프라인 실행 시 usd 파싱 항목 수가 dev 와 같은지(§A100_HANDOFF §3.2).
4. **end-to-end**: `T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda` → GLB 생성.
5. prompt_lab: `pytest -q` → 184 passed 기대(GPU 무관 테스트).

---

## A100 Claude 에게 요청할 것 (필터링)

> "위 매니페스트의 각 경로를 이 서버에서 확인해서, **이미 존재(크기 일치)** / **없어서 전송 필요** /
> **부분·불일치(재확인)** 로 분류해줘. 특히 §B 모델은 이 서버에 원래 있던 것이라 대부분 존재할 것이니
> 먼저 확인하고, 실제로 전송이 필요한 건 §A 코드·§C movie_usd·§D 문서 위주일 가능성이 높아."
