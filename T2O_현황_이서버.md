# T2O 현황 스냅샷 — 이 서버 (A100 / previs-prep)

> **작성일**: 2026-08-12  
> **목적**: 개인(4090) 서버에서 실험이 많아져 헷갈릴 때, **이 A100 서버의 t2o 상태만** 한눈에 복구.  
> **범위**: 코드/설정 **수정 없이** 디스크·git·문서·산출물을 조사한 결과.  
> **정본 경로**: `/root/previs_proj/t2o_pipeline/` (원격 `https://github.com/nykimgo/t2o.git`)
>
> ⚠️ **2026-08-20 기준 구성은 `T2O_구성_스냅샷_20260820.md` 를 볼 것.** 그 이후 `t2i` env 삭제 ·
> `unirig` env 및 오토리깅 파이프라인 신규 구축이 있었고, 아래 §1 GPU 대수(×4)와 §6 T2I 속도
> (2.5s/장)는 실측으로 정정됐다. 본 문서는 **트랙별 실험 진행도**(프롬프트·edit3d)의 정본으로 유효하다.

---

## 0. 30초 요약

| 트랙 | 이 서버에서의 상태 | 한 줄 |
|---|---|---|
| **배포 파이프라인** (T2I→TRELLIS.2→USD) | ✅ 동작·동결 | `feature/trellis2` @ `7cfe36b`, origin 동기화 |
| **스캐폴딩 / 리깅 대비 자세** | ✅ 무생물 확정 · ⚠️ 생명체 잠정 | `t2i_prompt_builder.py` 에 배포됨. 실제 오토리깅 툴 연동은 **없음** |
| **프롬프트 최적화 (`prompt_lab`)** | ✅ **실험 종료** (v1 동결) | git 제외 로컬 사본 있음. 상세 로그는 `prompt_lab/docs/EXPERIMENT_LOG.md` |
| **부분편집 (`edit3d`)** | 📋 **계획만** | `edit3d/docs/PLAN.md` 만 존재. Phase 0 미착수 |
| **시간 최적화** | ✅ **계측 완료** · 속도 손잡이 실험은 보류 | T2I/I2O 분리 + stage `[TIME]`. A100은 FLUX 상주(~2.5s/장). 4090용 fp8 등은 **미적용·재검증 필요** |

**다음으로 잡혀 있던 일(문서 기준)**: 4090에서 `edit3d` Phase 0 착수 (`docs/SERVER_4090_SETUP.md` §5).

---

## 1. 이 서버 / repo 식별

| 항목 | 값 |
|---|---|
| 호스트 역할 | A100 공유 서버, 컨테이너 `previs-prep`, 경로 `/root/previs_proj` |
| GPU (실측) | A100 80GB ×4 |
| conda | `trellis2` (배포), `prompt_metrics` (채점 격리), 기타 `t2i`/`trellis`/`previz` |
| 브랜치 | `feature/trellis2` → `origin/feature/trellis2` **up to date** |
| HEAD | `7cfe36b` *Consolidate handoff docs into SERVER_4090_SETUP* (2026-08-04) |
| 직전 핵심 커밋 | `96d51b8` *Fix prompt assembly bugs, add stage timings, and freeze T2I template v1* |
| 워킹트리 | 깨끗함 (서브모듈 `trellis/nxt/mesh/flexicubes` 만 modified — t2o 본선과 무관) |
| 런처 | 루트 `object_generate.sh` (repo 밖). **3단계 wall-clock 요약이 A100에 있음** |

> 4090 개발서버(`/home/sr/...`)와 역할이 갈린다: **프롬프트 캠페인·핸드오프는 A100에서 마무리 → 부분편집은 4090에서 이어받기로 문서화됨.**

---

## 2. 파이프라인 본선 (배포 상태)

**흐름**: USD 파싱 → (옵션) ollama 번역/필터 → FLUX.1-schnell(T2I) → TRELLIS.2(I2O) → GLB → trimesh+pxr USD 주입.

| 구성요소 | 위치 | 비고 |
|---|---|---|
| 파싱·증강 | `previz_pipeline/usd_parser.py`, `usd_parse_and_augment.py` | `rig_type` 최상위 우선, 구 `etc.rig_type` 하위호환 |
| T2I 템플릿 | `previz_pipeline/t2i_prompt_builder.py` | **v1 동결 대상** |
| TRELLIS.2 | `previz_pipeline/trellis2_inference_core.py` + `trellis2_src/` | backend 기본 `trellis2` |
| 가중치 | `hf_models/TRELLIS.2-4B`, `FLUX.1-schnell` | git 제외 |
| 실행 가이드 | `docs/RUN_GUIDE.md` | A100 실측 절차 |
| 서버 차이·확정 프롬프트 | `docs/SERVER_4090_SETUP.md` | **핸드오프 정본** |

**산출물 규칙**
- 로그/프리뷰: `t2o_results/{모델}/{YYYYMMDD}/run_*`
- 실제 GLB·geometry: 소스 USD 옆 `objects/assets/{object}/` + 원본 USDA에 reference 주입

**이 서버에 남은 공식 run**: `t2o_results/TRELLIS.2-4B/` 의 **2026-07-28 ~ 07-29** (그 이후 공식 run 디렉터리 없음).  
예: `run_184218_en-filter` CSV에 `t2i_time`/`i2o_time` 분리 기록 확인됨 (예: T2I ~8.6s / I2O ~21.7s @A100).

**2026-08-12 스모크**: `tmp/t2o_gpu_repro*` 로 stage1까지는 통과했으나, stage2에서 `No module named 'rembg'` / Trellis2InferenceCore 로드 실패 로그가 남음 → **환경/PYTHONPATH 이슈로 보이며, ‘파이프라인 폐기’가 아님**. 정상 실행은 `conda activate trellis2` + `object_generate.sh` 경로를 따를 것 (`docs/RUN_GUIDE.md`).

---

## 3. 스캐폴딩 / 리깅 실험 — 어디까지 됐나

### 목적
후속 **오토리깅**을 위해, 생명체 T2I 단계에서 자세를 고정(A-pose / 사족 기립 / 날개 펼침 등).  
자세 분류 taxonomy는 **Tripo rig-type을 참고**했을 뿐, Tripo/UniRig를 리거로 붙인 적은 **없음**.

### 확정 vs 잠정

| 갈래 | 상태 | 내용 |
|---|---|---|
| **무생물** | ✅ **v1 확정·동결** | `{base_description}, {appearance}, {object}, single centered object, … unoccluded` — ablation+LOO 8/8, A100 재현 |
| **생명체** | ⚠️ **잠정** | `rig_type` 또는 키워드 → `biped`/`quadruped`/`bird`/`insect` 스캐폴딩 **verbatim** 삽입. 짧은 테스트만 (“형태별 > 단일 A-pose 통일”). **사람 O/X 채점 미실시**. VQA는 자세 품질을 못 봄 |
| **`insect`** | ⚠️ §9에 없던 확장 | 업스트림 `rig_type=insect` 대응용 잠정 문구 |

### 코드에 실제로 깔린 라우팅
1. USD `customData` **최상위 `string rig_type`** 우선 (`static_object` → 무생물)
2. 없으면 category/객체명 키워드 dict
3. 생명체 프롬프트는 코드상 `{object}, {scaffolding}, {base_description}` — §9 원안 “object만”보다 **정체성 보존을 위해 base_description을 포함**하도록 배포본이 진화함 (`RUN_GUIDE.md`·빌더 docstring과 일치). `SERVER_4090_SETUP.md` §7의 “object만” 서술은 **실험 §9 원안 요약**이라 배포 코드와 한 줄 어긋날 수 있음 → **실행 시 코드/RUN_GUIDE를 따름**.

### 실험 캠페인과의 관계
- 2026-07~08 `prompt_lab` 캠페인은 **무생물 고정 템플릿**이 주전장.  
- **생명체 스캐폴딩 정련은 그 이후로 안 건드림** (문서 명시).

---

## 4. 프롬프트 최적화 (`prompt_lab`) — 종료된 실험

| 항목 | 상태 |
|---|---|
| 결론 | **배포 템플릿 v1 동결** (`EXPERIMENT_LOG.md` §10.10). 바꿀 근거 없음(반복 시 arm 차이=노이즈, 제작 레코드는 천장 효과) |
| git | `.gitignore` 에 `prompt_lab/` — **repo에 포함하지 않음**. 이 서버엔 로컬 디렉터리로 남아 있음 |
| 정본 사본 | 문서상 **4090에 있는 사본**을 정본으로 둠 (`SERVER_4090_SETUP.md` §4) |
| 이 서버 잔존물 | `prompt_lab/docs/EXPERIMENT_LOG.md` (~1000줄), `runs/` (07-28~29 ablation/template), `run_*.sh` |
| 지표 설계 | CV 게이트 + 주목적 **VQAScore(전체 프롬프트)**. ImageReward/CLIP 보조 |
| 미해결로 남긴 것 | CV 임계값 미보정(좋은 이미지 거부), 목적함수 사각지대(중복/정체성/시점), 생명체 자세 O/X |

**접근 A/B/C 요약**
- A(ablation): 우승 템플릿 = 현 배포 무생물 문구  
- B(LLM 템플릿 탐색): 1차에 목적함수 게이밍(`object` 누락) 발견 → 하드제약 후 재실행  
- A→B: A를 못 넘김. 이후 반복에서 arm 간 차이 소멸 → **배포 유지 결정**

---

## 5. 부분편집 (`edit3d`) — 계획 단계

| 항목 | 상태 |
|---|---|
| 디렉터리 | `edit3d/` → **`docs/PLAN.md` 만** (coords/variant/region 코드 **없음**) |
| 목표 | 재생성 없이 Detail Variation(B) → Region Editing(C) |
| 조사 결론 | TRELLIS.2에 B/C 공개 API 없음. B는 v1 `run_variant` 이식, C는 논문 서술+마스킹 샘플러 직접 구현 예정 |
| 블로커 후보 | `sparse_structure_encoder` 부재 → C는 coords 직접 조작으로 우회 설계. `encode_shape_slat` 왕복 손실이 Phase 0 게이트 |
| 실행 예정지 | **4090** (24GB → `low_vram`/순차 적재 필요). A100에서는 VRAM 제약 미검증 |
| 배포 연동 | Phase 3 전까지 `previz_pipeline` 건드리지 않기로 함 |

```
Phase 0 전제검증 → Phase 1 B이식 → Phase 2 C마스킹 → Phase 3 배포통합
         ↑ 여기까지 안 함
```

---

## 6. 시간 최적화 / 계측 — 무엇을 했고 무엇을 안 했나

### 완료된 것 (코드·런처에 있음)
- stage1: `[TIME]` USD 파싱 / LLM 번역 / 필터 등 (`usd_parse_and_augment.py`)
- stage2: 객체별 **T2I vs I2O** 분리 (`t2i_time`, `i2o_time` → CSV/JSON)
- `object_generate.sh`: 3단계 wall-clock 요약 (`⏱️ 단계별 소요 시간`) — **A100 루트 스크립트에 존재**. (4090 핸드오프 문서는 “4090엔 없을 수 있다”고 적어 둠)

### 속도 특성 (실험이 아니라 하드웨어 경로 차이)
| 서버 | FLUX | 대략 |
|---|---|---|
| A100 80GB | 상주 | ~2.5 s/장 (실측) |
| 4090 24GB | CPU offload | ~25–41 s/장 |

### 보류·미실행
- **4090 fp8 상주**(Kijai / optimum-quanto 등): 문서에 옵션으로만 존재. **프롬프트는 정밀도에 전이되지 않으므로 fp8 전환 시 재-ablation 필요** → 단순 스위치 아님
- TRELLIS.2 `low_vram`: edit3d용으로 언급, 배포 파이프라인 성능 캠페인으로 돌린 기록 없음
- “시간 최적화 실험”이라는 별도 캠페인 산출물(리포트/runs)은 **없음**. 한 일은 **병목 가시화(계측)** + A100 상주 이점 활용

---

## 7. 문서 지도 (어디를 보면 되나)

| 보고 싶은 것 | 파일 |
|---|---|
| 이 서버에서 돌리는 법 | `docs/RUN_GUIDE.md` |
| 4090 복귀·edit3d 착수·확정 프롬프트 | `docs/SERVER_4090_SETUP.md` |
| 부분편집 설계 | `edit3d/docs/PLAN.md` |
| 프롬프트 실험 전체 기록 | `prompt_lab/docs/EXPERIMENT_LOG.md` (git 밖) |
| env 핀 / 이관 | `ENV_REBUILD_GUIDE.md`, `NEW_SERVER_SETUP.md`, `docs/a100_reconstruct/` |
| 짧은 메모리 인덱스 | `docs/claude_memory/MEMORY.md` |

---

## 8. 알려진 미해결 이슈 (이 서버에도 해당)

1. **카메라 USDA `customData` 오배치** → shot 카메라 로드 실패 (object 생성엔 무해). 미수정.  
2. **CV 게이트 미보정** — mock 임계값이라 실이미지 과잉 거부.  
3. **목적함수 사각지대** — 중복 객체 / 정체성 / 시점. 프롬프트 최적화만으로는 불가.  
4. **생명체 스캐폴딩 미검증** — 배포는 되어 있으나 신뢰도는 잠정.  
5. **경량 `movie_usd`** — description 비면 stage1 “필수 정보 0개”. 설명 채운 사본 또는 filter 경로 필요 (`RUN_GUIDE` §2.5).  
6. (이력) 4090 NVML driver mismatch — A100과 별개이나 4090 복귀 시 확인 항목.

---

## 9. “다른 서버에서 하던 일”과 맞추는 체크리스트

개인 서버 실험이 많아졌을 때, **이 서버 기준으로 ‘끝난 것 / 안 한 것’만** 대조:

- [ ] 무생물 T2I 템플릿을 또 바꾸고 있었나? → 여기선 **바꾸지 말기로 동결됨**  
- [ ] 생명체 자세 문구를 튜닝 중이었나? → 여기선 **잠정 배포만**, O/X 채점 없음  
- [ ] 부분편집 프로토타입을 짰나? → 여기선 **PLAN만**, 코드 트리 비어 있음  
- [ ] fp8/양자화로 시간 줄였나? → 여기선 **미적용** (A100은 bf16 상주로 충분)  
- [ ] prompt_lab 새 run을 돌렸나? → 이 서버 최신 run은 **2026-07-29**  
- [ ] git은 어디에 맞춰야 하나? → **`feature/trellis2` @ `7cfe36b`**

---

## 10. 조사 메모 (재현용)

```text
조사일: 2026-08-12
방법: t2o_pipeline 트리·git log/status, docs/*, edit3d/*, prompt_lab/docs/EXPERIMENT_LOG.md,
      t2o_results 날짜, object_generate.sh 계측, 당일 tmp/t2o_gpu_repro 로그
변경: 본 문서(T2O_현황_이서버.md) 신규 작성만. 기존 코드/파일 미수정.
```
