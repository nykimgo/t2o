# EXP2 — Near-miss 검색 결과 재활용 실험 계획

> 목적: 검색 임계값 미달로 버려지는 top-K 에셋(`retrieval_result`)을 생성의 **참조 입력**으로
> 되돌리면, 텍스트 단독 생성 대비 형태·비율 정확도가 오르는지 검증한다.
> 배경: 프리비즈 품질 기준은 "형태 인식 가능성·비율 정확도"(CONTEXT.md §6)이고, near-miss는
> 이미 USD에 실물까지 들어와 있어(`retrieval_objects/` ~29MB) 획득 비용이 0이다.
> 실측 사례: hidden_time의 object_1~3에 각 top-5, 총 15후보가 저장되어 있다.
> object_1 rank_1은 score 0.2998로 임계값 미달 생성 폴백이었다.
> 유사 방향의 선행: RefAny3D(ICLR 2026, 3D 에셋 멀티뷰 렌더를 이미지 생성의 조건으로),
> Retrieval-Augmented Score Distillation, MV-RAG. 프로덕션 검색 DB의 "임계값 미달" 세팅은 미탐구.
> 작성: 2026-08-06 / 현실 대조·P0 비GPU 준비: 2026-08-07
> **GPU 착수 시점: EXP1 생성·리깅 종료 및 산출물 고정 이후**

## 0. 2026-08-07 코드 현실과 실행 경계

- EXP1은 파일럿을 마쳤고 `EXP1_rigging/`에서 본실험 산출물을 생성 중이다. EXP2의 GPU
  작업(Qwen/Edit, FLUX, TRELLIS.2)은 EXP1과 동시에 실행하지 않는다.
- EXP1의 `s2_lift.py`에는 TRELLIS.2가 반환한 **메모리상 mesh**의 4뷰 렌더가 구현되어
  있다. 기존 USDC/GLB를 입력으로 받는 렌더러는 아니므로 카메라·뷰 조립 로직만 재사용한다.
- `prompt_lab`의 `lift_only()` 인터페이스는 있지만 TRELLIS.2 adapter는
  `NotImplementedError`이므로 경유하지 않는다. 실제 lift adapter는 원본 코드를 수정하지
  않으면서 `previz_pipeline/trellis2_inference_core.py`의 검증된 `self.pipeline.run(...)`
  호출과 sampler/preprocess 설정을 재사용한다.
- 객체 설명은 단일 `description_en`이 아니다. 실제 customData의
  `base_description.{en,ko}`와 `appearance.{en,ko}`가 생성·평가 설명의 정본이다. 반면
  `retrieval_result.query_text`는 검색 당시의 `name.ko + ". " + appearance.ko` 기록이므로
  생성 target text와 혼용하지 않는다.
- P0 비GPU 준비 코드는 공유 파이프라인을 수정하지 않고 프로젝트 루트의
  `EXP2_nearmiss/`에 격리했다. GPU 번호·개수·락은 고정하지 않으며 실제 실행 서버에서
  adapter 실행 인자 또는 외부 환경으로 지정한다.

---

## 1. 가설

| ID | 가설 | 기각되면 |
| --- | --- | --- |
| H1 | near-miss 참조를 쓰면 텍스트 단독 대비 형태·비율 정확도가 오른다 | 현행 유지, retrieval_result 미소비가 정당화됨 |
| H2 | 참조의 효용은 유사도 score에 단조 의존한다 — "충분히 가까운" 구간에서만 이득 | score 무관하면 임계값 설계 단순화 |
| H3 | 참조 사용은 자세 스캐폴딩과 충돌할 수 있다 (near-miss가 접힌 자세면 자세 오염) | 충돌 없으면 EXP1 결과와 독립적으로 도입 가능 |

H2가 사실이면 실용적 함의가 크다: 검색 임계값을 2단으로 나눠
"채택(≥T1) / **참조 생성(T2~T1)** / 순수 생성(<T2)"의 3경로 폴백을 제안할 수 있다.
(임계값 소유는 검색 모듈이므로, 결과는 제안 근거 자료로 담당자에게 전달)

## 2. 비교 경로 (독립변수 1 — 파이프라인 변형)

| 경로 | 구성 | 역할 |
| --- | --- | --- |
| A0 | 텍스트 → FLUX → TRELLIS.2 (현행) | baseline |
| A1 | near-miss 멀티뷰 렌더 + 지시문 → **이미지 편집 모델** → 참조 이미지 → TRELLIS.2 | 본 제안. "이 형태를 유지하되 대상 객체로 수정" |
| A2 | near-miss 대표 렌더 → `trellis2_inference_core.py`의 I2O 호출 계약으로 직접 lift | 대조군 — `prompt_lab.lift_only()`는 사용하지 않음 |

**A1 편집 모델 선정 제약 (라이선스 — CONTEXT.md §6)**:
FLUX.1 Kontext [dev]는 비상업 라이선스라 **제품 경로 후보에서 제외** (진단 용도만).
1차 후보: **Qwen-Image-Edit (Apache-2.0)**.
⬜ 선행 확인: 4090 24GB 단일 카드 탑재 가능 여부(양자화 필요할 수 있음), 실행 시간.
탑재 불가 시 파일럿 한정으로 Kontext dev를 진단용으로 쓰되 결과 문서에 라이선스 제약 명기.

## 3. 실험 조건 (독립변수 2)

| 변수 | 수준 | 비고 |
| --- | --- | --- |
| 경로 | A0 / A1 / A2 | |
| 객체 | 12~15종 | 아래 표집 방법 |
| score 구간 | 3구간 (예: 0.25~0.30 / 0.30~0.35 / 0.35~T1) | **H2의 핵심 축.** 구간 경계는 DB 분포 보고 확정 |
| 시드 | 2 | 생성 비용 고려 (EXP1보다 런당 비용 큼) |

**객체 표집**: ① 실전 케이스 — `movie_usd`에서 near-miss가 실존하는 객체 전수
② 합성 케이스 — 검색 DB에서 질의문과의 score가 목표 구간에 들어가는 에셋을 역으로 뽑아
가상의 "미달 검색" 시나리오 구성. → ⬜ **검색 모듈 담당자에게 DB 접근과 score 분포 요청 필요.**

총 런 수(예): 15객체 × 3경로 × 2시드 = **90런** + 편집 단계.

**현재 실존 표본의 정확한 규모**: target 객체는 3개(갈매기·스마트폰·차량), 후보는 각
top-5로 15개다. object_2는 질의가 스마트폰인데 후보 다수가 seabird라 near-miss라기보다
검색 실패에 가깝다. 15후보를 독립 객체 15개로 간주하면 pseudoreplication이므로,
파일럿에서는 `object_key`로 군집을 보존하고 본실험 12~15객체 조건은 합성 케이스 확보 뒤
충족한다. 후보별 A0 행은 비교표 정렬을 위해 manifest에 존재하지만 실제 생성 산출물은
`object_key + seed + prompt hash` 단위로 중복 제거한다. 준비된 plan은 A0 산출물과 A1/A2의
공통 참조 렌더에 `execute_once`/`reuse_from_run_id`를 기록해 중복 실행 대상을 명시한다.

## 4. 측정 (종속변수)

| 지표 | 방법 | 비고 |
| --- | --- | --- |
| 형태 인식 가능성 | 사람 채점 5점 — "렌더만 보고 무엇인지 알아볼 수 있나" | 주 사용자 관점(연출·감독, §3). 블라인드 채점 |
| 비율 정확도 | 사람 채점 5점 — 의도 기술문 대비 비례·구조 | 〃 |
| 의미 정합 | VQAScore 또는 CLIP text-image (`base_description.en` + `appearance.en` ↔ 4뷰) | 자동 보조 지표. 정확한 모델 revision 고정 |
| 참조 누출 | near-miss와의 과유사 여부 — "검색이 미달 판정한 것을 그대로 복제하지 않았나" | A1·A2의 함정 검출. DINOv2 신규 설치 또는 CLIP image-image + 사람 확인 |
| 자세 오염 (H3) | 생물 객체 한정: EXP1의 S1·S2 채점 재사용 | 참조 자세 vs 스캐폴딩 문구 충돌 관찰 |
| 비용 | 경로별 wall-clock | A1의 편집 단계 추가 비용 정량화 |

## 5. 분석 계획

1. 주 결과: 경로별(A0/A1/A2) 형태·비율 점수 — H1.
2. score 구간 × 경로 상호작용 — H2. 구간별로 A1−A0 이득 곡선.
3. 참조 누출률 — A1이 "새 객체 생성"과 "기존 에셋 복제" 사이 어디에 있는지.
4. 자세 오염 — A1의 S1/S2 통과율을 EXP1의 A0 값과 대조 — H3.
5. 실용 제안 도출: 3경로 폴백(T1/T2)의 근거 수치 정리 → 검색 모듈 담당자 전달.

## 6. 실행 절차

- **P0a 비GPU 준비 (완료, `EXP2_nearmiss/`)**: ① `retrieval_result` reader와 실전 표본
  inventory ② 실행 manifest와 append-only JSONL/CSV schema ③ A0/A1/A2 stage orchestration
  dry-run ④ USDC/GLB CPU 로딩·경계 상자·4뷰 카메라 계산 prototype.
- **P0b GPU/렌더 선행 작업 (EXP1 종료 후)**: ① CPU prototype의 카메라 계약을 실제
  raster renderer에 연결 ② Qwen-Image-Edit 단일 GPU 탑재 테스트(양자화 여부 포함)
  ③ A1 지시문 템플릿 초안 ("preserve overall shape and proportions, change to a <대상>...")
  ④ 검색 담당자에게 DB score 분포·접근 요청.
- **P1 파일럿 (1일)**: 실전 near-miss 케이스(hidden_time/object_1 포함) 2~3건으로
  A0/A1/A2 전 경로 관통. 편집 결과의 질을 눈으로 확인하고 지시문 조정.
- **P2 본실험 (1~2일)**: 90런. 런별 CSV (object, path, score_bin, seed, 각 지표, 시간).
- **P3 채점·분석 (1~2일)**: 블라인드 채점(경로 숨김), 분석, 제안서 정리.

## 7. 판정과 후속

| 결과 시나리오 | 후속 조치 |
| --- | --- |
| A1 > A0 전 구간 | 참조 생성 경로 도입 검토 + 3경로 폴백 제안. **논문화 우선 검토** (아이디어 신규성이 EXP1보다 높음 — 2026-08-06 대화) |
| A1 > A0 고score 구간만 | H2 확정 — 임계값 2단화 제안. 논문화 가능 (이득 곡선이 기여) |
| A1 ≈ A0 | 현행 유지. retrieval_result 미소비 결정(§2-1)이 데이터로 정당화됨 — 그 자체로 기록 가치 |
| A2가 의외로 강함 | "미달" 임계값이 과하게 보수적이라는 신호 → 검색 담당자와 임계값 재논의 |

## 8. 리스크

- Qwen-Image-Edit 24GB 탑재 실패 → 양자화 / 파일럿 한정 Kontext dev(비상업, 진단만) / API 대체
- usdc 렌더 파이프 구축 공수가 예상보다 클 수 있음 → CPU 로딩·카메라 prototype은 완료,
  실제 raster backend는 P0b에서 조기 판단
- 실전 near-miss 표본이 적음 (3객체·15후보, 유효 near-miss는 사실상 object_1과
  object_3 중심) → 합성 케이스로 보강하되 결과 해석 시 구분
- 검색 DB 접근 협조 지연 → 실전 케이스만으로 축소 파일럿 먼저
- A1 편집이 참조를 과하게 복제 → 참조 누출 지표로 검출, 지시문에서 창작 자유도 조정

## 9. 비GPU 준비 산출물과 재현 명령

`EXP2_nearmiss/README.md`의 명령으로 inventory와 manifest를 재생성한다. 현재 manifest는
15후보 × 3경로 × 2시드 = 90행이며, 각 행은 `object_key`, `case_id`, retrieval UID/rank/score,
원본 경로, 예상 산출물 경로를 가진다. `prompt_snapshot`은 의도적으로
`UNRESOLVED_BEFORE_GPU_RUN` 상태다. EXP1 결론과 실제 실행 코드가 고정된 뒤 prompt 전문,
scaffolding digest, 모델 revision, code commit을 채우기 전에는 GPU 본실험을 시작하지 않는다.

Canonical EXP2 렌더 뷰는 `front / side / top / three_quarter`로 고정했다. CPU prototype은
USD stage의 up-axis와 world bound 또는 GLB scene bounds로 네 카메라의 camera-to-world와
world-to-camera 행렬을 계산한다. 실제 raster backend도 이 계약과 동일한 FOV·margin을
사용해야 한다.
