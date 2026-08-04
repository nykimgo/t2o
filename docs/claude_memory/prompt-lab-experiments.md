---
name: prompt-lab-experiments
description: "previs_proj/prompt_lab 프롬프트 최적화 실험 — 로깅 관례, objective 설계 결정, 실행 env"
metadata: 
  node_type: memory
  type: project
  originSessionId: de28af6e-e059-414a-9062-a17eccad12fd
  modified: 2026-07-28T02:08:42.384Z
---

`/home/sr/previs_proj/prompt_lab` 의 text→image(FLUX schnell)→3D 프롬프트 최적화 실험.

**사용자 요구**: 실험 기록을 **논문에 쓸 수 있게 맥락까지 상세히** 남길 것. 숫자만이 아니라
왜 그 결정을 했는지·어떤 반전이 있었는지 기록.
- **실험 로그: `prompt_lab/docs/EXPERIMENT_LOG.md`** — 실험할 때마다 append. 방법/결과/해석/한계/재현법 포함.
  실측 vs mock 반드시 구분(mock 숫자는 임의값, 결론 근거 금지).

**실행 env (2개)**:
- lab 본체 + FLUX 생성 + CV = conda **`trellis2`** ([[trellis2-env-pins]]). pytest 추가함. `pytest -q`→184 passed,1 skipped.
- ImageReward + VQAScore 채점 = conda **`prompt_metrics`** (격리; transformers 4.36.1/diffusers 0.31.0). subprocess로 호출.
- FLUX.1-schnell 가중치는 로컬 `t2o_pipeline/hf_models/FLUX.1-schnell`(54G). 스텁 A는 `previz_pipeline/text_to_image.py` 로딩을 미러링(단 _POSITIVE_SUFFIX 미포함 + max_sequence_length=256).

**objective 설계 (확정, 2026-07 사용자 승인)**:
- 게이트(차단): CV(결정론) — global_failure 조기 차단.
- **주 목적함수: VQAScore(이미지, 전체 프롬프트)** — "맞는 물체 + 3D입력 조건"을 동시 측정.
- 보조: ImageReward / CLIP_obj (앙상블).
- 근거: 정렬 지표를 **객체에만** 걸면 어수선한 배경(bare)을 선호 = 3D에 독(CLIP·IR·VQA 일관).
  **전체 프롬프트**에 걸면 VQAScore가 iso(깨끗) 0.90 vs bare 0.70으로 잘 분리 = 원하는 동작. CV와 교차검증됨.
- 사용자 관점: text→image 최적화가 가치의 95%. image-to-3D는 대체로 잘 됨, "스마트폰→벽돌 두께" 류 거친 비율 실패만 나중에 체크.

**입력 구조**: USD 계층에서 현재 #1(scene/objects/object_N.usda) 필드만 생성에 사용(shot 오버라이드/장면설명/이미지는 미사용, 종합은 미구현). 종합이 실패했던 건 shot 필드가 대부분 장면맥락이라 isolation을 깨서. 이미지 조건화는 스토리보드가 너무 거친 스케치라 보류(retrieval 썸네일은 "생성 대신 검색" 산출물이라 역방향, 쓰면 안 됨).

**T-pose(생명체 하드 요구)**: 리깅 위해 생명체는 T2I부터 T-pose. FLUX가 잘 만듦(refsheet 문구 최고). **단 CV가 펼친 날개를 multiple_objects로 오판 → 생명체는 CV single_object를 하드게이트에서 빼고 VQAScore가 심판.** category 조건부 T-pose 절 주입 + shot action 억제.

**진행 상태**: Phase 1~3 완료. 스텁 A(FLUX)·VQAScore metric·CV soft-gate·T-pose assembly 절 구현. **Phase 3 auto 루프 실작동 확인**(config `image_seagull_tpose_auto.yaml`, ollama gpt-oss:20b): LLM 후보가 base를 이김(수용 100% vs 67%). 회귀 184 passed. 상세는 EXPERIMENT_LOG.md.
- gpt-oss:20b는 reasoning 모델 → `llm.params.max_tokens` 넉넉히(4096) 줘야 JSON 안 잘림. 콜당 ~99s. parse_candidates는 잘림/후행텍스트에서 완성 후보 salvage하도록 강건화됨.
- 다음(본 스윕): 세대/후보 확대, 업스트림 8레코드 일반화, tpose 가중 조정, shot-필드 ablation.

**고정 템플릿 최적화 3-arm 실험 진행 중** (사용자 목표 = 객체무관 T2I 시스템 프롬프트 1개):
- ablation에 `isolation_text` 축 추가함(ablation.py+config.py, 테스트 갱신, 184 passed). 2단계 아키텍처 확립: Phase1 병렬 FLUX 생성+CV → Phase2 메인프로세스 데몬으로 VQA 사후채점(`scratchpad/phase2_vqa_ablation.py`). **이유: VQA 데몬을 병렬 워커 안에서 띄우면 BrokenPipe로 죽음.** budget 기본값 함정 주의(max_objects 200, max_llm_calls는 0 주면 즉시 stop).
- **접근 A 완료** (run 20260727-165903): 우승 고정 템플릿 = `{base_description}, {appearance}, {object}, single centered object, neutral background, full object visible in frame, unoccluded` (fields 전부, order reversed, neutral 격리). LOO 8/8 안정, mean_vqa 0.884. **반전: 갈매기서 이긴 matte_white/product-photography가 무생물 8객체선 최하위 — 객체별 최적≠고정 템플릿 실증.**
- 남음: 접근 B(LLM 템플릿 탐색, 신규 코드), 접근 A→B(A우승 시작점). 3개 최종 비교.

**확정 설계 (2026-07, EXPERIMENT_LOG §9)**: category 라우팅 → 시스템 프롬프트 2개.
- 무생물: 고정 템플릿(검증). 생명체: LLM 빌더 1콜(body-plan을 biped/quadruped/bird로 분류 + 고정 자세 스캐폴딩 verbatim 삽입, 창작 금지. object만). 자세 기준은 Tripo rig-type 참고(실제 리거로 쓰는 건 아님).
- 생명체 자세 3종은 짧은 테스트 잠정값(형태별>A통일 확인: A통일은 말=뒷발서기/개구리=의인화 붕괴). 정련하려면 O/X 채점 필요(**VQA는 자세품질 판단 불가**).
**A100 이관 (2026-07)**: 작업을 A100 공유 도커 서버로 옮김(동료 공유용 text→image→3D 환경 셋팅이 급선무). 핸드오프 문서 작성(구 `A100_HANDOFF.md` — 2026-08-04 삭제, 유효 내용은 `docs/SERVER_4090_SETUP.md` 로 이관) — 맥락+env핀+A100차이(FLUX schnell 상주, fp8 불필요, sm_80 빌드)+확정 프롬프트2개+함정 총정리. A100은 80GB라 4090의 offload/NVML/fp8 문제 없음. ComfyUI는 최종산출물용(후순위).

- **(4090 한정) fp8 상주 모델** — FLUX schnell bf16 ~34G라 24GB 4090 상주 불가(offload로 장당 25-41s). fp8이면 상주+~2-3s. 옵션: Kijai/flux-fp8(transformer-only e4m3fn, from_single_file) 또는 optimum-quanto 즉석양자화(로컬 bf16 재사용, 무다운로드). 양자화는 프롬프트 재검증 필요(§9 원칙).
