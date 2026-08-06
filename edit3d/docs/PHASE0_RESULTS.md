# Phase 0 — 전제 검증 결과 (게이트)

> 실행: 2026-08-04, 4090 개발서버(sr-ESC8000-E11), env `trellis2`. 계획: `PLAN.md` §2.
> 상태: **0-1 ✅ / 0-2 ✅ / 0-3 ✅ / 0-4 ✅ — 게이트 통과.** C(Region Editing) 진행 가능.

---

## 0-3. 배포 `pipeline_type` / `ss_res` 확정 ✅

**결론: 배포는 `1024_cascade` → `ss_res = 32`.**

근거(코드 + 실측):
- 호출 체인이 아무도 `pipeline_type` 을 override 하지 않는다:
  - `previz_pipeline/json_parse_and_inference.py:260` — `--pipeline_type` 기본 `None`
  - `:288` — `Trellis2InferenceCore(pipeline_type=args.pipeline_type)` (None 전달)
  - `previz_pipeline/trellis2_inference_core.py:124` — `pipeline_type=None → 모델 default '1024_cascade'`
  - `:232-236` — `self.pipeline.run(pipeline_type=self.pipeline_type)`
- 모델 default: `trellis2/pipelines/trellis2_image_to_3d.py:56,108` — `default_pipeline_type='1024_cascade'`
- `ss_res` 규약: 같은 파일 `:541` — `{'512':32,'1024':64,'1024_cascade':32,'1536_cascade':32}` → cascade = **32**
- **실측**: 스모크 로그(seagull 1객체)에 객체당 `Sampling shape SLat` 진행바가 **정확히 2회** = cascade(lr+hr 2단계) 확증.

**설계 함의 (PLAN §0 해상도 규약):** 고정 구조 해상도가 32³ 라 v1(64³)보다 거칠다.
= SLat 이 형상의 더 많은 부분을 결정 = **B(detail variation)가 v1 보다 형상을 더 바꿀 수 있다** → Phase 1 실측 항목.

---

## 0-4. 평가 세트 고정 ✅

`edit3d/eval_set.json` (6개). hidden_time 캠페인 산출 GLB, 성공/실패·형상 다양성 혼합:

| id | category | geometry | quality | 크기 |
|---|---|---|---|---|
| car_big | vehicle | bulky | detailed | 37 MB |
| car_small | vehicle | bulky | degenerate | 1.3 MB |
| phone_big | phone | thin_flat | detailed | 37 MB |
| phone_small | phone | thin_flat | degenerate | 0.96 MB |
| bird_a | bird | appendage | detailed | 41 MB |
| bird_b | bird | appendage | detailed | 39 MB |

- **성공/실패**: 상세(37~41MB) vs 퇴화 의심(<1.4MB). 퇴화 케이스는 얇은 폰·단순 차량에서 나옴.
- **형상 스트레스**: bulky(차) / thin_flat(폰) / appendage(날개 편 새) — encode 왕복(0-1)의 난이도 스펙트럼.

---

## 0-1. `encode_shape_slat` 왕복 손실 ✅ — **C 실현가능**

러너: `edit3d/phase0_roundtrip.py` (결과 `edit3d/phase0_results.json`).
GLB → `preprocess_mesh` → `encode_shape_slat(res=1024)` → `decode_shape_slat(1024)` → mesh.
원본(preprocess 후)과 디코드 메시를 **단위 큐브 정규화 후** 비교(축 정규화 불일치 오염 배제).

| id | quality | geometry | Chamfer↓ | 실루엣 IoU↑ | slat 복셀 | dec faces |
|---|---|---|---|---|---|---|
| car_big | detailed | bulky | 0.00530 | 0.997 | 26,750 | 28.3M |
| car_small | degenerate | bulky | 0.00265 | 0.998 | 7,093 | 3.7M |
| phone_big | detailed | thin_flat | 0.00485 | 0.997 | 20,806 | 21.4M |
| phone_small | degenerate | thin_flat | 0.00237 | 1.000 | 4,720 | 2.5M |
| bird_a | detailed | appendage | 0.00308 | 0.992 | 8,384 | 9.9M |
| bird_b | detailed | appendage | 0.00187 | 0.993 | 2,365 | 2.7M |

- **Chamfer 0.0019~0.0053** — 참조값(동일형상 큐브 vs 0.9배 큐브 = 0.009)보다도 낮다. 즉 encode→decode
  왕복이 형상을 **거의 바꾸지 않는다.** thin_flat(폰)·appendage(날개 편 새) 같은 난형상도 동일.
- **실루엣 IoU 0.99+** (전 뷰 평균). 형상 겹침 거의 완전.
- **결론: `encode_shape_slat` 왕복 손실이 무시할 수준 → C 는 §4d 하이브리드 폴백 없이 진행 가능.**
  마스크 밖을 원본 SLat 으로 되돌리는 정공법(§4b)이 성립한다.

## 0-2. 좌표 왕복 손실 ✅ — **ss_res 에서 완전 복원**

디코드 메시를 `ss_res=32` 로 재복셀화(`o_voxel`, open3d-free) → 원본 메시의 32³ coords 와 집합 IoU.

- **6/6 모두 `coords_iou_ss = 1.000`.** 구조 해상도(32³)에서 왕복 좌표가 **완전 일치.**
- 부수 검증: IoU=1.0 이 나오려면 디코드 메시가 원본과 **같은 canonical [-0.5,0.5] 프레임**에 있어야 하므로,
  `decode_shape_slat` 이 preprocess 프레임을 보존함이 확인됨(0-2 설계의 AABB 정규화 우려 해소).
- 외부 반입 에셋도 32³ 구조 좌표는 신뢰 가능. (자체 생성 에셋은 `coords.py` 캐시로 손실 0 — 설계대로.)

---

## VRAM / 로딩 (§2-3 · SERVER_4090 §2-3 검증)

- **부분 로드 성공**: `shape_slat_encoder` + `shape_slat_decoder` **두 모델만** (`models.from_pretrained`,
  대형 flow model 제외) `low_vram=True` 패턴으로 스텝마다 GPU↔CPU. **peak VRAM 18.15 GB** → 24GB 여유.
- edit3d Phase 1/2 는 여기에 flow model(shape_slat_flow) 또는 texturing 파이프라인이 더해지므로,
  **T2I_GPU 분리 / low_vram / 순차 적재**를 계속 지켜야 함(24GB 제약, [[t2o-4090-gpu-split]] 참조).

## 다음 (Phase 1 — B: Detail Variation)

- `ss_res=32` 확정 → `run_variant_v2` 는 32³ 구조를 고정하고 stage-2 재실행(PLAN §3).
- 실측 항목: "32³ 고정이 v1(64³)보다 형상을 얼마나 더 바꾸는가."
- coords 주입 경로는 `coords.py.mesh_to_coords`(+ 캐시) 그대로 사용 가능(0-2 로 신뢰도 확인됨).
