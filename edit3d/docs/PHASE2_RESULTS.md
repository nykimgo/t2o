# Phase 2 — C: Region Editing 게이트 결과 (§4c)

> 실행: 2026-08-04, 4090(GPU0), env `trellis2`. 러너 `edit3d/phase2_region.py`,
> 구현 `edit3d/region.py`, 결과 `edit3d/phase2_results.json`, 정성 `edit3d/phase2_out/`.
> 시나리오: seagull_209626 → encode(64³ SLat, 8384 토큰) → **날개끝 영역(1995 토큰, 축0 상위 25%)만
> 재생성**(seed 42), 나머지 고정. 형상 latent 만(HR 단일 단계).

## 구현 (`region.py`)

- **`RegionMaskedFlowEulerGuidanceIntervalSampler`** — 배포 샘플러(`FlowEulerGuidanceIntervalSampler`,
  σ_min=1e-5, steps 12, guidance 7.5/interval [0.6,1]/rescale 0.5, rescale_t 3.0)를 서브클래싱.
  매 스텝 후 마스크 밖을 **실코드 노이즈 규약** `x_t=(1-t)x0+(σ_min+(1-σ_min)t)ε` 로 되돌림
  (PLAN §4b 단순식과 다름 — flow_euler.py:30 이 정본). CFG 는 `_inference_model` 체인이라 그대로 작동.
- **HR 단일 단계**: 기존 에셋의 `encode_shape_slat` 결과가 이미 64³ SLat 이므로 cascade LR 없이
  `shape_slat_flow_model_1024` 로 인페인팅. (stage-1 encoder 부재 우회 = §4a)
- **정확 앵커 2중**: 루프 종료 시 정규화 공간 x0 + 반환 직전 **비정규화 공간 원본 feats** 로 덮음
  → 정규화 왕복 fp 잔차(~1.9e-6)까지 제거.
- §4a 도구: `bbox_mask` / `quantile_halfspace_mask` / `fill_bbox_coords`(추가) / `merge_coords`.
- 함정 기록: 재구현 루프의 `t_seq` 는 **`.tolist()` 필수** — np.float64 × SparseTensor 는 numpy
  브로드캐스팅 경로로 새어 `ValueError: ...exceed the maximum number of dimension` 으로 죽는다.

## 게이트 판정 — 전 항목 통과 ✅

| §4c 항목 | 기준 | 실측 | 판정 |
|---|---|---|---|
| 마스크 밖 latent 불변 | **정확히 0** | **max|Δ| = 0.0** | ✅ |
| 영역 내 변화 | 실제 변화 | latent mean|Δ| = **2.50** (강한 재생성) | ✅ |
| 비편집 영역 점유 | IoU ≈ 1 | **64³ outside IoU = 1.000** (6230/6230 복셀) | ✅ |
| 경계 연속성 | 이음매 없음 | 실루엣 diff 에 경계 아티팩트 없음(`compare_region.png`) | ✅ (정성) |
| 전체 형상 | 원본 유지 | chamfer 0.0032, 실루엣 IoU 0.992 | ✅ |

peak VRAM **7.62GB** (encoder+flow1024+decoder, low_vram) — 24GB 여유.
소요: encode 9s + 샘플링 11s + 디코드×2.

## 해석과 한계

1. **인페인팅 메커니즘이 성립한다.** 마스크 밖은 수학적으로 완전 불변, 마스크 안은 latent 가 크게
   재생성되며, 모델이 전체 토큰을 보므로 경계도 매끄럽다.
2. **coords 고정 편집은 sub-voxel 디테일만 바꾼다.** inside 점유 IoU 0.99 — 토큰 집합을 안 바꾸면
   64³ 구조는 그대로이고 표면 디테일만 재생성된다(chamfer 0.003). 구조를 바꾸는 편집(제거/추가)은
   §4a 의 coords 조작(`fill_bbox_coords`/`merge_coords`, 구현됨·**미실측**)을 거쳐야 한다.
3. 이번 게이트는 **같은 조건 이미지**로 재생성했다. 의미 있는 편집(예: "날개를 접은 모양으로")은
   편집 지시를 반영한 **다른 조건**(수정 이미지 또는 텍스트→이미지)이 필요 — Phase 3 통합 주제.

## 다음

- §4a 구조 편집 실측: bbox 제거/추가(fill) → stage-2 가 새 구조를 채우는지, 경계 연속성 유지되는지
- 편집 조건 이미지 생성 경로(FLUX 재활용) 설계 → 의미 편집 E2E
- 텍스처 파이프라인 연결(`Trellis2TexturingPipeline.sample_tex_slat`) → 편집 후 재텍스처링
- Phase 3: USD 편집 지시 인터페이스 + `prompt_lab` `local_edit()` 계약 정합(PLAN §5)
