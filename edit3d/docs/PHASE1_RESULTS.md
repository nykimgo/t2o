# Phase 1 — B: Detail Variation 실측 결과

> 실행: 2026-08-04, 4090(GPU0), env `trellis2`. 러너 `edit3d/phase1_variant.py`,
> 결과 `edit3d/phase1_results.json`, 정성 `edit3d/phase1_out/`. 계획: `PLAN.md` §3.
> 대상: seagull_209626 (조건이미지 = 그 `_ref.png`), ss_res=32, seeds=[42,123,777]. **형상만**(텍스처 생략).

## 구현
- `edit3d/variant.py` — `run_variant_shape/full`: run() 의 1024_cascade 경로에서 `sample_sparse_structure`
  만 **주입 coords** 로 대체, 나머지(cascade LR→upsample→HR→decode)는 그대로. trellis2_src 무수정.
- 부분 로드(SS+shape flow+shape decoder, **tex 제외**) → **peak VRAM 4.32GB** (형상만이라 decode_latent 대신
  decode_shape_slat, 매우 가벼움).

## 실측 (단위 큐브 정규화 후 지표)

두 경로 비교:
- **A(캡처)**: 정상 SS 로 뽑은 32³ coords 고정 → seed 만 변주. '구조 고정 looseness' 순수 측정.
- **B(제품)**: 기존 GLB 를 `coords.py.mesh_to_coords(32)` 로 복셀화해 주입. 실제 편집 경로.

| 비교 | Chamfer↓ | 실루엣 IoU↑ | bbox 종횡비 L1↓ |
|---|---|---|---|
| **A: variant vs baseline** (구조고정, seed변주) | 0.0044 | **0.97** | 0.006~0.010 |
| A: seed 간 pairwise 평균 | 0.0045 | 0.967 | — |
| A: baseline vs 원본에셋 | 0.0264 | 0.745 | 0.173 |
| **B: variant vs 원본에셋** (3 seed) | 0.0055 | **0.938** | 0.032~0.043 |
| coords IoU(A_ss, B_voxel) | — | **0.344** | — |

## 결론

1. **32³ 구조 고정은 타이트하다.** 고정 coords + 동일 조건에서 stage-2 seed 를 바꿔도 형상 드리프트가
   작다(chamfer 0.0044, 실루엣 **0.97**, 종횡비 거의 불변). cascade 가 hr 구조를 seed 마다 재유도하지만
   (n_hr 8014/7986/8050 로 미세 변동) 결과는 원본 근처에 머문다.
   → **PLAN §0 의 "32³ 이라 v1 64³ 보다 형상을 더 흔들 것" 우려는, 적어도 동일 조건 seed 변주에서는
      기각.** B(구조보존 편집)에 유리.
2. **제품 경로 B 가 원본에 더 충실.** 에셋을 복셀화한 coords(B)로 재생성하면 원본 실루엣 **0.938** 재현.
   신규 SS coords(A)로 재생성한 것(0.745)보다 훨씬 낫다.
3. **복셀화 coords ≠ SS coords (IoU 0.344) 인데도 B 가 잘 된다.** 부피는 유사(1909 vs 1858)하나 위치가
   다름 — shell(복셀화) vs SS occupancy 표현 차이. 그럼에도 조건이미지+coords 가 형상을 원본으로 끌어당긴다.
   즉 **편집 대상 에셋에는 신규 SS 보다 '그 에셋의 복셀화 coords' 가 더 좋은 앵커다.**
4. 정성(`phase1_out/compare_silhouette.png`, 동일 뷰): 세 생성물이 서로 거의 동일한 coherent 갈매기,
   원본과는 소폭 다름 — 정량치와 일치. 파이프라인·coords 주입 정상 동작 확인.

## 한계 / 다음
- 이번은 **동일 조건이미지 + seed 변주**만. **진짜 detail variation(다른 텍스트/이미지 프롬프트)** 은 미검증 —
  그때 형상 변화가 더 클 수 있다(§3 의 "찻잔→머그컵" 류). 다음 실험: 조건이미지를 바꿔 구조 고정도 재측정.
- 텍스처 미포함. 외형 변화(텍스처 히스토그램) 지표는 `run_variant_full` + tex 모델 로드로 별도 측정.
- B 의 aspect_l1(0.03~0.04)이 A 의 seed 변주(0.006)보다 큼 = 복셀화 앵커가 종횡비를 약간 바꾼다. 허용 범위.
