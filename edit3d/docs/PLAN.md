# 3D 에셋 부분수정 (Partial Editing) — 구현 계획

> **작성**: 2026-08-04 (A100 공유서버 세션에서 조사). **실행 예정**: 4090 개발서버.
> **목표**: TRELLIS.2 로 생성된 3D 에셋을 **재생성 없이** 부분 수정한다.
> **범위**: B(Detail Variation) → C(Region Editing) 순차. 둘 다 필요하다는 사용자 확정.
> 관련: `../../docs/SERVER_4090_SETUP.md`,
> 프롬프트 최적화 맥락은 `prompt_lab/docs/EXPERIMENT_LOG.md` §10 (해당 실험은 종료, v1 확정).

---

## 0. 조사 결과 — 무엇이 있고 무엇이 없나 (전부 코드에서 확인함)

### 논문 vs 공개 코드

TRELLIS 논문(arXiv:2412.01506) §3.4 는 편집 모드를 **두 개** 서술한다.

| 모드 | 논문 | TRELLIS v1 공개코드 | **TRELLIS.2 (현 배포)** |
|---|---|---|---|
| **B. Detail Variation** — 구조 고정, 2단계만 재실행 | ✅ | ✅ `run_variant()` (**text-to-3D 전용**) | ❌ 없음 |
| **C. Region-Specific Editing** — bbox 로 복셀 영역 지정, 인페인팅식 | ✅ Fig.7(b) | ❌ **없음** | ❌ 없음 |
| (참고) 형상 고정 + 텍스처만 재생성 | — | — | ✅ `Trellis2TexturingPipeline` |

논문 §3.4 인용:
- Detail Variation: *"preserving the asset's structure and executing the second generation stage with different text prompts"*
- Region Editing: *"Given a bounding box for the voxels to be edited, we modify our flow models' sampling processes to create new content in that region, **conditioned on the unchanged areas**"*
- 성립 근거: *"the **locality of SLat** allows for region-specific editing by altering voxels and latents in targeted areas while leaving others unchanged"*
- **tuning-free** (재학습 불필요)

→ **B 는 v1 로직을 v2 에 이식**, **C 는 논문 서술을 보고 직접 구현**해야 한다.

### 구현을 좌우하는 API 사실 4가지

1. **`sample_shape_slat(cond, flow_model, coords, sampler_params)` 가 `coords` 를 인자로 받는다.**
   → B 는 샘플러 수정 없이 **좌표만 갈아끼우면 된다.** (`trellis2/pipelines/trellis2_image_to_3d.py`)
2. **`Trellis2TexturingPipeline.encode_shape_slat(mesh, resolution)` 가 존재하고 `ckpts/shape_enc_next_dc_f16c32_fp16` 가 로컬에 있다.**
   → 기존 에셋을 SLat 으로 되돌릴 수 있다. **C 의 핵심 전제.**
3. **`sparse_structure_encoder` 가 없다** (ckpts 에도, `pipeline.json`/`texturing_pipeline.json` 에도).
   → 기존 에셋의 stage-1 latent(`z_s`)를 얻을 수 없다. **논문 방식의 stage-1 영역편집은 불가.** → §3 에서 우회.
4. **`FlowEulerSampler.sample()` 이 `sample = out.pred_x_prev` 한 줄 루프다.**
   → C 의 개입 지점이 단 한 곳. 서브클래싱으로 충분하며 `trellis2_src` 원본 수정 불필요.

### 해상도 규약 (B/C 설계에 직접 영향)

`run(pipeline_type)` 의 sparse structure 해상도:
```python
ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
```
**배포는 cascade 로 보인다**(런 로그에 `Sampling shape SLat` 이 객체당 2회) → `ss_res = 32`.
v1 의 `run_variant` 는 64³ 고정이었으므로, **v2 에서 고정되는 구조는 v1 보다 거칠다.**
= 형상의 더 많은 부분을 SLat 이 결정한다 = **detail variation 이 v1 보다 형상을 더 바꿀 수 있다.**
→ Phase 1 의 실측 항목. (Phase 0 에서 `pipeline_type` 을 먼저 확정할 것.)

---

## 1. 코드 위치와 원칙

- **위치: `t2o_pipeline/edit3d/`** (신규 모듈, 사용자 확정)
- **`trellis2_src` 는 읽기 전용.** git clone 이므로 수정하지 않는다. 샘플러는 **서브클래싱**, 파이프라인은
  **공개 메서드 조합**으로 처리한다. (§0 의 API 사실 1·4 덕분에 가능)
- **T2O 배포 파이프라인(`previz_pipeline/`)은 Phase 3 전까지 건드리지 않는다.**
- `prompt_lab` 은 종료된 실험이므로 무관.

```
t2o_pipeline/edit3d/
  docs/PLAN.md            이 문서
  coords.py               메시/캐시 → 복셀 좌표
  variant.py              Phase 1 (B)
  region.py               Phase 2 (C): 마스크 + 마스킹 샘플러
  eval.py                 평가 지표 (IoU / Chamfer / 경계 연속성)
  tests/
```

---

## 2. Phase 0 — 전제 검증 (게이트, 반나절)

**여기서 막히면 Phase 2 가 무너진다. 먼저 잰다.**

| # | 항목 | 성공 기준 / 산출 |
|---|---|---|
| 0-1 | **`encode_shape_slat` 왕복 손실** — GLB → encode → decode → 메시. 원본과 Chamfer distance, 실루엣 IoU | **C 의 실현가능성을 결정하는 숫자.** 손실이 크면 §3 의 하이브리드 폴백으로 |
| 0-2 | **좌표 왕복 손실** — 생성 시 coords 보관 → 디코드된 메시를 재복셀화 → 두 coords 집합의 IoU | 외부 반입 에셋 편집의 신뢰도 상한 |
| 0-3 | **배포 `pipeline_type` 확정** (`trellis2_inference_core.py` 실측) | `ss_res` 32/64 확정 → B/C 설계 분기 |
| 0-4 | 평가 세트 고정 | 캠페인 산출 GLB 5~8개(권총·자동차·비행기·비행모·핸드폰). 성공/실패 사례 혼합 |

### 0-2 에 대한 설계 결정 — 좌표는 **왕복시키지 말고 캐시한다**

메시 → 좌표 복원에서 **정수 나눗셈 자체는 오차원이 아니다.** 파이프라인이 내부에서 하는 것과 같은 연산이다:
```python
decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5   # "미세복셀 하나라도 차면 점유"
```
`idx // ratio` 후 `unique()` 와 동일하다. **진짜 손실은 다른 두 곳이다.**

1. **AABB 정규화 불일치** — 좌표는 `[-0.5,0.5]` 큐브 기준이어야 한다. 편집 대상 메시를 다시 정규화하면
   원본 생성 때와 중심·스케일이 어긋나 좌표가 통째로 밀린다. → `preprocess_mesh` 를 그대로 재사용해 규약을 맞춘다.
2. **메시 왕복** — 원본 coords 는 `decoder(z_s)>0` 에서 나왔고, 재복셀화는 **디코딩된 표면**에서 나온다.
   원리적으로 다르다(디코더 점유가 팽창돼 있거나 내부 복셀이 있을 수 있음). **이게 실제 손실이며 0-2 로 측정한다.**

**→ 우리가 편집할 에셋은 대부분 우리가 생성한 것이므로, 생성 시점에 `coords` 를 GLB 옆에 캐시로 남긴다.**
- **자체 생성 에셋**: 캐시 사용 → **손실 0**
- **외부 반입 에셋**: 재복셀화 → 손실 있음, 상한은 0-2 의 IoU

> ⚠️ **`open3d` 가 trellis2 env 에 없다**(A100 기준 실측; 4090 은 §SERVER_4090_SETUP 에서 확인).
> v1 의 `o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds` 경로는 못 쓴다.
> 대신 `o_voxel.convert.mesh_to_flexible_dual_grid` 의 `voxel_indices` 를 다운샘플한다 — **기존 의존성만 사용.**

---

## 3. Phase 1 — B: Detail Variation 이식 (1~2일)

```python
def run_variant_v2(pipeline, mesh_or_coords, image, seed, ss_res):
    cond   = pipeline.get_cond([image], resolution)
    coords = load_cached_coords(...) or mesh_to_coords(mesh, ss_res)   # ← sample_sparse_structure 대체
    shape_slat = pipeline.sample_shape_slat(cond, flow_model, coords)  # 이하 run() 과 동일
    ... decode_shape_slat → sample_tex_slat → decode_tex_slat → glb
```
cascade 인 경우 `sample_shape_slat_cascade` 경로도 동일하게 좌표 주입.

**검증 지표 — 정량 + 정성 둘 다** (프롬프트 캠페인의 교훈: 지표만 보면 속는다)
- **구조 보존**: 원본 대비 실루엣 IoU, 바운딩박스 종횡비 변화율 → *얼마나 고정되는지를 숫자로*
- **외형 변화**: 텍스처 히스토그램 거리 → *실제로 바뀌긴 하는지*
- **정성**: 렌더 이미지를 직접 보고 판정 (찻잔이 머그컵이 됐는지, 비율이 유지되는지, 손잡이가 살아있는지)

**이 Phase 의 진짜 산출물**: "v2 에서 detail variation 이 형상을 얼마나 바꾸는가"의 실측치.
`ss_res=32` 면 예상보다 많이 바뀔 수 있고, 그러면 B 의 용도 자체가 재정의된다.

**용도 한계(미리 명시)**: B 는 **점유 격자를 입력으로 받아 고정**하므로 "비율이 이상하다 / 손잡이가 뭉개졌다 /
다리가 하나 없다" 류의 **형상 결함은 못 고친다.** 망가진 형상이 고정된 채 표면만 바뀐다. 그건 C 의 일이다.

---

## 4. Phase 2 — C: Region Editing (3~5일)

### 4a. 편집 영역 정의 — stage-1 encoder 부재 우회

논문은 stage-1(sparse structure)에서 영역 내 새 구조를 생성한다. 그런데 §0-3 대로 **`sparse_structure_encoder`
가 없어 기존 에셋의 `z_s` 를 못 얻는다.** → **stage-1 을 건드리는 대신 `coords` 집합을 직접 조작한다.**

- **제거/교체**: 기존 coords 에서 bbox 내 복셀을 부분집합으로 선택
- **추가**: bbox 내부를 복셀로 채워 coords 에 합집합 → **stage-2 가 그 안의 실제 형상을 결정**한다
  (flexible dual grid 디코더는 복셀 안을 비울 수도 있으므로 "채웠다고 반드시 꽉 차지 않는다")

논문의 "stage 1 이 영역 내 새 구조 생성" 을 "복셀을 열어두고 stage 2 가 결정" 으로 대체하는 셈이다.
**이 근사가 논문 결과와 얼마나 다른지는 실측으로 확인해야 한다.**

### 4b. 마스킹 샘플러 (`RegionMaskedFlowEulerSampler`)

rectified flow 는 `x_t = (1-t)·x_0 + t·ε`. 매 스텝 후 **마스크 밖을 노이즈된 원본으로 되돌린다**:

```python
class RegionMaskedFlowEulerSampler(FlowEulerSampler):
    # 원본 x0(=encode_shape_slat 결과)과 고정 ε 를 들고 있는다
    def sample(self, model, noise, ...):
        ...
        for t, t_prev in t_pairs:
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            sample[~mask] = (1 - t_prev) * self.x0_orig[~mask] + t_prev * self.eps[~mask]   # ← 개입
        ...
```
2D 인페인팅과 같은 구조다. 모델은 **전체를 보되**(경계 연속성 확보), 마스크 밖 **결과는 원본으로 고정**.
`SparseTensor` 이므로 마스크는 좌표 기준 불리언 인덱스.

### 4c. 검증

| 항목 | 성공 기준 |
|---|---|
| **마스크 밖 불변성** | 편집영역 밖 latent 변화량이 **정확히 0**. 0 이 아니면 구현 버그 |
| **경계 연속성** | 경계 복셀 근방 법선 불연속 측정. 이음매가 보이면 마스크 확장/블렌딩 필요 |
| **영역 내 변화** | 요청한 게 실제로 생겼는가 — **정성 판정** |
| **비편집 영역 렌더 동일성** | 같은 카메라로 렌더해 픽셀 차이 ≈ 0 |

### 4d. 실패 모드 폴백

Phase 0-1 의 왕복 손실이 크면 → 마스크 밖을 **원본 SLat 이 아니라 원본 메시**로 되돌리는 하이브리드
(영역만 생성 → 디코드 → 원본 메시와 부울 병합). 품질은 떨어지지만 **불변성은 보장**된다.

---

## 5. Phase 3 — 통합 (1~2일)

- 배포 파이프라인 인터페이스 설계: USD 의 어느 필드로 편집 지시를 받을지 (bbox + 지시문)
- `previz_pipeline` 연결은 **이 단계에서 처음** 손댄다
- `prompt_lab` 의 `LiftBackend.local_edit()` 계약 정합:
  현재 `adapters/lift/trellis2.py` 에 `supports_local_edit = True` 로 적혀 있으나 **본문은 `NotImplementedError`
  이고 TRELLIS.2 가 그걸 뒷받침하는 API 를 제공하지 않는다.** `CLAUDE.md` §4-4 의
  "로컬 편집으로 수리 가능(TRELLIS2 지원)" 도 사실과 다르다. → **여기서 바로잡는다.**

---

## 6. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| **`encode_shape_slat` 왕복 손실이 큼** | C 전체 흔들림 | Phase 0-1 에서 먼저 측정. 크면 §4d 하이브리드 |
| `sparse_structure_encoder` 부재 | 구조 추가가 논문 방식과 다름 | §4a 복셀 집합 직접 조작. **논문 대비 차이는 실측 필요** |
| cascade 에서 lr/hr 두 단계 마스킹 필요 | C 복잡도 증가 | Phase 0-3 에서 `pipeline_type` 확정 후 설계 |
| 논문 §3.4 에 하이퍼파라미터 없음 | C 튜닝 시행착오 | 2D 인페인팅 관행(마스크 확장, 경계 블렌딩)에서 출발 |
| **4090 24GB VRAM** | 두 파이프라인(image-to-3D + texturing) 동시 적재 시 OOM | v2 파이프라인의 `low_vram` 플래그 사용. 순차 적재 |

---

## 7. 실행 순서 요약

```
Phase 0 (게이트, 0.5일)  전제 검증 4항목 → C 실현가능성 판정
   ↓
Phase 1 (1~2일)          B: Detail Variation 이식 + 형상 고정도 실측
   ↓
Phase 2 (3~5일)          C: 복셀 마스크 + 마스킹 샘플러 + 불변성 검증
   ↓
Phase 3 (1~2일)          배포 통합 + prompt_lab 문서 정합
```
