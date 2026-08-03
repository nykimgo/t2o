# 4090 개발서버 복귀 셋팅 — A100 에서 달라진 점과 되돌릴 것

> **작성**: 2026-08-04 (A100 공유서버 세션 실측). **독자**: 4090 개발서버에서 `edit3d` 작업을 이어받는 사람.
> **관계**: `A100_HANDOFF.md` 는 **4090 → A100** 방향 문서다. 이 문서는 그 **역방향**이며,
> A100 에서 진행된 작업(2026-07-28 ~ 08-04)이 4090 에서 돌게 하려면 무엇을 확인/조정해야 하는지를 적는다.
> 계획 문서: `../edit3d/docs/PLAN.md`

---

## 0. 30초 요약

- A100 세션의 **코드 변경은 전부 하드웨어 중립**이다. 4090 에서 그대로 돈다.
- 다만 **하드코딩된 경로 1건**, **성능 특성 1건**, **누락 패키지 1건**을 확인해야 한다.
- `prompt_lab` 은 **종료된 실험**이며 repo 에 포함하지 않는다(§4). 4090 에 남아 있는 사본이 정본이다.

---

## 1. A100 환경 실측치 (4090 대조용)

| 항목 | 4090 개발서버 | **A100 공유서버 (이번 세션)** |
|---|---|---|
| GPU | RTX 4090 24GB × 4, **sm_89** | A100 80GB × 4 + RTX 3050 6GB, **sm_80** |
| 프로젝트 경로 | `/home/sr/previs_proj/` | `/root/previs_proj/` (도커 `previs-prep` 내부) |
| conda 루트 | `/home/sr/miniconda3` | `/root/miniconda3` |
| env | `trellis2`, `prompt_metrics` | 동일 (+ `t2i`, `trellis`, `previz`) |
| torch | 2.6.0+cu124 | 2.6.0+cu124 (동일) |
| diffusers / flash_attn / nvdiffrast / trimesh | 0.39.0 / 2.7.3 / 0.3.3 / 4.12.2 | **동일** |
| ollama | 로컬 `localhost:11434` (스크립트가 자동 기동) | **호스트 ollama, `http://172.17.0.1:11434`** (도커 브리지) |
| FLUX 적재 | 24GB < 40GB 임계 → **CPU offload** | 80GB ≥ 40GB → **상주** |
| T2I 처리량 | **장당 25~41초** (offload) | **장당 2.5초** (상주, 실측) |

---

## 2. 4090 에서 확인/조정할 것

### 2-1. ⚠️ `prompt_lab` 의 하드코딩 경로 (해당 사본을 쓸 때만)

`prompt_lab/previs_lab/adapters/metrics/vqa_remote.py`:
```python
_DEFAULT_PY = "/home/sr/miniconda3/envs/prompt_metrics/bin/python"
```
**4090 기준 경로라 4090 에서는 정상 동작한다.** A100 세션에서는 코드를 고치지 않고
config 의 `metrics[].params.python_bin` 으로 덮어썼다. 4090 으로 돌아오면 기본값이 다시 맞다.
→ **조치 불필요.** 다만 세 번째 서버로 옮길 땐 config override 방식을 쓸 것.

### 2-2. ⚠️ `open3d` 미설치 — `edit3d` Phase 0 에 직접 영향

A100 의 `trellis2` env 에 **`open3d` 가 없다**(실측). TRELLIS v1 의 복셀화 경로
(`o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds`)를 쓸 수 없다.

**4090 에서 먼저 확인할 것:**
```bash
conda activate trellis2
python -c "import open3d; print(open3d.__version__)"
```
- 있으면: v1 방식도 선택지에 들어온다
- 없으면: `PLAN.md` §2 대로 `o_voxel.convert.mesh_to_flexible_dual_grid` 경로로 간다 (**권장 — 기존 의존성만 사용**)

> `edit3d` 는 **의도적으로 open3d 에 의존하지 않게** 설계했다. 설치하지 않아도 된다.

### 2-3. VRAM 24GB — `edit3d` 의 실질 제약

`edit3d` 는 상황에 따라 파이프라인 **두 개**를 쓴다:
- `Trellis2ImageTo3DPipeline` (생성/variant)
- `Trellis2TexturingPipeline` (`encode_shape_slat` — C 의 전제)

**24GB 에 동시 적재하면 OOM 위험이 크다.** 대응:
- v2 파이프라인의 **`low_vram` 플래그**를 켠다 (`if self.low_vram:` 분기가 모델을 스텝마다 GPU↔CPU 로 옮긴다)
- 또는 **순차 적재**: encode 끝내고 파이프라인을 내린 뒤 생성 파이프라인을 올린다
- A100 에서는 이 제약이 없어 **검증되지 않은 경로**다. 4090 에서 처음 밟게 되므로 초기에 확인할 것

### 2-4. 컴파일 확장 arch

A100 env 는 **sm_80** 으로 빌드됐다(`flash_attn`, `o_voxel`, `flex_gemm`, `cumesh` 등).
4090 은 **sm_89** 다. `A100_HANDOFF.md` §2 가 이미 지적한 항목이며 **방향만 반대**다.
4090 에 기존 env 가 남아 있으면 그건 원래 sm_89 빌드이므로 **그대로 쓰면 된다.**
새로 만들 경우에만 `TORCH_CUDA_ARCH_LIST=8.9` 로 재빌드.

### 2-5. ollama 엔드포인트

A100 은 도커 브리지 게이트웨이(`172.17.0.1:11434`)로 호스트 ollama 에 붙었다.
4090 은 로컬 실행이므로 `object_generate.sh` 가 **자동 기동**한다(`OLLAMA_BASE_URL` 미지정 시).
→ **조치 불필요.** A100 세션에서 쓰던 `OLLAMA_BASE_URL=http://172.17.0.1:11434` 는 4090 에서 **지울 것.**

### 2-6. GPU 선택 관행

A100 은 공유 서버라 동료와 겹치지 않게 **GPU 2번 단독**(`CUDA_VISIBLE_DEVICES=2`)으로 돌렸다.
4090 전용 서버면 4장을 다 쓸 수 있다. 스크립트에 박힌 `CUDA_VISIBLE_DEVICES=2` 가 있으면 **해제할 것.**
(`prompt_lab/run_*.sh` 에 들어 있다 — 해당 사본을 쓸 때만 해당)

---

## 3. A100 세션에서 만든 코드 변경 (하드웨어 중립 — 그대로 동작)

이번 push 에 포함된 것들. 전부 GPU/경로 비의존이다.

| 파일 | 변경 | 비고 |
|---|---|---|
| `previz_pipeline/usd_parser.py` | `rig_type` 탐색 순서를 **최상위 `string rig_type` 우선**으로. 구 `etc.rig_type` 은 하위호환 | 의도분석 모듈 실제 산출 형태에 맞춤 |
| `previz_pipeline/usd_parse_and_augment.py` | ① `appearance` 를 essential 에 복사(§ `{appearance}` 슬롯이 항상 비던 버그) ② 필터 입력에 `appearance` 추가 + 필터 ON 이면 raw appearance 미사용 ③ 필터 JSON 파싱 실패 시 **경고 3중 출력** ④ 단계별 `[TIME]` 계측 | ③ 은 실패가 조용히 넘어가던 문제 |
| `previz_pipeline/step2_filter_prompt.txt` | 입력 필드에 `appearance` 명시 + 맥락 제거/디테일 보존 예시 2개 추가 | "held in hand" 가 프롬프트에 남아 손이 생성되던 문제 |
| `previz_pipeline/trellis2_inference_core.py` | **T2I / I2O 분리 계측** + 객체별 합계 로그. `t2i_time`/`i2o_time` 을 결과 CSV·JSON 에 기록 | 기존엔 `generation_time` 하나로 합쳐져 구분 불가 |
| `previz_pipeline/trellis_inference_core.py` | 2단계 종료 시 소요시간 요약 | |
| `previz_pipeline/t2i_prompt_builder.py` | **신규 추적** (기존 미추적). §9 확정 시스템 프롬프트 = **v1 동결 대상** | 이번 캠페인 결론: 변경 근거 없음 |
| `docs/RUN_GUIDE.md` | `rig_type` 정본을 최상위 형태로 갱신 | |
| `docs/` 전체 | 신규 추적 (A100_HANDOFF / FILE_TRANSFER_MANIFEST / a100_reconstruct 등) | |

> ⚠️ **`object_generate.sh` 는 이 repo 밖(`previs_proj/` 최상위)에 있어 push 에 포함되지 않는다.**
> A100 에서 추가한 3단계 wall-clock 계측(`⏱️ 단계별 소요 시간`)이 4090 에는 없다. 필요하면 별도 이관.

---

## 4. `prompt_lab` 취급

- **종료된 실험**이다. 시스템 프롬프트 v1 이 확정됐고(`EXPERIMENT_LOG.md` §10.10), 더 나은 템플릿을
  고를 근거가 없다는 것이 결론이다.
- **이 repo 에 포함하지 않는다**(`.gitignore` 에 등재). 4090 에 있는 사본이 정본이다.
- A100 세션에서 `prompt_lab` 에 추가된 것(참고용, 옮기려면 수동 복사):
  - `previs_lab/template_search.py`, `loops/template.py` — 접근 B(LLM 고정 템플릿 탐색) 구현
  - `tests/test_template_search.py` — 22개 (전체 206 passed)
  - `docs/EXPERIMENT_LOG.md` §10 — 이번 캠페인 전 기록(목적함수 게이밍 발견 포함)
  - `configs/*.yaml` — 실험 config 6종
  - `runs/` — 원점수 보존소. **재집계에 필요하면 같이 옮길 것** (수십 GB 아님, 이미지 포함 시 커짐)

---

## 5. 4090 복귀 체크리스트

```bash
# 1) repo 최신화
cd <4090>/previs_proj/t2o_pipeline
git fetch origin && git checkout feature/trellis2 && git pull

# 2) env 확인 (재구축 불필요, 확인만)
conda activate trellis2
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"   # (8,9) 기대
python -c "import diffusers, flash_attn, nvdiffrast, trimesh; print('ok')"
python -c "import o_voxel; print('o_voxel ok')"
python -c "import open3d" 2>/dev/null && echo "open3d 있음" || echo "open3d 없음(무방)"

# 3) A100 전용 설정 해제
#    - OLLAMA_BASE_URL / OLLAMA_HOST 환경변수 제거
#    - 스크립트의 CUDA_VISIBLE_DEVICES=2 제거

# 4) 파이프라인 스모크 (1객체)
T2I_GPU=0 ./object_generate.sh movie_usd/<movie>/<movie>.usda --no-filter -- --max_items 1
#    기대: offload 경로, 장당 25~41초. 상주(2.5초)는 A100 기준이므로 여기선 안 나온다.

# 5) edit3d Phase 0 착수 → edit3d/docs/PLAN.md §2
```

---

## 6. 알려진 이슈 (A100 에서 발견, 4090 에도 해당)

1. **카메라 USD 10개가 파싱 실패** — `customData` 가 레이어 메타데이터 블록에 들어가 있다.
   USD 는 거기서 `customData` 를 허용하지 않는다(`customLayerData` 만). shot_1~10 카메라가
   스테이지에 로드되지 않는다. object 생성엔 무해하나 렌더/카메라 사용 시 터진다. **미수정.**
2. **CV 게이트 미보정** — `prompt_lab` 의 CV 임계값이 mock 튜닝 값이라 실이미지에서 **좋은 이미지를 거부**한다
   (이상적인 헬멧·권총 이미지가 `valid=0.0`). `CLAUDE.md` §6 이 지시한 O/X → `--calibrate` 단계가 미수행.
3. **목적함수 사각지대 3종** — 중복 객체 / 객체 정체성 / 시점. `EXPERIMENT_LOG.md` §10.9 결론 참조.
   프롬프트 최적화로는 해결 불가이며 지표를 고쳐야 한다.
