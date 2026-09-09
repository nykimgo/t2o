# 기술이전 최소 코드 매니페스트 — Text→Image→3D (t2o)

> 목적: **Text(USD 프롬프트) → Image(ERNIE) → 3D(TRELLIS.2) → GLB/USD** 파이프라인을
> 외부에 이전할 때 넘겨야 하는 **최소 필수 집합**과, 각 항목의 라이선스 근거를 한 곳에 정리.
> 오픈소스 검증(라이선스 스캔) 대상은 이 문서의 A+B+C 가 전부다.
> 작성: 2026-09-09. 라이선스 상세 근거: `THIRD_PARTY_NOTICES.md`.
>
> **제외**: edit3d(부분편집)·에셋작업실 웹뷰어(미완), experiments/(검증 하네스),
> autorigging/rigging, prompt_lab, TRELLIS **v1**(`trellis/` — 이번 정리에서 실행 경로에서
> 완전히 제거됨, 이전 대상 아님), holopart_src(미사용·**한국에서 라이선스 무허가**, 절대 포함 금지).

## A. 자체 코드 — previz_pipeline/ 12파일 (총 ~4.8k 줄, 688KB)

| 파일 | 줄 | 역할 |
|---|---:|---|
| `usd_parser.py` | 1267 | USD customData 파싱 (pxr) |
| `usd_parse_and_augment.py` | 965 | 1단계: USD→JSON, 선택적 Ollama 캡션 필터 |
| `trellis_inference_core.py` | 454 | 배치/경로/기록 공통 베이스 (v1 실행코드 제거됨) |
| `trellis2_inference_core.py` | ~390 | 2단계 코어: ERNIE→TRELLIS.2→GLB, 상업 배포 가드, 깊이붕괴 게이트 |
| `json_parse_and_inference.py` | 328 | 2단계 orchestrator (trellis2 단일 백엔드) |
| `t2i_prompt_builder.py` | 249 | §9 확정 T2I 프롬프트 템플릿 (rig_type 라우팅) |
| `text_to_image.py` | 134 | ERNIE-Image-Turbo 로더/CLI (서브프로세스 실행) |
| `merge_glb_to_usd.py` | 639 | 3단계: GLB→USD 변환·원본 주입 |
| `glb_to_usd_native.py` | 201 | GLB→USD 네이티브 변환기 (trimesh+pxr, 의존성 0) |
| `bilingual.py` | 77 | ko/en 필드 선택 헬퍼 |
| `t2i_config.py` | 9 | T2I 기본값 단일 정의 |
| `e2e_test.py` | 46 | E2E 스모크 (인수 검증용) |

엔트리: `run_usd_to_3D_object.sh` (3단계 오케스트레이션 + run manifest 기록).

## B. 필수 패치 — patches/ (28KB)

| 파일 | 대상 | 이유 |
|---|---|---|
| `install_o_voxel_commercial.sh` + `o_voxel/` | o_voxel 설치본/벤더본 | 검증·상업 배포용 최종 UV 래스터라이저 파일 적용 |
| `trellis2_transformers5_compat.patch` | trellis2_src | transformers 5.16.1 호환 (DINOv3 `.layer`, BiRefNet FP32) |

## C. 외부 소스/모델 (이전받는 쪽이 clone/다운로드)

| 항목 | 출처 / 고정 버전 | 라이선스 |
|---|---|---|
| TRELLIS.2 소스 | `microsoft/TRELLIS.2` @ `75fbf01` → `trellis2_src/` (+B 패치 2종 적용) | MIT |
| TRELLIS.2-4B 가중치 | `microsoft/TRELLIS.2-4B` (+ v1 `TRELLIS-image-large` 의 ss_dec 체크포인트 — pipeline.json 이 참조) | MIT |
| ERNIE-Image-Turbo (+pe/, pe_tokenizer/) | `baidu/ERNIE-Image-Turbo` @ `bc68c81` | Apache-2.0 |
| DINOv3 ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 License — **제품에 "Built with DINOv3" 표기 의무** |
| BiRefNet | `ZhengPeng7/BiRefNet` | MIT (briaai/RMBG-2.0 은 CC BY-NC 라 교체됨, 가드가 강제) |
| env 재구축 | `ENV_REBUILD_GUIDE.md` + `docs/ERNIE_DEFAULT_SETUP.md` | — |

## D. 파이썬 의존성 (자체 코드가 직접 import 하는 것만)

torch/torchvision(BSD-3) · transformers/diffusers/accelerate/huggingface-hub(Apache-2.0) ·
trimesh(MIT) · usd-core/pxr(Apache-2.0/TomorrowOpen) · numpy/scipy(BSD) · Pillow(MIT-CMU) ·
opencv-python-headless(Apache-2.0) · PyYAML(MIT) · flash-attn(BSD-3) ·
o_voxel/cumesh/flex_gemm(MIT, TRELLIS.2 동봉) · utils3d(MIT)

선택(기본 OFF): Ollama + gemma3(캡션 필터, `--filter` 시에만; Gemma Terms of Use 별도 검토 필요)

## E. 이번 정리에서 제거된 것 (2026-09-09)

- **TRELLIS v1 실행 경로 전체**: base 의 v1 `load_pipeline`/`_generate_single`(208줄)/gaussian·radiance 렌더/`trellis.` import, orchestrator 의 `--backend trellis` 분기·`--simplify`(v1 전용). 748→454줄. → **`trellis/` 디렉토리는 이전 대상에서 제외 가능**해졌고, 3DGS(INRIA 비상업) 흔적이 스캔 범위에서 사라짐
- **프리뷰 렌더 경로 전체**: MP4/JPG 생성과 렌더러 import를 제거하고 GLB/USD만 지원
- 미사용 import/죽은 변수: `subprocess`·`UsdGeom`(merge), `os`(native), `sysconfig`(parser), 죽은 카운터 2개
- 검증: pyflakes 청정(가드형 import 제외) + 전 파일 컴파일 + E2E(ERNIE→TRELLIS.2→GLB) 실측 통과

## ⚠️ 운용 주의 — prompt enhancer 는 빌더 없이 쓰지 말 것

ERNIE prompt enhancer(기본 ON)는 **격리 문구 없는 맨 프롬프트를 받으면 실내 장면
전체를 지어낸다** (E2E 로 재현 확인 — 의자 하나가 러그·탁자·창문 있는 방으로 부풀었다).
프로덕션 경로는 항상 `t2i_prompt_builder.build_t2i_prompt()`(§9)가 single object /
neutral background 문구를 넣으므로 안전하다. **T2I 를 직접 호출할 때도 반드시 빌더를
거쳐라.** e2e_test.py 가 그 올바른 호출 예시다.

## F. 인수 검증 절차 (이전받는 쪽)

```bash
conda activate trellis2   # ENV_REBUILD_GUIDE.md + docs/ERNIE_DEFAULT_SETUP.md 로 구축
cd t2o_pipeline
./patches/install_o_voxel_commercial.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=previz_pipeline:trellis2_src python previz_pipeline/e2e_test.py
./run_usd_to_3D_object.sh <scene.usda> <출력경로>   # 실전 경로
```
