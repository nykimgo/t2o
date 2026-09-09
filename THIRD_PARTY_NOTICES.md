# Third-Party Notices — 상업 배포 의무사항

이 파이프라인(t2o: USD → FLUX → TRELLIS.2 → GLB/USD)을 **상업 제품에 포함할 때
지켜야 하는** 서드파티 라이선스 의무를 정리한다. 마지막 검증: 2026-09-08.

## ⚠️ 제품에 반드시 반영할 것

### 1. "Built with DINOv3" 표기 (의무)
이미지 컨디셔닝에 `facebook/dinov3-vitl16-pretrain-lvd1689m` 을 사용한다
(TRELLIS.2 pipeline.json 의 `image_cond_model`). [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) 는
상업 사용을 허용하되 다음을 요구한다:

- 관련 **웹사이트·UI·블로그·소개 페이지·문서에 "Built with DINOv3" 를 눈에 띄게 표기**
- DINOv3 모델(또는 파생물)을 재배포하면 라이선스 계약서 사본 동봉
- 군사·핵·스파이 용도 금지, 수출규제 준수

MAU 임계값이나 Llama 식 추가 상업 조항은 없다. **제품 크레딧/약관 페이지에
"Built with DINOv3" 한 줄이 계약 요건이다 — 빠뜨리면 라이선스 위반.**

### 2. MIT/BSD/Apache 고지문 동봉
배포물(설치 패키지·약관·오픈소스 고지 페이지)에 아래 컴포넌트의 저작권 고지와
라이선스 전문을 포함할 것:

| 컴포넌트 | 라이선스 | 비고 |
|---|---|---|
| TRELLIS.2 (코드+가중치 4B) | MIT © Microsoft | |
| o-voxel / CuMesh / FlexGEMM | MIT © Jianfeng Xiang | TRELLIS.2 커스텀 CUDA 확장 |
| FLUX.1-schnell | Apache-2.0 © Black Forest Labs | **dev 아님** — dev 는 비상업 |
| BiRefNet (ZhengPeng7/BiRefNet) | MIT © ZhengPeng7 | 배경 제거 |
| utils3d | MIT © EasternJournalist | |
| flash-attention | BSD-3-Clause | |
| trimesh / usd-core(pxr) 등 | MIT / Apache-2.0 | GLB→USD 네이티브 변환 경로 |

## 🚫 제거된 비상업 컴포넌트 (되돌리면 안 됨)

| 컴포넌트 | 라이선스 | 상태 |
|---|---|---|
| briaai/RMBG-2.0 | CC BY-NC 4.0 (상업은 Bria 유료계약) | `pipeline.json`/`texturing_pipeline.json` 에서 `ZhengPeng7/BiRefNet`(MIT) 로 교체 — `load_pipeline` 이 자동 재교정. 검증: `experiments/rembg_swap/FINDINGS.md` |
| FLUX.1-dev | FLUX.1-dev Non-Commercial License | 진단 용도로만 (`T2I_MODEL_PATH` 수동 지정 시). 배송 기본값은 schnell |
| diff-gaussian-rasterization (3DGS) | INRIA 연구 전용 | TRELLIS **v1** 의존성 — v2 백엔드는 사용하지 않음 |

## 🎛️ edit3d (부분편집) 의존성 — 기술이전 시 추가 확인 사항

edit3d 웹UI/worker 스택 감사 결과 (2026-09-08):

### 문제 없음 (Apache-2.0 / MIT)
| 컴포넌트 | 라이선스 | 용도 |
|---|---|---|
| SAM 2.1 (코드+`facebook/sam2.1-hiera-small`) | Apache-2.0 | 마스크 미리보기 |
| `Qwen/Qwen-Image-Edit` (20B) | Apache-2.0 (MAU 제한 없음) | 2D 편집 |
| `CIDAS/clipseg-rd64-refined` | Apache-2.0 | 텍스트 기반 영역 지정 |
| UniRig (코드+가중치, VAST-AI) | MIT | 리깅 (`rig_service.py`) |
| Nano3D (JAMESYJL/Nano3D) | MIT | `ss_flowedit.py` 가 샘플링 로직 파생 — **파일 내 저작권 고지 유지 필수** (추가됨) |
| trimesh/open3d/scipy/flask/skimage 등 | MIT/BSD | |

### GPL — 서브프로세스 격리로 사용 중 (유지 조건 필수)
| 컴포넌트 | 라이선스 | 격리 상태 |
|---|---|---|
| CGAL 불리언 (`igl.copyleft.cgal`) | **GPL-3.0** (GeometryFactory 상업 라이선스 별매) | `replace_cgal_union.py`·`p5_causal_guard.py` 를 별도 python 으로 subprocess 호출, npz 파일 I/O — FSF 해석상 "별개 프로그램" |
| Blender (`bpy`) | **GPL-3.0** | `canonical_views.py` 를 unirig env python 으로 subprocess 호출 |

**기술이전 시 의무**: (a) 이 격리 구조(in-process import 금지)를 문서로 못박아 전달할 것 —
이전받은 쪽이 편의상 in-process 로 합치면 그쪽 코드가 GPL 오염된다. (b) GPL 컴포넌트의
라이선스 전문·소스 제공 의무는 해당 스크립트/바이너리에만 적용. (c) GPL 을 아예 없애려면:
CGAL 불리언 → **manifold3d(Apache-2.0)** 로 교체 가능, bpy 렌더 → pyrender/moderngl 로 교체 가능.
격리 유지가 부담스러우면 교체가 깔끔하다. 참고로 igl 코어(`import igl`)는 MPL-2.0 이라 무방 —
`igl.copyleft.*` 만 GPL 이다.

### 🚨 HoloPart — 한국에서 사용 불가 (현재 미사용, 유지할 것)
`holopart_src/` 는 vendored 만 되어 있고 **호출 코드 없음 + git 미추적** (repair_data
문서에 "보류" 로만 언급). 절대 활성화하지 말 것:

- 저장소 라벨은 MIT 이지만 NOTICE 상 **Tencent FlashVDM 파생 코드**(vecset 디코더) 포함
- FlashVDM Community License 의 Territory 가 **EU·영국·대한민국을 명시 제외** —
  한국에서는 상업/비상업 구분 없이 **사용 자체가 무허가**. MIT 라벨은 Tencent 파생
  부분을 재라이선스하지 못한다
- 허용 지역에서도 MAU 1M 제한
- 부품 완성 기능이 필요해지면: 디코더 교체 or Tencent 별도 허가 or 대체 모델 검토

## ⚖️ 법무 검토 필요 (미해결 리스크)

TRELLIS.2 학습 데이터 중 ABO·HSSD 는 CC BY-NC 다 (Objaverse-XL 은 ODC-By,
TexVerse 는 대부분 CC-BY/CC0). NC 데이터로 학습된 모델의 **출력물**이 상업적으로
오염되는지는 법적으로 확립되지 않은 문제다. Microsoft 는 가중치를 MIT 로 배포했고,
관련 문의(microsoft/TRELLIS.2 issue #22)에 응답하지 않았다. 제품화 전 법무 판단 필요.
