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
| nvdiffrast / nvdiffrec | NVIDIA Source Code License (**연구·평가 전용**) | `to_glb` 베이킹에서 제거 — `patches/apply_o_voxel_no_nvdiffrast.sh`, `load_pipeline` 이 자동 재적용. 검증: `experiments/uv_raster_swap/FINDINGS.md` |
| briaai/RMBG-2.0 | CC BY-NC 4.0 (상업은 Bria 유료계약) | `pipeline.json`/`texturing_pipeline.json` 에서 `ZhengPeng7/BiRefNet`(MIT) 로 교체 — `load_pipeline` 이 자동 재교정. 검증: `experiments/rembg_swap/FINDINGS.md` |
| FLUX.1-dev | FLUX.1-dev Non-Commercial License | 진단 용도로만 (`T2I_MODEL_PATH` 수동 지정 시). 배송 기본값은 schnell |
| diff-gaussian-rasterization (3DGS) | INRIA 연구 전용 | TRELLIS **v1** 의존성 — v2 백엔드는 사용하지 않음 |

**프리뷰 렌더 경고**: `trellis2/renderers/*`(턴테이블 mp4/jpg)는 여전히
nvdiffrast/nvdiffrec 를 쓴다. 상업 배포 구성에서는 `--formats` 에 `mp4`/`jpg` 를
넣지 말 것 (GLB 산출과 무관, `want_preview` 게이트로 차단됨).

## ⚖️ 법무 검토 필요 (미해결 리스크)

TRELLIS.2 학습 데이터 중 ABO·HSSD 는 CC BY-NC 다 (Objaverse-XL 은 ODC-By,
TexVerse 는 대부분 CC-BY/CC0). NC 데이터로 학습된 모델의 **출력물**이 상업적으로
오염되는지는 법적으로 확립되지 않은 문제다. Microsoft 는 가중치를 MIT 로 배포했고,
관련 문의(microsoft/TRELLIS.2 issue #22)에 응답하지 않았다. 제품화 전 법무 판단 필요.
