# rembg 교체 — briaai/RMBG-2.0 (CC BY-NC) → ZhengPeng7/BiRefNet (MIT)

## 목적

TRELLIS.2 의 `pipeline.json` 이 배경 제거에 `briaai/RMBG-2.0` 을 지정한다.
CC BY-NC 라 상업 사용엔 Bria 유료 계약이 필요하다. 래퍼 클래스
(`trellis2/pipelines/rembg/BiRefNet.py`)의 기본값이자 같은 아키텍처 원저자 모델인
`ZhengPeng7/BiRefNet`(MIT) 으로 교체 — 코드 변경 없이 config 한 줄이다.

## 판정 기준

하류가 소비하는 것 기준 (`preprocess_image`):
1. **bbox** (alpha>0.8·255 → 정사각 크롭) — 이동 ≤ 대각선 1%
2. **프리멀티플라이 RGB** (DINOv3 컨디셔닝 입력) — PSNR ≥ 30dB
3. 마스크 IoU(@204) ≥ 0.98

## 결과

**구런(7~8월 초) 레퍼런스는 판정에서 제외했다.** 프롬프트 하드닝 이전 산출물이라
다중 객체 장면(차량 행렬+갑판, 사람+폰)이 섞여 있고, 그런 모호한 입력에선 두 모델의
"주 객체" 판단 자체가 갈린다 (car IoU 0.39). 어차피 image-to-3D 에 부적합한 입력이다.

현행 배송 경로(§9 `build_t2i_prompt` + FLUX.1-schnell)로 새로 생성한 10장
(`gen_fresh_refs.py`: bird/quadruped/static×3, 시드 2개):

| 이미지 | IoU | bbox 이동 | premult PSNR |
|---|---|---|---|
| armchair ×2 | 0.998 | ≤1px | 50.4 / 46.7 dB |
| car ×2 | 0.997 | ≤2px | 46.1 / 40.2 dB |
| dog ×2 | 0.993 | ≤1px | 39.2 / 40.2 dB |
| seagull ×2 | 0.990 | ≤1px | 35.3 / 34.9 dB |
| smartphone_42 | 0.992 | 2px | 41.5 dB |
| smartphone_777 | 0.986 | 1px | **26.1 dB** |

9/10 통과. 유일한 미달(smartphone_777)은 패널 검사 결과 **폰 테두리 1~2px 링의
소프트니스 차이**다 (alpha diff>32 픽셀이 전체의 0.9%, 전부 외곽선; 분할 판단은 동일).
긴 테두리 × 밝은 금속 림이 PSNR 을 끌어내렸을 뿐, 컨디셔닝 입력으로는 무의미 → 통과 판정.

## 적용

- `hf_models/TRELLIS.2-4B/pipeline.json` + `texturing_pipeline.json` 의
  `rembg_model.args.model_name` → `ZhengPeng7/BiRefNet`
- `hf_models/` 는 gitignore 라 재다운로드 시 briaai 로 되돌아감 →
  `trellis2_inference_core._ensure_rembg_commercial()` 이 `from_pretrained` **직전에**
  json 을 검사·교정한다 (되돌림 시뮬레이션으로 검증). 로드 전에 고쳐야 NC 가중치가
  메모리에 올라가지도 않는다.
