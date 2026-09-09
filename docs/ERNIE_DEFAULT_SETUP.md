# ERNIE 기본 모델 및 trellis2 환경 통합

2026-09-08부터 production T2I 기본값은 `baidu/ERNIE-Image-Turbo`다. 현재 설정은 **BF16 / 8 steps / guidance 1.0 / 1024×1024 / prompt enhancer ON**이다. 사용자의 후속 품질 확인에 따라 자동 보강을 기본 ON으로 변경했다. 4090에서는 model CPU offload와 VAE tiling을 사용하며, 기존 `t2i_prompt_builder.py`의 가이드를 공식 ERNIE prompt enhancer에 전달한 뒤 보강 결과로 이미지를 생성한다.

## 적용된 호출 경로

- `previz_pipeline/t2i_config.py`: 기본 모델 경로·steps·guidance의 공통 정의.
- `previz_pipeline/text_to_image.py`: 실제 ERNIE 로더와 CLI. FLUX 전용 파라미터·attention slicing은 사용하지 않는다.
- `previz_pipeline/trellis2_inference_core.py`, `json_parse_and_inference.py`, `e2e_test.py`: 공통 ERNIE 기본 경로 사용.
- Python 기본값은 `sys.executable`이다. `trellis2`에서 메인 파이프라인을 실행하면 T2I도 같은 conda 환경을 사용한다. 별도 실험 venv가 필요하지 않다.
- `T2I_MODEL_PATH` 또는 기존 `--t2i_model_path`/`--model`은 **ERNIE 체크포인트 경로**를 덮어쓰는 용도다. 이 인자에 Qwen이나 FLUX 경로를 주는 것으로 모델 종류가 전환되지는 않는다. Qwen 선택 옵션은 후속 계획이다.

가중치 경로는 `t2o_pipeline/hf_models/ERNIE-Image-Turbo`다. 현재 서버에서는 기존 다운로드를 재사용하는 상대 심볼릭 링크(`../../.runtime/models/ernie`)다. 다른 서버로 옮길 때는 링크만 복사하지 말고 실제 가중치도 배치한다. 원본 revision은 `bc68c81e2a1730a394d5fc9fae70713dee940140`이며 `.runtime/manifests/ernie.json`에 다운로드 목록이 있다.

새 서버에서는 `t2o_pipeline` 디렉터리에서 다음처럼 받을 수 있다.

```python
from huggingface_hub import snapshot_download
snapshot_download('baidu/ERNIE-Image-Turbo',
                  revision='bc68c81e2a1730a394d5fc9fae70713dee940140',
                  local_dir='hf_models/ERNIE-Image-Turbo')
```

자동 보강용 `pe/`와 `pe_tokenizer/`도 필수다(추가 약 7.7GB). 현재 서버에는 다운로드 완료했다. 누락되면 조용히 OFF로 실행하지 않고 오류를 낸다. `T2I_PROMPT_ENHANCER=1`이 기본이며, 명시적인 진단에만 `--no-prompt-enhancer` 또는 `T2I_PROMPT_ENHANCER=0`을 사용한다. 출력 PNG 옆 JSON에는 원본·보강 프롬프트, seed와 생성 설정을 저장한다. PE의 샘플링 RNG도 선택한 seed로 고정한다.

## 단일 conda 환경

현재 환경: `/home/sr/miniconda3/envs/trellis2` (Python 3.10).

| 구성 | 적용 버전 |
|---|---|
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 (변경 없음) |
| diffusers / accelerate | 0.39.0 / 1.14.0 (변경 없음) |
| transformers | 5.16.1 |
| huggingface_hub | 1.30.0 |
| tokenizers | 0.23.2 |
| peft / bitsandbytes | 0.20.0 / 0.50.2 (Qwen INT8 실험도 같은 환경에서 실행) |
| hf-xet | 1.5.2 (서버 기존 설치값; 재구축 파일에도 반영) |

환경 통합에는 아래 **TRELLIS 호환 패치 두 개가 필수**다. Transformers만 올리고 패치를 빠뜨리면 기존 3D 전처리가 깨진다.

1. `trellis2_src/trellis2/modules/image_feature_extractor.py`: 4.x의 `model.layer`와 5.x의 `model.model.layer`를 모두 지원한다. 마지막 학습 LayerNorm 이전 특징에 기존 `F.layer_norm`을 적용하는 계산은 유지한다.
2. `trellis2_src/trellis2/pipelines/rembg/BiRefNet.py`: 로딩 dtype을 명시적으로 FP32로 고정한다. Transformers 5의 체크포인트 dtype 자동 추론으로 FP16이 로딩되어 FP32 입력과 충돌하는 것을 방지하며, 이전 계산 정밀도를 유지한다.

두 변경은 현재 소스에 적용했고, 새 checkout용 패치도 `patches/trellis2_transformers5_compat.patch`에 저장했다. 이미 적용된 서버에서는 다시 적용하지 않는다. 새 checkout에서, `t2o_pipeline` 기준:

```bash
git -C trellis2_src apply --check ../patches/trellis2_transformers5_compat.patch
git -C trellis2_src apply ../patches/trellis2_transformers5_compat.patch
conda activate trellis2
python -m pip install transformers==5.16.1 huggingface_hub==1.30.0 tokenizers==0.23.2
python -m pip install peft==0.20.0 bitsandbytes==0.50.2
python -m pip check
```

`environment-trellis2.yml`과 `requirements-trellis2.lock`의 관련 핀도 갱신했다. 이 파일들의 다른 패키지는 기존 값이므로 기존 서버별 재구축 주의사항도 함께 확인한다. 이전 `ENV_REBUILD_GUIDE.md`의 4.56.2 고정·5.x 금지 설명은 **이 호환 패치가 없던 FLUX 구성의 기록**이며, 이번 ERNIE 구성에는 본 문서를 적용한다.

## 실행 및 확인

작업공간 루트에서, 비어 있는 물리 GPU 번호를 지정한다.

```bash
conda activate trellis2
CUDA_VISIBLE_DEVICES=1 python t2o_pipeline/previz_pipeline/text_to_image.py \
  --prompt 'A wooden chair, single object, neutral background, full object visible.' \
  --out /tmp/ernie-chair.png --seed 101
```

이미 TRELLIS.2를 GPU에 로딩한 메인 경로에서는 `T2I_GPU`로 T2I용 별도 카드를 지정한다. 4090 한 장에서 전체 작업을 수행하려면 T2I와 3D 단계를 순차 실행해 메모리를 반환해야 한다. **conda 환경 통합과 모델의 동시 GPU 상주는 별개**다. 이번 변경은 기존 subprocess 메모리 수명 관리를 유지한다.

환경 통합 검증 결과(마지막 항목은 자동 보강 OFF였던 초기 검증 기록):

- 설치 후 `pip check` 통과. torch/CUDA 버전 유지.
- Transformers 4.56.2에서 저장한 작은 DINOv3 모델·고정 입력을 5.16.1에서 읽어 특징 비교: 최대 오차 **0.0**.
- 실제 캐시된 DINOv3 가중치로 TRELLIS 특징 추출 성공: `(1, 1029, 1024)`, 모든 값 유한.
- 실제 BiRefNet FP32 배경 제거 성공: 1024×1024 RGBA, alpha 범위 0–255.
- flash_attn / flex_gemm / cumesh / o_voxel import 성공.
- 메인 `Trellis2InferenceCore._t2i_generate()`가 같은 `trellis2` Python과 기본 ERNIE 경로로 1024² RGB 생성 성공. 로딩·subprocess 시작·저장 포함 **42.1초**. 동일 seed 101의 기존 ERNIE 비교 샘플과 **픽셀 단위 동일**.

자동 보강 ON 추가 검증: 메인 `_t2i_generate()`의 기본 호출로 1024² 생성 및 보강 프롬프트 JSON 저장 성공. 같은 seed의 실험 ON 이미지와 픽셀 단위 동일. subprocess 시작·모델 로딩·PE·생성·저장 포함 **58.3초**였다(다른 GPU 비교 작업과 병행한 서버 측정).

검증 기록·이미지는 `.runtime/ernie-integration/`에 있다. 전체 3D 재구성 배치는 실행하지 않았다. 현재 자동 보강 ON 및 Qwen INT8 품질 비교는 `experiments/t2i_quality/outputs/quantization_v1/`에 별도 보존하며, 원래 OFF 결과는 덮어쓰지 않는다.

모델 채택 결정과 Qwen 양자화 현황: [T2I_MODEL_DECISION_2026-09-08.md](T2I_MODEL_DECISION_2026-09-08.md).
