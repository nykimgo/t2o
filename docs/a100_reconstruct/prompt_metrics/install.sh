#!/usr/bin/env bash
# prompt_metrics env 재구축 (A100). VQAScore 채점 전용, trellis2 핀 보호용 격리 env.
# arch 의존 컴파일 확장 없음(순수 VLM 채점) → A100 sm_80 에서 그대로 동작.
# 4090 실측: python 3.11.15 / torch 2.6.0+cu124 / transformers 4.36.1 / diffusers 0.31.0
set -euo pipefail

ENV_NAME="${1:-prompt_metrics}"

conda create -y -n "$ENV_NAME" python=3.11
# 아래는 해당 env 의 pip 로 실행 (conda run 사용)
run() { conda run -n "$ENV_NAME" "$@"; }

# torch 는 cu124 인덱스에서 (t2v-metrics 채점은 arch 무관)
run pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ★함정: image-reward 가 OpenAI 'clip' 패키지를 요구 → PyPI clip 아님, git 설치 필수
run pip install "git+https://github.com/openai/CLIP.git"

# 채점 스택. diffusers 는 image-reward 가 끌어오지만 채점엔 미사용(ReFL만).
# 0.31.0 이 transformers 4.36.1 + hf_hub 0.36 과 호환.
run pip install \
  transformers==4.36.1 \
  diffusers==0.31.0 \
  t2v-metrics==1.2 \
  image-reward==1.5 \
  open-clip-torch==2.32.0 \
  accelerate==1.14.0 \
  sentencepiece==0.2.2

echo
echo "[OK] env '$ENV_NAME' 생성 완료."
echo "  - VQAScore VLM(clip-flant5-xl, ~4GB)은 첫 채점 실행 시 자동 다운로드."
echo "  - ImageReward.pt(~2GB)도 첫 실행 시 자동 다운(쓸 때만)."
echo "  - 세부 핀 대조: pip_freeze.reference.txt (4090 라이브 119개 전량)."
echo "  - prompt_lab 에서 이 env 를 VQA 데몬으로 호출 (subprocess). 병렬 워커 밖(메인)에서만 기동(Phase2)."
