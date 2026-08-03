#!/usr/bin/env bash
# config 의 stale FLUX 경로 치환. dev(4090) /home/sr/previs_proj → A100 target.
# A100 repo 루트에서 실행. TARGET 을 실제 경로로 확인/수정할 것 (기본: /root/previs_proj).
set -euo pipefail

TARGET="${1:-/root/previs_proj}"
DEV="/home/sr/previs_proj"

echo "치환 대상 파일 (치환 전 검사):"
grep -rln "$DEV/" prompt_lab/configs/ || { echo "  (해당 경로 없음 — 이미 수정됨?)"; exit 0; }

sed -i "s#${DEV}/#${TARGET}/#g" prompt_lab/configs/*.yaml

echo
echo "[OK] '$DEV/' → '$TARGET/' 치환 완료. 남은 stale 경로 확인:"
grep -rn "$DEV/" prompt_lab/configs/ && echo "⚠️ 아직 남음!" || echo "  (없음 — 정상)"
echo
echo "확인: FLUX 가중치가 실제로 그 경로에 있는지 (du -sh \$TARGET/t2o_pipeline/hf_models/FLUX.1-schnell → 54G)"
