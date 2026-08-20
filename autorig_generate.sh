#!/usr/bin/env bash
# AutoRigging for previs_proj (USD 스캔 → UniRig 리깅 → 기본모션 → 규칙 USD 추가).
# Usage:
#   ./autorig_generate.sh movie_usd/welcom2dmk/welcom2dmk.usda
#   ./autorig_generate.sh movie_usd/welcom2dmk/scene_1/scene_1.usda --dry-run
#   ./autorig_generate.sh movie_usd/welcom2dmk/welcom2dmk.usda --only 동구 --no-motion
set -euo pipefail

# 이 스크립트는 t2o_pipeline/ 안에 둔다. previs_proj 루트의 동명 런처와 동일 동작.
T2O_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREVIS_PROJ="$(cd "${T2O_DIR}/.." && pwd)"
AUTORIG_DIR="${T2O_DIR}/autorigging"

# unirig env 파이썬 (README/DESIGN §5 — bpy 4.2 + usd-core + torch 2.6 cu124)
UNIRIG_PYTHON="${UNIRIG_PYTHON:-/root/miniconda3/envs/unirig/bin/python}"
# UniRig 저장소 (로컬 패치 2건 적용본 — 재clone 시 패치 재적용 필수)
export UNIRIG_DIR="${UNIRIG_DIR:-${T2O_DIR}/autorigging_transfer/unirig}"
# 리깅 GPU 1장 — FLUX·TRELLIS 와 동시 실행 금지 (RAM OOM 실측, DESIGN §5)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'EOF'
Usage: autorig_generate.sh <usd_file> [--dry-run] [--only <이름부분일치>] [--no-motion]

Arguments:
  <usd_file>      영화 usda (movie_usd/<영화>/<영화>.usda) 또는 씬 usda
  --dry-run       스캔 결과만 출력 (리깅/모션 미실행)
  --only <str>    이름 부분일치 대상만 처리
  --no-motion     리깅(FBX)까지만, 기본모션/USD 조립 생략

Environment variables:
  UNIRIG_PYTHON          unirig env 파이썬 (기본: /root/miniconda3/envs/unirig/bin/python)
  UNIRIG_DIR             UniRig 저장소 (기본: t2o_pipeline/autorigging_transfer/unirig)
  CUDA_VISIBLE_DEVICES   리깅 GPU (기본: 0)
  AUTORIG_OUT            리포트 출력 루트 (기본: ${PREVIS_PROJ}/autorig_out)

산출물 (원본 트리에 추가만 — 기존 파일 무수정):
  assets/<이름>/rigs/<이름>_{skeleton,rigged}.fbx    리깅 원본 (다음 파트 참조용)
  assets/<이름>/<이름>.rt_default.usda               기본모션 레이어
  <원본 usda 옆>/<이름>.rt_default.usda              래퍼 (autorig customData 포함)
  autorig_out/<run_id>/                              매니페스트·리포트·로그

⚠️ FLUX/TRELLIS 등 대형 모델과 동시 실행 금지 (RAM OOM 실측 — 단계는 직렬로).
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

USD_FILE="$1"
shift

if [[ "${USD_FILE}" != /* ]]; then
  USD_FILE="${PREVIS_PROJ}/${USD_FILE}"
fi
USD_FILE="$(readlink -f "${USD_FILE}")"

if [[ ! -f "${USD_FILE}" ]]; then
  echo "❌ USD 파일을 찾을 수 없습니다: ${USD_FILE}"
  exit 1
fi
if [[ ! -x "${UNIRIG_PYTHON}" ]]; then
  echo "❌ unirig 파이썬을 찾을 수 없습니다: ${UNIRIG_PYTHON}"
  echo "   t2o_pipeline/autorigging_transfer/README.md §2 로 env 를 구축하세요."
  exit 1
fi

# 이 서버 전역 LD_LIBRARY_PATH(/root/previs_proj/usd/lib)가 bpy 번들 MaterialX 와
# libcuda 를 가려 ImportError/CUDA 실패를 일으킨다 (실측) — 반드시 제거하고 실행.
unset LD_LIBRARY_PATH

OUTPUT_ROOT="${AUTORIG_OUT:-${PREVIS_PROJ}/autorig_out}"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
ln -sfn "${RUN_ID}" "${OUTPUT_ROOT}/latest"

echo "🦴 AutoRigging 시작"
echo "📁 입력 USD : ${USD_FILE}"
echo "📁 Run 디렉토리: ${RUN_DIR}"
echo "🤖 UniRig    : ${UNIRIG_DIR} (GPU ${CUDA_VISIBLE_DEVICES})"

# bpy 는 파이썬 종료 시 segfault 를 낼 수 있다(teardown, 산출물 무해 — 실측).
# 성공 판정은 exit code 가 아니라 리포트/로그 기반으로 한다.
set +e
"${UNIRIG_PYTHON}" "${AUTORIG_DIR}/autorig.py" "${USD_FILE}" --run-dir "${RUN_DIR}" "$@"
RC=$?
set -e

if [[ -f "${RUN_DIR}/report.json" ]]; then
  echo ""
  echo "📄 리포트: ${RUN_DIR}/report.md"
  if [[ ${RC} -ne 0 ]]; then
    # 리포트가 있으면 오케스트레이터가 정상 완주한 것 — 원본 훼손(1)만 하드 실패.
    ORIG_PASS=$("${UNIRIG_PYTHON}" - "${RUN_DIR}/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
o = r.get("originals_unchanged") or {}
print("1" if o.get("pass") else "0")
PY
)
    if [[ "${ORIG_PASS}" == "0" ]]; then
      echo "❌ 원본 불변 검증 실패 — report.md 를 확인하세요."
      exit 1
    fi
  fi
  echo "✅ AutoRigging 완료"
  exit 0
fi

# 리포트조차 없으면 (dry-run 포함이 아닌 한) 실행 실패
if [[ -f "${RUN_DIR}/autorig_manifest.json" ]]; then
  echo "✅ 스캔 완료 (dry-run 또는 조기 종료) — ${RUN_DIR}/autorig_manifest.json"
  exit ${RC}
fi
echo "❌ AutoRigging 실패 (exit ${RC}) — ${RUN_DIR}/autorig.log 확인"
exit ${RC}
