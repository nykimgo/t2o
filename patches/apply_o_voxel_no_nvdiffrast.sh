#!/usr/bin/env bash
# o_voxel 에서 nvdiffrast 의존성을 제거한다.
#
# 왜: o_voxel.postprocess.to_glb 의 텍스처 베이킹이 nvdiffrast 를 쓰는데,
# nvdiffrast 는 NVIDIA Source Code License = 비상업 전용이다. 이 호출이 GLB 산출
# 경로에 남은 마지막 nvdiffrast 의존성이라, 여기만 걷어내면 배송 경로가 깨끗해진다.
# 대체 구현(patches/o_voxel/uv_raster.py)은 의존성 0 추가의 순수 torch 다.
# 검증 근거는 experiments/uv_raster_swap/FINDINGS.md.
#
# trellis2_src/ 는 .gitignore 대상(6.5GB 외부 소스)이라 직접 수정분이 버전관리에
# 안 남는다. 그래서 패치를 리포에 두고 이 스크립트로 입힌다. setup.sh 로 env 를
# 재구축한 뒤에는 반드시 다시 실행해야 한다 — 안 그러면 조용히 nvdiffrast 로 돌아간다.
#
# 사용법:
#   conda activate trellis2
#   ./patches/apply_o_voxel_no_nvdiffrast.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO}/patches/o_voxel"

SITE_PKG="$(python -c 'import o_voxel, os; print(os.path.dirname(o_voxel.__file__))')"
VENDORED="${REPO}/trellis2_src/o-voxel/o_voxel"

echo "🔧 o_voxel nvdiffrast 제거"
echo "   설치본:   ${SITE_PKG}"
echo "   벤더본:   ${VENDORED}"
echo ""

apply_to() {
    local target="$1" label="$2"
    if [ ! -d "${target}" ]; then
        echo "⏭️  ${label}: 없음, 건너뜀 (${target})"
        return 0
    fi

    if grep -q "^from \. import uv_raster" "${target}/postprocess.py"; then
        echo "✅ ${label}: 이미 패치됨"
    else
        # 원본이 예상한 그대로인지 확인하고 나서만 손댄다. 업스트림이 바뀌었으면
        # 조용히 실패하는 대신 멈춘다.
        if ! grep -q "^import nvdiffrast.torch as dr" "${target}/postprocess.py"; then
            echo "❌ ${label}: postprocess.py 가 예상 원본이 아니다. 업스트림 변경 여부 확인 필요."
            return 1
        fi
        cp "${target}/postprocess.py" "${target}/postprocess.py.orig"
        patch -s -p0 --binary "${target}/postprocess.py" < "${PATCH_DIR}/postprocess.py.patch" \
            || { echo "❌ ${label}: 패치 실패"; return 1; }
        echo "✅ ${label}: postprocess.py 패치 (원본은 postprocess.py.orig)"
    fi

    cp "${PATCH_DIR}/uv_raster.py" "${target}/uv_raster.py"
    echo "✅ ${label}: uv_raster.py 설치"
    rm -rf "${target}/__pycache__"
}

apply_to "${SITE_PKG}" "설치본"
apply_to "${VENDORED}" "벤더본"

echo ""
echo "🔍 검증: nvdiffrast 없이 import 되는지"
python - <<'PY'
import sys

# nvdiffrast 가 설치돼 있어도 없는 것처럼 만들어, import 가 정말 끊겼는지 본다.
class _Blocked:
    def find_spec(self, name, path=None, target=None):
        if name == "nvdiffrast" or name.startswith("nvdiffrast."):
            raise ImportError("nvdiffrast is blocked by the commercial-use check")
        return None

sys.meta_path.insert(0, _Blocked())
import o_voxel.postprocess  # noqa: F401
print("✅ nvdiffrast 차단 상태에서 o_voxel.postprocess import 성공")
PY

echo ""
echo "완료. 프리뷰 렌더(trellis2/renderers/*) 는 여전히 nvdiffrast/nvdiffrec 를 쓴다 —"
echo "상업 배포 시에는 --formats 에 mp4/jpg 를 넣지 말 것 (want_preview 게이트)."
