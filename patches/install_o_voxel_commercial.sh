#!/usr/bin/env bash
# 검증·상업 배포용 o_voxel 파일을 설치본과 TRELLIS.2 벤더본에 적용한다.
# 환경을 새로 만들거나 o_voxel을 다시 설치한 뒤 실행한다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERRIDE_DIR="${REPO}/patches/o_voxel"
SITE_PKG="$(python -c 'import o_voxel, os; print(os.path.dirname(o_voxel.__file__))')"
VENDORED="${REPO}/trellis2_src/o-voxel/o_voxel"

install_to() {
    local target="$1" label="$2"
    if [[ ! -d "${target}" ]]; then
        echo "⏭️  ${label}: 없음, 건너뜀 (${target})"
        return 0
    fi

    install -m 0644 "${OVERRIDE_DIR}/postprocess.py" "${target}/postprocess.py"
    install -m 0644 "${OVERRIDE_DIR}/uv_raster.py" "${target}/uv_raster.py"
    rm -f "${target}/postprocess.py.orig"
    rm -rf "${target}/__pycache__"
    echo "✅ ${label}: 상업 배포용 파일 적용"
}

install_to "${SITE_PKG}" "설치본"
install_to "${VENDORED}" "벤더본"

python - <<'PY'
from pathlib import Path
import o_voxel.postprocess as postprocess

source = Path(postprocess.__file__).read_text(encoding="utf-8")
if "from . import uv_raster" not in source:
    raise RuntimeError("o_voxel 상업 배포용 파일 적용을 확인할 수 없습니다.")
print("✅ o_voxel 상업 배포용 파일 검증 완료")
PY
