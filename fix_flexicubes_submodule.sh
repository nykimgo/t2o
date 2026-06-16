#!/usr/bin/env bash
# flexicubes 서브모듈 초기화 스크립트
# 도커 컨테이너 내에서 실행하거나, 로컬에서 실행 가능

set -euo pipefail

# 기본값 설정
TRELLIS_DIR="${1:-/root/previz-t2o-main}"

echo "🔧 flexicubes 서브모듈 초기화 시작"
echo "   TRELLIS 디렉토리: ${TRELLIS_DIR}"
echo ""

# 디렉토리 확인
if [ ! -d "${TRELLIS_DIR}" ]; then
    echo "❌ TRELLIS 디렉토리를 찾을 수 없습니다: ${TRELLIS_DIR}"
    exit 1
fi

cd "${TRELLIS_DIR}"

# .git 디렉토리 확인 (git 저장소인지 확인)
if [ ! -d ".git" ]; then
    echo "⚠️  .git 디렉토리가 없습니다. git 저장소가 아닐 수 있습니다."
    echo "   flexicubes를 직접 설치합니다..."
    
    # flexicubes 디렉토리 확인
    FLEXICUBES_DIR="trellis/representations/mesh/flexicubes"
    
    if [ ! -d "${FLEXICUBES_DIR}" ] || [ -z "$(ls -A ${FLEXICUBES_DIR} 2>/dev/null)" ]; then
        echo "📦 flexicubes 디렉토리가 비어있습니다. 직접 클론합니다..."
        rm -rf "${FLEXICUBES_DIR}"
        mkdir -p "${FLEXICUBES_DIR}"
        git clone https://github.com/MaxtirError/FlexiCubes.git "${FLEXICUBES_DIR}"
        echo "✅ flexicubes 클론 완료"
    else
        echo "✅ flexicubes 디렉토리가 이미 존재합니다"
    fi
else
    echo "📦 git 서브모듈 초기화 중..."
    
    # 서브모듈 업데이트
    if git submodule update --init --recursive trellis/representations/mesh/flexicubes; then
        echo "✅ 서브모듈 초기화 완료"
    else
        echo "⚠️  서브모듈 초기화 실패. 직접 클론을 시도합니다..."
        
        FLEXICUBES_DIR="trellis/representations/mesh/flexicubes"
        rm -rf "${FLEXICUBES_DIR}"
        mkdir -p "${FLEXICUBES_DIR}"
        git clone https://github.com/MaxtirError/FlexiCubes.git "${FLEXICUBES_DIR}"
        echo "✅ flexicubes 직접 클론 완료"
    fi
fi

# flexicubes 디렉토리 확인
FLEXICUBES_DIR="trellis/representations/mesh/flexicubes"
if [ -d "${FLEXICUBES_DIR}" ] && [ -n "$(ls -A ${FLEXICUBES_DIR} 2>/dev/null)" ]; then
    echo ""
    echo "✅ flexicubes 서브모듈 확인 완료"
    echo "   위치: ${TRELLIS_DIR}/${FLEXICUBES_DIR}"
    echo "   파일 수: $(find ${FLEXICUBES_DIR} -type f | wc -l)개"
    
    # flexicubes.py 파일 확인
    if [ -f "${FLEXICUBES_DIR}/flexicubes.py" ]; then
        echo "   ✅ flexicubes.py 파일 확인됨"
    else
        echo "   ⚠️  flexicubes.py 파일을 찾을 수 없습니다"
        echo "   디렉토리 내용:"
        ls -la "${FLEXICUBES_DIR}" | head -10
    fi
else
    echo "❌ flexicubes 디렉토리가 여전히 비어있습니다"
    exit 1
fi

echo ""
echo "✨ 완료! 이제 TRELLIS 모듈을 임포트할 수 있습니다."

