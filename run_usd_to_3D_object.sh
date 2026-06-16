#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: run_usd_to_3D_object.sh <usd_file> <output_dir> [--model <trellis_model>] [--translate] [--filter] [-- <extra_trellis_args>]

Arguments:
  --model <model_path>   TRELLIS 모델 경로 또는 HF 모델명 (예: microsoft/TRELLIS-text-large)
  --translate            1단계 번역 활성화 (한국어 → 영어 번역)
  --filter               2단계 필터링 활성화 (프롬프트 증강)

Environment variables:
  OLLAMA_MODEL           1단계 번역용 Ollama 모델명 (기본: gemma3:12b)
  OLLAMA_FILTER_MODEL    2단계 필터링용 Ollama 모델명 (기본: gemma3:12b)
  OLLAMA_BASE_URL        Ollama 서버 URL (기본: unset)
  PARSE_TYPE             usd_parse_and_augment 파싱 타입 (object|actor|both, 기본: object)
  TRELLIS_MODEL_PATH     TRELLIS 모델 경로 또는 HF 모델명 (기본: microsoft/TRELLIS-text-xlarge)
  TRELLIS_BASE_OUTPUT    TrellisInferenceCore base_output (기본: /mnt/sdb_1TB/previz/text_to_3d)
  TRELLIS_CONFIG         TRELLIS YAML 설정 경로 (미지정 시 기본 설정 사용)

JSON 경로는 자동 생성됩니다: {output_dir}/{model_name}/{YYYYMMDD}/usd_results.json

참고: 기본적으로 LLM 번역은 실행되지 않습니다. --translate 또는 --filter 옵션을 사용하여 활성화하세요.

예시:
  # 기본 실행 (LLM 번역 없이 USD 파싱만 수행)
  ./run_usd_to_3D_object.sh scene.usda /mnt/output

  # 1단계 번역만 활성화
  ./run_usd_to_3D_object.sh scene.usda /mnt/output --translate

  # 1단계 번역 + 2단계 필터링 활성화
  ./run_usd_to_3D_object.sh scene.usda /mnt/output --translate --filter

  # 환경변수로 모델 지정
  OLLAMA_MODEL=gemma3:12b OLLAMA_FILTER_MODEL=gemma3:12b \
    TRELLIS_MODEL_PATH=microsoft/TRELLIS-text-large \
    ./run_usd_to_3D_object.sh scene.usda /mnt/output --translate --filter

  # TRELLIS 모델만 인자로 지정
  ./run_usd_to_3D_object.sh scene.usda /mnt/output --model microsoft/TRELLIS-text-large

  # 추가 인자 전달
  ./run_usd_to_3D_object.sh scene.usda /mnt/output --model microsoft/TRELLIS-text-large -- --max_items 5
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

USD_FILE="$1"
OUTPUT_DIR="$2"
shift 2

# --model, --translate, --filter 옵션 파싱
TRELLIS_MODEL_ARG=""
ENABLE_TRANSLATE=false
ENABLE_FILTER=false
TRELLIS_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      if [[ $# -lt 2 ]]; then
        echo "❌ --model 옵션에는 모델 경로가 필요합니다"
        exit 1
      fi
      TRELLIS_MODEL_ARG="$2"
      shift 2
      ;;
    --translate)
      ENABLE_TRANSLATE=true
      shift
      ;;
    --filter)
      ENABLE_FILTER=true
      shift
      ;;
    --)
      shift
      TRELLIS_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      TRELLIS_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:4b}"
OLLAMA_FILTER_MODEL="${OLLAMA_FILTER_MODEL:-gemma3:4b}"
PARSE_TYPE="${PARSE_TYPE:-object}"
# --model 인자가 있으면 우선 사용, 없으면 환경변수, 둘 다 없으면 기본값
TRELLIS_MODEL_PATH="${TRELLIS_MODEL_ARG:-${TRELLIS_MODEL_PATH:-microsoft/TRELLIS-text-base}}"
TRELLIS_BASE_OUTPUT="${TRELLIS_BASE_OUTPUT:-${OUTPUT_DIR}}"

# Ollama 서버 자동 시작 (번역 또는 필터링이 활성화된 경우에만)
OLLAMA_API_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_STARTED_BY_SCRIPT=false

# 번역 또는 필터링이 활성화된 경우에만 Ollama 서버 확인/시작
if [[ "${ENABLE_TRANSLATE}" == "true" ]] || [[ "${ENABLE_FILTER}" == "true" ]]; then
    # OLLAMA_BASE_URL이 설정되지 않았거나 localhost인 경우에만 로컬 ollama 확인/시작
    if [[ -z "${OLLAMA_BASE_URL:-}" ]] || [[ "${OLLAMA_BASE_URL}" == "http://localhost:11434" ]] || [[ "${OLLAMA_BASE_URL}" == "localhost:11434" ]]; then
    # Ollama가 실행 중인지 확인
    if command -v curl >/dev/null 2>&1; then
        if ! curl -s "${OLLAMA_API_URL}/api/tags" >/dev/null 2>&1; then
            echo "🔄 Ollama 서버가 실행 중이 아닙니다. 시작 중..."
            # ollama serve를 백그라운드로 시작
            if command -v ollama >/dev/null 2>&1; then
                # 기존 ollama 프로세스가 있는지 확인
                if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
                    nohup ollama serve >/tmp/ollama.log 2>&1 &
                    OLLAMA_PID=$!
                    OLLAMA_STARTED_BY_SCRIPT=true
                    echo "✅ Ollama 서버 시작됨 (PID: ${OLLAMA_PID})"
                    # 서버가 준비될 때까지 대기 (최대 30초)
                    echo "⏳ Ollama 서버 준비 대기 중..."
                    for i in {1..30}; do
                        if curl -s "${OLLAMA_API_URL}/api/tags" >/dev/null 2>&1; then
                            echo "✅ Ollama 서버 준비 완료"
                            break
                        fi
                        if [[ $i -eq 30 ]]; then
                            echo "❌ Ollama 서버가 30초 내에 시작되지 않았습니다."
                            exit 1
                        fi
                        sleep 1
                    done
                else
                    echo "ℹ️ Ollama 서버가 이미 실행 중입니다."
                fi
            else
                echo "❌ 'ollama' 명령을 찾을 수 없습니다. Ollama가 설치되어 있는지 확인하세요."
                exit 1
            fi
        else
            echo "✅ Ollama 서버가 이미 실행 중입니다."
        fi
    else
        echo "⚠️ curl이 설치되어 있지 않아 Ollama 서버 상태를 확인할 수 없습니다."
        echo "   Ollama가 실행 중인지 수동으로 확인하세요."
    fi
    else
        echo "ℹ️ 원격 Ollama 서버 사용: ${OLLAMA_BASE_URL}"
    fi
else
    echo "ℹ️ LLM 번역이 비활성화되어 Ollama 서버를 시작하지 않습니다."
fi

# 스크립트 종료 시 정리 함수
cleanup() {
    if [[ "${OLLAMA_STARTED_BY_SCRIPT}" == "true" ]] && [[ -n "${OLLAMA_PID:-}" ]]; then
        echo "🔄 스크립트에서 시작한 Ollama 서버 종료 중..."
        kill "${OLLAMA_PID}" 2>/dev/null || true
        # 프로세스가 완전히 종료될 때까지 대기
        wait "${OLLAMA_PID}" 2>/dev/null || true
        echo "✅ Ollama 서버 종료 완료"
    fi
}

# 종료 시 정리 함수 호출
trap cleanup EXIT INT TERM

# 모델명에서 마지막 부분만 추출 (microsoft/TRELLIS-text-xlarge -> TRELLIS-text-xlarge)
MODEL_NAME=$(basename "${TRELLIS_MODEL_PATH}")
# 날짜 생성 (YYYYMMDD)
CURRENT_DATE=$(date +%Y%m%d)

# JSON 경로 자동 생성: {output_dir}/{model_name}/{YYYYMMDD}/usd_results.json
OUTPUT_JSON="${OUTPUT_DIR}/${MODEL_NAME}/${CURRENT_DATE}/usd_results.json"

# JSON 디렉토리 생성
mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "📁 JSON 저장 경로: ${OUTPUT_JSON}"

# USD 파일의 기본 디렉토리 (원본 USD 파일들이 있는 위치)
USD_BASE_DIR="$(dirname "${USD_FILE}")"

PREVIZ_PIPELINE_DIR="${SCRIPT_DIR}/previz_pipeline"
USD_SCRIPT="${PREVIZ_PIPELINE_DIR}/usd_parse_and_augment.py"
TRELLIS_JSON_SCRIPT="${PREVIZ_PIPELINE_DIR}/json_parse_and_inference.py"

if [[ ! -f "${USD_SCRIPT}" ]]; then
  echo "❌ usd_parse_and_augment.py를 찾을 수 없습니다: ${USD_SCRIPT}"
  exit 1
fi

if [[ ! -f "${TRELLIS_JSON_SCRIPT}" ]]; then
  echo "❌ json_parse_and_inference.py를 찾을 수 없습니다: ${TRELLIS_JSON_SCRIPT}"
  exit 1
fi

if [[ ! -f "${USD_FILE}" ]]; then
  echo "❌ USD 파일을 찾을 수 없습니다: ${USD_FILE}"
  exit 1
fi

# 이전 실행의 JSON이 남아 있으면 1단계 실패 시에도 2단계가 stale 데이터로 진행될 수 있음
if [[ -f "${OUTPUT_JSON}" ]]; then
  echo "ℹ️ 기존 JSON 제거: ${OUTPUT_JSON}"
  rm -f "${OUTPUT_JSON}"
fi

echo "🔄 1/2 USD 파싱 및 프롬프트 증강 실행"
if [[ "${ENABLE_TRANSLATE}" == "true" ]]; then
  echo "   1단계 번역: 활성화 (모델: ${OLLAMA_MODEL})"
else
  echo "   1단계 번역: 비활성화"
fi
if [[ "${ENABLE_FILTER}" == "true" ]]; then
  echo "   2단계 필터링: 활성화 (모델: ${OLLAMA_FILTER_MODEL})"
else
  echo "   2단계 필터링: 비활성화"
fi
# PYTHONPATH를 프로젝트 루트로 설정하여 trellis 모듈 import 가능하도록 함
USD_CMD=(python "${USD_SCRIPT}" "${USD_FILE}" --output "${OUTPUT_JSON}" --type "${PARSE_TYPE}" --model "${OLLAMA_MODEL}" --filter-model "${OLLAMA_FILTER_MODEL}")
if [[ -n "${OLLAMA_BASE_URL:-}" ]]; then
  USD_CMD+=(--base-url "${OLLAMA_BASE_URL}")
fi
if [[ "${ENABLE_TRANSLATE}" == "true" ]]; then
  USD_CMD+=(--translate)
fi
if [[ "${ENABLE_FILTER}" == "true" ]]; then
  USD_CMD+=(--filter)
fi
echo "📌 ${USD_CMD[*]}"
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${USD_CMD[@]}"

if [[ ! -f "${OUTPUT_JSON}" ]]; then
  echo "❌ 1단계가 JSON을 생성하지 못했습니다: ${OUTPUT_JSON}"
  exit 1
fi

echo "🔄 2/3 TRELLIS 추론 실행"
# PYTHONPATH를 프로젝트 루트로 설정하여 trellis 모듈 import 가능하도록 함
TRELLIS_CMD=(python "${TRELLIS_JSON_SCRIPT}" --json "${OUTPUT_JSON}" --output "${OUTPUT_DIR}" --model_path "${TRELLIS_MODEL_PATH}" --base_output "${TRELLIS_BASE_OUTPUT}" --usd_root "${USD_BASE_DIR}")
if [[ -n "${TRELLIS_CONFIG:-}" ]]; then
  TRELLIS_CMD+=(--config "${TRELLIS_CONFIG}")
fi
if [[ ${#TRELLIS_EXTRA_ARGS[@]} -gt 0 ]]; then
  TRELLIS_CMD+=("${TRELLIS_EXTRA_ARGS[@]}")
fi
echo "📌 ${TRELLIS_CMD[*]}"
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${TRELLIS_CMD[@]}"

echo ""
echo "🔄 3/3 GLB → USD 변환 및 병합 실행"
MERGE_SCRIPT="${PREVIZ_PIPELINE_DIR}/merge_glb_to_usd.py"

if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "❌ merge_glb_to_usd.py를 찾을 수 없습니다: ${MERGE_SCRIPT}"
  exit 1
fi

# GLB 파일은 이제 원본 USD 옆 assets 폴더에 저장됩니다.
# 구조: {원본 object_n.usda 디렉토리}/assets/{object_name}/*.glb
# 따라서 JSON의 usd_file_path(원본 USD 절대 경로)를 기준으로 GLB를 찾습니다.

echo "📁 원본 USD 기본 디렉토리: ${USD_BASE_DIR}"

# JSON 파일에서 object_path/actor_path 읽기
if [[ ! -f "${OUTPUT_JSON}" ]]; then
  echo "⚠️ JSON 파일을 찾을 수 없습니다: ${OUTPUT_JSON}"
  echo "   GLB → USD 변환 단계를 건너뜁니다."
else
  # Python 스크립트로 GLB 파일 찾기 및 USD 변환
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python3 <<EOF
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys

json_path = Path("${OUTPUT_JSON}")
usd_base_dir = Path("${USD_BASE_DIR}")
merge_script = Path("${MERGE_SCRIPT}")
script_dir = Path("${SCRIPT_DIR}")
trellis_output_base = Path("${OUTPUT_DIR}") / "${MODEL_NAME}" / "${CURRENT_DATE}"

if not json_path.exists():
    print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
    sys.exit(1)

if not merge_script.exists():
    print(f"❌ merge_glb_to_usd.py를 찾을 수 없습니다: {merge_script}")
    sys.exit(1)

# JSON 파일 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data.get('results', []) if isinstance(data, dict) else data
if not results:
    print("⚠️ JSON에 결과가 없습니다.")
    sys.exit(0)

print(f"📊 총 {len(results)}개 항목 처리 중...")

processed_count = 0
skipped_count = 0
error_count = 0


def resolve_original_usd(item, path_key, object_name):
    """원본 USD(object_n.usda/actor_n.usda)의 절대 경로를 찾습니다."""
    # 1) JSON에 절대 경로(usd_file_path)가 있으면 최우선 사용
    usd_file_path = item.get('usd_file_path', '')
    if usd_file_path:
        p = Path(usd_file_path)
        if p.exists():
            return p

    # 2) usd_relative_path가 있으면 원본 USD 루트 기준으로 해석
    usd_relative_path = item.get('usd_relative_path', '')
    if usd_relative_path:
        normalized = usd_relative_path[2:] if usd_relative_path.startswith("./") else usd_relative_path
        # path_key의 상위 컴포넌트(scene/shot 등)로 디렉토리 구성
        parent_components = [c for c in path_key.split('/') if c][:-1]
        ref_base = usd_base_dir
        for comp in parent_components:
            ref_base = ref_base / comp
        for cand in (ref_base / normalized, usd_base_dir / normalized):
            cand = cand.resolve()
            if cand.exists():
                return cand
            for alt_suffix in ('.usda', '.usd'):
                alt = cand.with_suffix(alt_suffix)
                if alt.exists():
                    return alt

    # 3) 폴백: 폴더 구조(scene/objects 또는 scene/shot/objects)를 모두 시도
    components = [c for c in path_key.split('/') if c]
    base = usd_base_dir
    for comp in components[:-1]:
        base = base / comp
    candidates = [
        base / "objects" / f"{object_name}.usda",
        base / "actors" / f"{object_name}.usda",
        base / f"{object_name}.usda",
        usd_base_dir / "/".join(components[:-1]) / "objects" / f"{object_name}.usda" if len(components) > 1 else None,
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def find_glb_fallback(object_name, path_key):
    """assets에 GLB가 없을 때 TRELLIS 출력 디렉토리에서 GLB를 찾습니다."""
    parts = [c for c in path_key.split('/') if c]
    scene_name = next((p for p in parts if p.startswith('scene_')), None)
    if not scene_name:
        return None

    search_dirs = [
        trellis_output_base / scene_name / scene_name / object_name,
        trellis_output_base / scene_name / "shot_unknown" / object_name,
        trellis_output_base / scene_name / "scene" / object_name,
    ]
    for glb_dir in search_dirs:
        if not glb_dir.exists():
            continue
        glb_files = list(glb_dir.glob('*.glb'))
        if glb_files:
            return sorted(glb_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


for idx, item in enumerate(results, 1):
    # Scene canonical만 GLB→USD 변환/주입 (스펙 2.1, #3)
    # shot override / meta-only 항목은 scene canonical reference로 geometry를 상속하므로 건너뜁니다.
    if item.get('object_scope') == 'shot' or item.get('_pipeline') == 'meta_only':
        skipped_count += 1
        continue

    object_path = item.get('object_path', '')
    actor_path = item.get('actor_path', '')
    path_key = object_path if object_path else actor_path

    if not path_key:
        print(f"⚠️ [{idx}/{len(results)}] object_path/actor_path가 없어 건너뜁니다.")
        skipped_count += 1
        continue

    # object_name 추출 (예: scene_1/object_1 -> object_1)
    object_name = item.get('object_name') or item.get('actor_name')
    if not object_name:
        parts = [p for p in path_key.replace('\\\\', '/').split('/') if p]
        object_name = next(
            (p for p in parts if re.match(r'^(object|actor)_', p, re.IGNORECASE)),
            parts[-1] if parts else None,
        )
    if not object_name:
        print(f"⚠️ [{idx}/{len(results)}] object_name을 추출할 수 없습니다: {path_key}")
        skipped_count += 1
        continue

    # 원본 USD 경로 확정
    original_usd = resolve_original_usd(item, path_key, object_name)
    if not original_usd:
        print(f"⚠️ [{idx}/{len(results)}] 원본 USD 파일을 찾을 수 없습니다: {path_key}")
        skipped_count += 1
        continue

    # scene canonical 우선 (#8): shot override 경로면 scene canonical로 정규화
    # .../scene_n/shot_m/objects/object_N.usda -> .../scene_n/objects/object_N.usda
    original_usd = Path(original_usd).resolve()
    if original_usd.parent.name == "objects" and re.match(r'^shot_', original_usd.parent.parent.name):
        canonical = original_usd.parent.parent.parent / "objects" / original_usd.name
        if canonical.exists():
            print(f"⚠️ [{idx}/{len(results)}] shot 경로를 scene canonical로 정규화: {canonical}")
            original_usd = canonical
        else:
            print(f"⚠️ [{idx}/{len(results)}] shot 경로이나 scene canonical을 찾을 수 없어 건너뜁니다: {original_usd}")
            skipped_count += 1
            continue

    # GLB 파일은 원본 USD 옆 assets/{object_name} 폴더에 있음
    glb_dir = original_usd.parent / "assets" / object_name
    glb_file = None

    if glb_dir.exists():
        glb_files = list(glb_dir.glob('*.glb'))
        if glb_files:
            glb_file = sorted(glb_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    if glb_file is None:
        fallback_glb = find_glb_fallback(object_name, path_key)
        if fallback_glb:
            glb_dir.mkdir(parents=True, exist_ok=True)
            dest_glb = glb_dir / fallback_glb.name
            if not dest_glb.exists() or fallback_glb.stat().st_mtime > dest_glb.stat().st_mtime:
                shutil.copy2(fallback_glb, dest_glb)
            glb_file = dest_glb
            print(f"ℹ️ [{idx}/{len(results)}] TRELLIS 출력에서 GLB 복사: {fallback_glb} → {dest_glb}")
        elif not glb_dir.exists():
            print(f"⚠️ [{idx}/{len(results)}] GLB 디렉토리를 찾을 수 없습니다: {glb_dir}")
            skipped_count += 1
            continue
        else:
            print(f"⚠️ [{idx}/{len(results)}] GLB 파일을 찾을 수 없습니다: {glb_dir}")
            skipped_count += 1
            continue

    print(f"🔄 [{idx}/{len(results)}] GLB → USD 변환 및 원본 주입: {glb_file.name}")
    print(f"   GLB: {glb_file}")
    print(f"   USD: {original_usd}")

    try:
        subprocess.run(
            [
                sys.executable,
                str(merge_script),
                str(glb_file),
                str(original_usd)
            ],
            cwd=str(script_dir),
            env={**os.environ, 'PYTHONPATH': f"{script_dir}:{os.environ.get('PYTHONPATH', '')}"},
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ [{idx}/{len(results)}] 완료: {object_name}")
        processed_count += 1
    except subprocess.CalledProcessError as e:
        print(f"❌ [{idx}/{len(results)}] 실패: {object_name}")
        print(f"   오류: {e.stderr}")
        error_count += 1

print("")
print(f"📊 처리 완료:")
print(f"   ✅ 성공: {processed_count}개")
print(f"   ⚠️ 건너뜀: {skipped_count}개")
print(f"   ❌ 오류: {error_count}개")
EOF
fi

echo ""
echo "✅ 전체 파이프라인 완료"


