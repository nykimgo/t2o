#!/usr/bin/env bash
# previs-prep 컨테이너를 재시작해도 되는지 점검한다.
# 호스트에서 실행. exit 0 = 재시작해도 안전, exit 1 = 방해받는 사용자/작업 있음.
#
# 배경: 이 컨테이너는 GPU cgroup이 끊기면 재시작이 유일한 복구책인데,
# PID1이 bare bash라 재시작 시 안에서 돌던 세션/작업이 전부 죽는다.
set -uo pipefail

CONTAINER="${1:-previs-prep}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "컨테이너가 실행 중이 아닙니다: $CONTAINER"
  exit 1
fi

BLOCKERS=0
note() { echo "  $*"; }
section() { echo; echo "── $* ──"; }

echo "컨테이너: $CONTAINER   (uptime: $(docker ps --filter "name=^${CONTAINER}$" --format '{{.Status}}'))"

# 컨테이너 내부 프로세스 목록. PID는 컨테이너 네임스페이스 기준.
PS_OUT="$(docker exec "$CONTAINER" ps -eo pid,ppid,etimes,tty,args --no-headers 2>/dev/null)"

section "SSH 로그인 세션"
# sshd: root@pts/N = 대화형 로그인, root@notty = 포트포워딩/원격도구 채널
SSH_SESSIONS="$(grep -E 'sshd: [^ ]+@(pts|notty)' <<<"$PS_OUT" | grep -v 'listener')"
if [[ -n "$SSH_SESSIONS" ]]; then
  while read -r line; do note "$line"; done <<<"$SSH_SESSIONS"
  BLOCKERS=$((BLOCKERS + $(wc -l <<<"$SSH_SESSIONS")))
else
  note "없음"
fi

section "VS Code 원격 세션"
VSCODE="$(grep -c 'vscode-server' <<<"$PS_OUT")"
if [[ "$VSCODE" -gt 0 ]]; then
  note "vscode-server 프로세스 ${VSCODE}개 — 원격 접속자가 붙어 있습니다."
  # 안에서 열린 통합 터미널 수 (사용자가 실제로 작업 중인지 판단용)
  TERMS="$(grep -c 'shellIntegration-bash' <<<"$PS_OUT")"
  [[ "$TERMS" -gt 0 ]] && note "통합 터미널 ${TERMS}개 열려 있음"
  BLOCKERS=$((BLOCKERS + 1))
else
  note "없음"
fi

section "대화형 셸 (PID 1 제외)"
# pts에 붙어 있고 PID 1이 아닌 bash = 누군가의 docker exec 또는 로그인 셸
SHELLS="$(awk '$1 != 1 && $4 ~ /pts/ && $5 ~ /bash$/' <<<"$PS_OUT")"
if [[ -n "$SHELLS" ]]; then
  while read -r line; do note "$line"; done <<<"$SHELLS"
  note "(이 스크립트가 띄운 셸이 포함될 수 있습니다)"
  BLOCKERS=$((BLOCKERS + 1))
else
  note "없음"
fi

section "장시간 실행 중인 작업"
# 5분(300초) 이상 돌고 있는 python/추론/서버 프로세스 = 죽이면 손실 발생
JOBS="$(awk '$3 > 300 && $5 ~ /(python|server\.py|codex|train|infer)/' <<<"$PS_OUT")"
if [[ -n "$JOBS" ]]; then
  while read -r pid ppid etimes tty rest; do
    printf "  PID %-7s %5d분  %s\n" "$pid" "$((etimes / 60))" "${rest:0:90}"
  done <<<"$JOBS"
  BLOCKERS=$((BLOCKERS + $(wc -l <<<"$JOBS")))
else
  note "없음"
fi

section "GPU 점유 (호스트 nvidia-smi 기준)"
# 컨테이너 내부 nvidia-smi는 cgroup이 끊기면 실패하므로 호스트에서 조회하고,
# docker top의 호스트 PID와 교차 대조해 이 컨테이너 소유분만 걸러낸다.
HOST_PIDS="$(docker top "$CONTAINER" 2>/dev/null | awk 'NR>1 {print $2}')"
GPU_APPS="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)"
if [[ -z "$GPU_APPS" ]]; then
  note "GPU를 쓰는 프로세스 없음"
else
  FOUND=0
  while IFS=, read -r gpid gmem gname; do
    gpid="$(tr -d ' ' <<<"$gpid")"
    if grep -qx "$gpid" <<<"$HOST_PIDS"; then
      note "host PID $gpid —${gmem} —${gname}"
      FOUND=$((FOUND + 1))
    fi
  done <<<"$GPU_APPS"
  if [[ "$FOUND" -gt 0 ]]; then
    note "⚠ 이 컨테이너가 GPU 작업 중입니다. 재시작하면 날아갑니다."
    BLOCKERS=$((BLOCKERS + FOUND))
  else
    note "이 컨테이너 소유 GPU 프로세스 없음 (다른 컨테이너/호스트 작업은 영향 없음)"
  fi
fi

echo
if [[ "$BLOCKERS" -eq 0 ]]; then
  echo "✅ 안전 — 재시작해도 잃을 작업이 없어 보입니다."
  exit 0
fi
echo "⛔ 사용 중 — 위 항목 ${BLOCKERS}건이 재시작으로 종료됩니다."
echo "   재시작 전 해당 사용자에게 확인하세요."
exit 1
