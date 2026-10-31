#!/usr/bin/env bash
# Start/stop the MentoPlan API + UI server (serves frontend/dist at http://127.0.0.1:8000)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$ROOT/.server.pid"
LOG="${MENTOPLAN_LOG:-$ROOT/.server.log}"
case "${1:-start}" in
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "already running"; exit 0; fi
    # cap math-library threads: leaves CPU for the desktop and limits solver workspace memory
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
    # hard memory ceiling for the server and its job processes (never lets it freeze the machine)
    WRAP=()
    if command -v systemd-run >/dev/null && systemd-run --user --scope -q -p MemoryMax=64M true 2>/dev/null; then
      WRAP=(systemd-run --user --scope -q -p "MemoryMax=${MENTOPLAN_MEMORY_MAX:-8G}" -p MemorySwapMax=0)
    fi
    nohup "${WRAP[@]}" "$ROOT/.venv/bin/python" -m mentoplan.cli serve --port "${PORT:-8000}" >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:${PORT:-8000}/api/health" >/dev/null && { echo "started"; exit 0; }; sleep 0.5; done
    echo "failed to start, see $LOG"; exit 1 ;;
  stop)
    [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"; echo "stopped" ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
esac
