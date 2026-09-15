#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
TOOLS_DIR="$(cd -- "${SCRIPT_DIR}/../../../" && pwd)"
SYNC_SCRIPT="${SCRIPT_DIR}/wandb_live_sync.py"
STATE_DIR="${TOOLS_DIR}/ultralytics/runs/detect"
PID_FILE="${STATE_DIR}/wandb-live-sync.pid"
LOG_FILE="${STATE_DIR}/wandb-live-sync.log"

is_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(<"${PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null || return 1
    [[ "$(ps -p "${pid}" -o args=)" == *"${SYNC_SCRIPT}"* || "$(ps -p "${pid}" -o args=)" == *"${TOOLS_DIR}/wandb_live_sync.py"* ]]
}

start() {
    if is_running; then
        echo "W&B live sync is already running (PID $(<"${PID_FILE}"))."
        return
    fi
    mkdir -p "${STATE_DIR}"
    printf '\n===== launch %s =====\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
    WANDB_MODE=shared nohup python3 "${SYNC_SCRIPT}" "$@" >>"${LOG_FILE}" 2>&1 &
    echo "$!" >"${PID_FILE}"
    echo "Started W&B live sync (PID $!)."
    echo "Log: ${LOG_FILE}"
}

stop() {
    if ! is_running; then
        echo "W&B live sync is not running."
        return
    fi
    local pid
    pid="$(<"${PID_FILE}")"
    kill -TERM "${pid}"
    echo "Requested W&B live sync stop for PID ${pid}."
}

status() {
    if is_running; then
        echo "W&B live sync is running (PID $(<"${PID_FILE}"))."
    else
        echo "W&B live sync is not running."
    fi
    [[ -f "${LOG_FILE}" ]] && tail -n 20 "${LOG_FILE}"
}

case "${1:-status}" in
    start)
        shift
        start "$@"
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    run)
        shift
        WANDB_MODE=shared exec python3 "${SYNC_SCRIPT}" "$@"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|run} [sync options]" >&2
        exit 2
        ;;
esac
