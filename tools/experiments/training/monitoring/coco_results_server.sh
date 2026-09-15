#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
TOOLS_DIR="$(cd -- "${SCRIPT_DIR}/../../../" && pwd)"
RUN_DIR="${TOOLS_DIR}/ultralytics/runs/detect/coco-yolo11n-dla"
DASHBOARD_DIR="${TOOLS_DIR}/coco-results"
PID_FILE="${RUN_DIR}/http.pid"
LOG_FILE="${RUN_DIR}/http.log"
PORT="${COCO_RESULTS_PORT:-8000}"

is_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(<"${PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null || return 1
    [[ "$(ps -p "${pid}" -o args=)" == *"http.server ${PORT}"* ]]
}

start() {
    if is_running; then
        echo "Results server is already running (PID $(<"${PID_FILE}"))."
        return
    fi

    mkdir -p "${DASHBOARD_DIR}"
    local file
    for file in \
        results.png results.csv train.log \
        BoxPR_curve.png BoxF1_curve.png BoxP_curve.png BoxR_curve.png \
        confusion_matrix.png confusion_matrix_normalized.png \
        labels.jpg val_batch0_labels.jpg val_batch0_pred.jpg \
        val_batch1_labels.jpg val_batch1_pred.jpg \
        val_batch2_labels.jpg val_batch2_pred.jpg; do
        ln -sfn "${RUN_DIR}/${file}" "${DASHBOARD_DIR}/${file}"
    done

    setsid python3 -m http.server "${PORT}" --bind 0.0.0.0 --directory "${DASHBOARD_DIR}" \
        >>"${LOG_FILE}" 2>&1 < /dev/null &
    echo "$!" >"${PID_FILE}"
    sleep 1
    status
}

stop() {
    if ! is_running; then
        echo "Results server is not running."
        return
    fi
    kill -TERM "$(<"${PID_FILE}")"
    echo "Results server stopped."
}

status() {
    if ! is_running; then
        echo "Results server is not running."
        return 1
    fi
    local lan_ip
    lan_ip="$(hostname -I | awk '{print $1}')"
    echo "Results server is running (PID $(<"${PID_FILE}"))."
    echo "Open: http://${lan_ip}:${PORT}/"
}

case "${1:-status}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *)
        echo "Usage: $0 {start|stop|status}" >&2
        exit 2
        ;;
esac
