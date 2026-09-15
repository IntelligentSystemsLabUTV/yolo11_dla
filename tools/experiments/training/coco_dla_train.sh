#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
TOOLS_DIR="$(cd -- "${SCRIPT_DIR}/../../" && pwd)"
TRAINER="${SCRIPT_DIR}/train_coco_dla.py"
RUNS_DIR="${TOOLS_DIR}/ultralytics/runs"
WANDB_ENTITY="alexandru-cretu-university-of-rome-tor-vergata"
WANDB_PROJECT="yolo_dla"

configure_model() {
    local model="yolo11n-dla"
    local next_is_model=false
    local next_is_wandb_run_id=false
    local custom_wandb_run_id=""
    local arg
    for arg in "$@"; do
        if [[ "${next_is_model}" == true ]]; then
            model="${arg}"
            next_is_model=false
        elif [[ "${next_is_wandb_run_id}" == true ]]; then
            custom_wandb_run_id="${arg}"
            next_is_wandb_run_id=false
        elif [[ "${arg}" == "--model" ]]; then
            next_is_model=true
        elif [[ "${arg}" == --model=* ]]; then
            model="${arg#--model=}"
        elif [[ "${arg}" == "--wandb-run-id" ]]; then
            next_is_wandb_run_id=true
        elif [[ "${arg}" == --wandb-run-id=* ]]; then
            custom_wandb_run_id="${arg#--wandb-run-id=}"
        fi
    done

    local task="detect"
    [[ "${model}" == *-seg ]] && task="segment"
    local run_name="coco-${model}"
    local wandb_run_id="${model//-/_}_coco2017"
    [[ -n "${custom_wandb_run_id}" ]] && wandb_run_id="${custom_wandb_run_id}"

    STATE_DIR="${RUNS_DIR}/${task}/${run_name}"
    PID_FILE="${STATE_DIR}/train.pid"
    LOG_FILE="${STATE_DIR}/train.log"
    WANDB_URL="https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${wandb_run_id}"
}

is_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(<"${PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null || return 1
    [[ "$(ps -p "${pid}" -o args=)" == *"${TRAINER}"* || "$(ps -p "${pid}" -o args=)" == *"${TOOLS_DIR}/train_coco_dla.py"* ]]
}

start() {
    if is_running; then
        echo "Training is already running (PID $(<"${PID_FILE}"))."
        return
    fi
    mkdir -p "${STATE_DIR}"
    printf '\n===== launch %s =====\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
    nohup python3 "${TRAINER}" "$@" >>"${LOG_FILE}" 2>&1 &
    echo "$!" >"${PID_FILE}"
    echo "Started training (PID $!)."
    echo "Log: ${LOG_FILE}"
}

stop() {
    if ! is_running; then
        echo "Training is not running."
        return
    fi
    local pid
    pid="$(<"${PID_FILE}")"
    kill -INT "${pid}"
    echo "Requested graceful stop for PID ${pid}."
    echo "It will save last.pt after the current batch and validation."
}

status() {
    if is_running; then
        echo "Training is running (PID $(<"${PID_FILE}"))."
        echo "W&B: ${WANDB_URL}"
        tail -c 32768 "${LOG_FILE}" | tr '\r' '\n' | tail -n 12
    else
        echo "Training is not running."
        echo "W&B: ${WANDB_URL}"
        [[ -f "${LOG_FILE}" ]] && tail -c 32768 "${LOG_FILE}" | tr '\r' '\n' | tail -n 12
    fi
}

case "${1:-run}" in
    start|resume)
        shift
        configure_model "$@"
        start "$@"
        ;;
    stop)
        shift
        configure_model "$@"
        stop
        ;;
    status)
        shift
        configure_model "$@"
        status
        ;;
    run)
        shift || true
        configure_model "$@"
        mkdir -p "${STATE_DIR}"
        echo "$$" >"${PID_FILE}"
        printf '\n===== foreground launch %s =====\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
        exec > >(tee -a "${LOG_FILE}") 2>&1
        exec python3 "${TRAINER}" "$@"
        ;;
    *)
        echo "Usage: $0 {start|resume|stop|status|run} [--model MODEL] [training options]" >&2
        exit 2
        ;;
esac
