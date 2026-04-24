#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/a6000_real.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${ASK2ACT_A6000_PYTHON_BIN:=python3}"
: "${ASK2ACT_A6000_HOST:=127.0.0.1}"
: "${ASK2ACT_A6000_PORT:=7862}"
: "${ASK2ACT_VLLM_BASE_URL:=http://127.0.0.1:8000/v1}"
: "${ASK2ACT_VLLM_MODEL:=mimo-vl}"
: "${ASK2ACT_DETECTOR_DEVICE:=cuda:1}"
: "${ASK2ACT_GEN_MAX_TOKENS:=8000}"
: "${ASK2ACT_THINK_HINT:=1}"
: "${ASK2ACT_STRETCH_TRANSPORT:=zmq}"
: "${ASK2ACT_STRETCH_ZMQ_ENDPOINT:=tcp://stretch-se3-3056.local:5557}"
: "${ASK2ACT_STRETCH_TIMEOUT_MS:=30000}"
: "${ASK2ACT_PIPELINE_MODE:=mock}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ASK2ACT_VLLM_BASE_URL
export ASK2ACT_VLLM_MODEL
export ASK2ACT_DETECTOR_DEVICE
export ASK2ACT_GEN_MAX_TOKENS
export ASK2ACT_THINK_HINT
export ASK2ACT_STRETCH_TRANSPORT
export ASK2ACT_STRETCH_ZMQ_ENDPOINT
export ASK2ACT_STRETCH_TIMEOUT_MS
export ASK2ACT_PIPELINE_MODE

cd "${REPO_ROOT}"
exec "${ASK2ACT_A6000_PYTHON_BIN}" -m uvicorn services.a6000_web.server:app --host "${ASK2ACT_A6000_HOST}" --port "${ASK2ACT_A6000_PORT}"
