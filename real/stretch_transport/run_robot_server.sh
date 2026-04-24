#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/robot_server.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${ASK2ACT_STRETCH_SERVER_PYTHON_BIN:=python3}"
: "${ASK2ACT_STRETCH_SERVER_BIND:=tcp://0.0.0.0:5557}"
: "${ASK2ACT_STRETCH_SERVER_OBSERVE_MODE:=command}"
: "${ASK2ACT_STRETCH_SERVER_EXECUTE_MODE:=command}"
: "${ASK2ACT_STRETCH_SERVER_ARTIFACT_ROOT:=${SCRIPT_DIR}/artifacts}"
: "${ASK2ACT_STRETCH_SERVER_AUTO_INSTALL:=1}"
: "${ASK2ACT_STRETCH_SERVER_REQUIREMENTS_FILE:=${SCRIPT_DIR}/requirements.txt}"

export ASK2ACT_STRETCH_SERVER_BIND
export ASK2ACT_STRETCH_SERVER_OBSERVE_MODE
export ASK2ACT_STRETCH_SERVER_EXECUTE_MODE
export ASK2ACT_STRETCH_SERVER_ARTIFACT_ROOT

cd "${REPO_ROOT}"

if [[ "${ASK2ACT_STRETCH_SERVER_AUTO_INSTALL}" == "1" ]]; then
  if ! "${ASK2ACT_STRETCH_SERVER_PYTHON_BIN}" -c "import zmq" >/dev/null 2>&1; then
    "${ASK2ACT_STRETCH_SERVER_PYTHON_BIN}" -m pip install -r "${ASK2ACT_STRETCH_SERVER_REQUIREMENTS_FILE}"
  fi
fi

exec "${ASK2ACT_STRETCH_SERVER_PYTHON_BIN}" -m real.stretch_transport.robot_server --bind "${ASK2ACT_STRETCH_SERVER_BIND}"
