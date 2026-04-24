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

export ASK2ACT_STRETCH_SERVER_BIND
export ASK2ACT_STRETCH_SERVER_OBSERVE_MODE
export ASK2ACT_STRETCH_SERVER_EXECUTE_MODE
export ASK2ACT_STRETCH_SERVER_ARTIFACT_ROOT

cd "${REPO_ROOT}"
exec "${ASK2ACT_STRETCH_SERVER_PYTHON_BIN}" -m real.stretch_transport.robot_server --bind "${ASK2ACT_STRETCH_SERVER_BIND}"
