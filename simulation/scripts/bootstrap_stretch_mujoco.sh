#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${ROOT_DIR}/stretch_mujoco"

if [[ -d "${SIM_ROOT}/.git" ]]; then
  echo "stretch_mujoco already exists at ${SIM_ROOT}"
else
  git clone --recurse-submodules https://github.com/hello-robot/stretch_mujoco.git "${SIM_ROOT}"
fi

"${SCRIPT_DIR}/setup_stretch_mujoco.sh"
