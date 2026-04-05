#!/usr/bin/env bash

_sourced=0
_had_errexit=0
_had_pipefail=0

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _sourced=1
fi

if [[ $- == *e* ]]; then
  _had_errexit=1
fi

if set -o | grep -q '^pipefail[[:space:]]\+on$'; then
  _had_pipefail=1
fi

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="${ROOT_DIR}/ament_ws/install/setup.bash"
SIM_ROOT="${ROOT_DIR}/stretch_mujoco"
SIM_LOG_DIR="${ROOT_DIR}/logs/sim"
SIM_NOTES_DIR="${ROOT_DIR}/notes"
SIM_GRASP_DIR="${ROOT_DIR}/sim_grasping"

source_with_nounset_guard() {
  local target="$1"
  local had_nounset=0
  local source_rc=0

  export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"

  if [[ $- == *u* ]]; then
    had_nounset=1
    set +u
  fi

  # shellcheck source=/dev/null
  source "${target}" || source_rc=$?

  if [[ "${had_nounset}" -eq 1 ]]; then
    set -u
  fi

  return "${source_rc}"
}

mkdir -p "${SIM_LOG_DIR}" "${SIM_NOTES_DIR}" "${SIM_GRASP_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${ROOT_DIR}/logs/ros}"
export STRETCH_SIM_ROOT="${SIM_ROOT}"
export STRETCH_SIM_LOG_DIR="${SIM_LOG_DIR}"
export STRETCH_SIM_GRASP_ROOT="${SIM_GRASP_DIR}"
export STRETCH_SIM_HEADLESS_MUJOCO_GL="${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}"

mkdir -p "${UV_CACHE_DIR}" "${ROS_LOG_DIR}"

if [[ -f "${ROS_SETUP}" ]]; then
  source_with_nounset_guard "${ROS_SETUP}"
fi

if [[ -f "${WS_SETUP}" ]]; then
  source_with_nounset_guard "${WS_SETUP}"
fi

echo "Simulation environment ready"
echo "  STRETCH_SIM_ROOT=${STRETCH_SIM_ROOT}"
echo "  STRETCH_SIM_LOG_DIR=${STRETCH_SIM_LOG_DIR}"
echo "  STRETCH_SIM_GRASP_ROOT=${STRETCH_SIM_GRASP_ROOT}"
echo "  STRETCH_SIM_HEADLESS_MUJOCO_GL=${STRETCH_SIM_HEADLESS_MUJOCO_GL}"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"

if [[ "${_sourced}" -eq 1 ]]; then
  if [[ "${_had_errexit}" -eq 0 ]]; then
    set +e
  fi
  if [[ "${_had_pipefail}" -eq 0 ]]; then
    set +o pipefail
  fi
fi
