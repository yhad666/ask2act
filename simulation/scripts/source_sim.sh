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
SIMULATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SIMULATION_ROOT}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
SIM_ROOT="${SIMULATION_ROOT}/stretch_mujoco"
SIM_LOG_DIR="${SIMULATION_ROOT}/logs/sim"
SIM_GRASP_DIR="${SIMULATION_ROOT}/sim_grasping"

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

mkdir -p "${SIM_LOG_DIR}" "${SIM_GRASP_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${SIMULATION_ROOT}/logs/ros}"
export STRETCH_SIM_ROOT="${SIM_ROOT}"
export STRETCH_SIM_LOG_DIR="${SIM_LOG_DIR}"
export STRETCH_SIM_GRASP_ROOT="${SIM_GRASP_DIR}"
export STRETCH_SIM_HEADLESS_MUJOCO_GL="${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}"
export STRETCH_WORKSPACE_ROOT="${REPO_ROOT}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

mkdir -p "${UV_CACHE_DIR}" "${ROS_LOG_DIR}"

if [[ -f "${ROS_SETUP}" ]]; then
  source_with_nounset_guard "${ROS_SETUP}"
fi

echo "Simulation environment ready"
echo "  STRETCH_SIM_ROOT=${STRETCH_SIM_ROOT}"
echo "  STRETCH_SIM_LOG_DIR=${STRETCH_SIM_LOG_DIR}"
echo "  STRETCH_SIM_GRASP_ROOT=${STRETCH_SIM_GRASP_ROOT}"
echo "  STRETCH_SIM_HEADLESS_MUJOCO_GL=${STRETCH_SIM_HEADLESS_MUJOCO_GL}"
echo "  MESA_D3D12_DEFAULT_ADAPTER_NAME=${MESA_D3D12_DEFAULT_ADAPTER_NAME}"
echo "  __NV_PRIME_RENDER_OFFLOAD=${__NV_PRIME_RENDER_OFFLOAD}"
echo "  __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME}"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"

if [[ "${_sourced}" -eq 1 ]]; then
  if [[ "${_had_errexit}" -eq 0 ]]; then
    set +e
  fi
  if [[ "${_had_pipefail}" -eq 0 ]]; then
    set +o pipefail
  fi
fi
