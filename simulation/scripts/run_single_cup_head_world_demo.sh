#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMULATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${HOME}/.local/bin/uv"
RUN_TAG="ask2act_grasp_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SIMULATION_ROOT}/logs/sim_validation/ask2act_grasp/${RUN_TAG}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/tmp/mesa_shader_cache}"
mkdir -p "${MESA_SHADER_CACHE_DIR}" "${OUTPUT_DIR}"

cd "${STRETCH_SIM_ROOT}"
MUJOCO_GL="${MUJOCO_GL:-${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}}" \
  "${UV_BIN}" run python "${SIMULATION_ROOT}/pipeline.py" \
  --run-dir "${OUTPUT_DIR}" \
  --headless \
  "$@"

echo "Ask2Act grasp pipeline artifacts: ${OUTPUT_DIR}"
