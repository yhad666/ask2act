#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${HOME}/.local/bin/uv"
SCENE_XML_PATH="${ROOT_DIR}/sim_grasping/tabletop_minimal_scene.xml"
RUN_TAG="layered_grasp_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/logs/sim_validation/minimal_grasping/${RUN_TAG}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/tmp/mesa_shader_cache}"
mkdir -p "${MESA_SHADER_CACHE_DIR}" "${OUTPUT_DIR}"

cd "${ROOT_DIR}/stretch_mujoco"
MUJOCO_GL="${MUJOCO_GL:-${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}}" \
  "${UV_BIN}" run python "${ROOT_DIR}/sim_grasping/layered_grasp_pipeline.py" \
  --scene-xml-path "${SCENE_XML_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo "Layered grasp pipeline artifacts: ${OUTPUT_DIR}"
