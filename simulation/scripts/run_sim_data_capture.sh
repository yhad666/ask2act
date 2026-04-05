#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${HOME}/.local/bin/uv"
SCENE_XML_PATH="${1:-${ROOT_DIR}/stretch_mujoco/stretch_mujoco/models/scene.xml}"
TAG="${2:-default_scene}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/tmp/mesa_shader_cache}"
mkdir -p "${MESA_SHADER_CACHE_DIR}"

cd "${ROOT_DIR}/stretch_mujoco"
MUJOCO_GL="${MUJOCO_GL:-${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}}" \
  "${UV_BIN}" run python "${ROOT_DIR}/sim_grasping/capture_validation_data.py" \
  --scene-xml-path "${SCENE_XML_PATH}" \
  --tag "${TAG}"
