#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCENE_XML_PATH="${ROOT_DIR}/sim_grasping/tabletop_single_cup_scene.xml"
TARGET_JSON="${ROOT_DIR}/sim_grasping/targets/single_cup_manual_target.json"

"${SCRIPT_DIR}/run_layered_grasp_pipeline.sh" \
  --scene-xml-path "${SCENE_XML_PATH}" \
  --target-selection-json "${TARGET_JSON}" \
  --head-pan-rad 0.0 \
  --head-tilt-rad -1.10 \
  --camera-hz 4.0 \
  --settle-seconds 2.0 \
  --record-video \
  "$@"
