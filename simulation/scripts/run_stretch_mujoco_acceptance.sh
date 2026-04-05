#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/sim_validation"
DEFAULT_SCENE="${ROOT_DIR}/stretch_mujoco/stretch_mujoco/models/scene.xml"
TABLETOP_SCENE="${ROOT_DIR}/sim_grasping/tabletop_minimal_scene.xml"

mkdir -p "${LOG_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

{
  echo "== Stretch MuJoCo Acceptance =="
  date -Iseconds
  echo

  echo "-- Smoke: headless --"
  "${SCRIPT_DIR}/smoke_test_sim.sh" headless
  echo

  echo "-- Smoke: perception --"
  "${SCRIPT_DIR}/smoke_test_sim.sh" perception
  echo

  echo "-- Smoke: GUI --"
  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    "${SCRIPT_DIR}/smoke_test_sim.sh" gui || true
  else
    echo "[WARN] DISPLAY/WAYLAND_DISPLAY not set; skip GUI smoke test"
  fi
  echo

  echo "-- Data capture: default scene --"
  "${SCRIPT_DIR}/run_sim_data_capture.sh" "${DEFAULT_SCENE}" default_scene
  echo

  echo "-- Motion validation --"
  "${SCRIPT_DIR}/run_sim_motion_validation.sh" "${DEFAULT_SCENE}"
  echo

  echo "-- Data capture: tabletop scene --"
  "${SCRIPT_DIR}/run_sim_data_capture.sh" "${TABLETOP_SCENE}" tabletop_scene
  echo

  echo "-- Scene snapshots --"
  "${SCRIPT_DIR}/run_sim_scene_snapshots.sh"
} | tee "${LOG_DIR}/acceptance_run.log"
