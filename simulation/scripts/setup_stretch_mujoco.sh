#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${ROOT_DIR}/stretch_mujoco"
UV_BIN="${HOME}/.local/bin/uv"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not installed at ${UV_BIN}" >&2
  echo "Install it first, then retry." >&2
  exit 1
fi

if [[ ! -d "${SIM_ROOT}/.git" ]]; then
  echo "stretch_mujoco repo missing at ${SIM_ROOT}" >&2
  echo "Clone the official repo first, then retry." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

cd "${STRETCH_SIM_ROOT}"

echo "== Stretch MuJoCo Setup =="
echo "repo: ${STRETCH_SIM_ROOT}"
echo "commit: $(git rev-parse --short HEAD)"
echo

echo "-- Verifying submodules --"
git submodule update --init --recursive
git submodule status
echo

echo "-- Syncing uv environment --"
"${UV_BIN}" sync
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" "${UV_BIN}" pip install --python .venv/bin/python PyYAML
echo

echo "-- Verifying core imports --"
"${UV_BIN}" run python - <<'PY'
import mujoco
import stretch_mujoco
import yaml
print("mujoco", mujoco.__version__)
print("stretch_mujoco", stretch_mujoco.__file__)
print("pyyaml", yaml.__version__)
PY
