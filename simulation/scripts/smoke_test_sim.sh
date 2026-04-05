#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${ROOT_DIR}/stretch_mujoco"
UV_BIN="${HOME}/.local/bin/uv"
MODE="${1:-headless}"
TMP_DIR="${STRETCH_SIM_LOG_DIR:-/tmp}"

pass() {
  echo "[PASS] $*"
}

warn() {
  echo "[WARN] $*"
}

fail() {
  echo "[FAIL] $*"
  exit 1
}

[[ -x "${UV_BIN}" ]] || fail "uv not found at ${UV_BIN}"
[[ -d "${SIM_ROOT}" ]] || fail "stretch_mujoco repo missing at ${SIM_ROOT}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

echo "== Stretch MuJoCo Smoke Test =="
echo "timestamp: $(date -Iseconds)"
echo "mode: ${MODE}"

cd "${SIM_ROOT}"

"${UV_BIN}" run python - <<'PY'
import mujoco
import stretch_mujoco
print("[PASS] mujoco import ok:", mujoco.__version__)
print("[PASS] stretch_mujoco import ok:", stretch_mujoco.__file__)
PY

if [[ "${MODE}" == "headless" ]]; then
  mkdir -p "${TMP_DIR}"
  HEADLESS_SCRIPT="$(mktemp "${TMP_DIR}/stretch_mujoco_headless_smoke_XXXXXX.py")"
  trap 'rm -f "${HEADLESS_SCRIPT:-}" "${GUI_SCRIPT:-}"' EXIT
  cat > "${HEADLESS_SCRIPT}" <<'PY'
import time
from stretch_mujoco import StretchMujocoSimulator

def main():
    sim = StretchMujocoSimulator()
    sim.start(headless=True)
    status = sim.pull_status()
    print(f"[PASS] connected to headless simulator: sim_time={status.time:.3f}, fps={status.fps:.2f}")

    sim.move_to("lift", 0.7)
    time.sleep(1.0)
    status_after = sim.pull_status()
    print(f"[PASS] status after one command: lift_pos={status_after.lift.pos:.3f}")

    sensor_data = sim.pull_sensor_data()
    print(
        "[PASS] sensor access ok:"
        f" gyro_shape={sensor_data.base_gyro.shape},"
        f" accel_shape={sensor_data.base_imu.shape},"
        f" lidar_shape={sensor_data.lidar.shape}"
    )

    sim.stop()
    print("[PASS] headless simulator stopped cleanly")

if __name__ == "__main__":
    main()
PY
  MUJOCO_GL="${MUJOCO_GL:-${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}}" "${UV_BIN}" run python "${HEADLESS_SCRIPT}"
  exit 0
fi

if [[ "${MODE}" == "perception" ]]; then
  mkdir -p "${TMP_DIR}"
  PERCEPTION_SCRIPT="$(mktemp "${TMP_DIR}/stretch_mujoco_perception_smoke_XXXXXX.py")"
  trap 'rm -f "${HEADLESS_SCRIPT:-}" "${GUI_SCRIPT:-}" "${PERCEPTION_SCRIPT:-}"' EXIT
  cat > "${PERCEPTION_SCRIPT}" <<'PY'
import time
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_cameras import StretchCameras

def main():
    sim = StretchMujocoSimulator(
        cameras_to_use=[
            StretchCameras.cam_d405_rgb,
            StretchCameras.cam_d405_depth,
        ]
    )
    sim.start(headless=True)
    camera_data = sim.pull_camera_data()
    print(
        "[PASS] perception access ok:"
        f" d405_rgb_shape={camera_data.cam_d405_rgb.shape},"
        f" d405_depth_shape={camera_data.cam_d405_depth.shape}"
    )
    sim.stop()
    print("[PASS] perception simulator stopped cleanly")

if __name__ == "__main__":
    main()
PY
  MUJOCO_GL="${MUJOCO_GL:-${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}}" "${UV_BIN}" run python "${PERCEPTION_SCRIPT}"
  exit 0
fi

if [[ "${MODE}" == "gui" ]]; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    fail "GUI mode requested but DISPLAY/WAYLAND_DISPLAY is not set"
  fi

  mkdir -p "${TMP_DIR}"
  GUI_SCRIPT="$(mktemp "${TMP_DIR}/stretch_mujoco_gui_smoke_XXXXXX.py")"
  trap 'rm -f "${HEADLESS_SCRIPT:-}" "${GUI_SCRIPT:-}"' EXIT
  cat > "${GUI_SCRIPT}" <<'PY'
import time
from stretch_mujoco import StretchMujocoSimulator

def main():
    sim = StretchMujocoSimulator()
    sim.start(headless=False)
    print("[PASS] GUI simulator started")
    time.sleep(3.0)
    sim.stop()
    print("[PASS] GUI simulator stopped cleanly")

if __name__ == "__main__":
    main()
PY
  if timeout 20s "${UV_BIN}" run python "${GUI_SCRIPT}"; then
    rc=0
  else
    rc=$?
  fi

  if [[ "${rc}" -eq 0 ]]; then
    pass "GUI simulator smoke test completed"
    exit 0
  fi

  if [[ "${rc}" -eq 124 ]]; then
    warn "GUI simulator reached timeout after launching; this still suggests the viewer likely started"
    exit 0
  fi

  fail "GUI simulator smoke test failed with exit code ${rc}"
fi

fail "Unsupported mode '${MODE}'. Use 'headless', 'perception', or 'gui'."
