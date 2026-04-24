from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_image_path() -> Path:
    return (
        _repo_root()
        / "simulation"
        / "logs"
        / "sim_validation"
        / "data_samples"
        / "tabletop_scene"
        / "head_d435i_rgb.png"
    )


def _forward_if_configured(request_path: Path, response_path: Path) -> bool:
    command = os.getenv("ASK2ACT_STRETCH_OBSERVE_FORWARD_COMMAND", "").strip()
    if not command:
        return False
    completed = subprocess.run(
        ["bash", "-lc", f"{command} {request_path} {response_path}"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "observe forward command failed")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: observe_hook.py REQUEST_JSON RESPONSE_JSON")

    request_path = Path(argv[1]).expanduser()
    response_path = Path(argv[2]).expanduser()
    request = json.loads(request_path.read_text(encoding="utf-8"))

    if _forward_if_configured(request_path, response_path):
        return 0

    raw_image_path = os.getenv("ASK2ACT_STRETCH_OBSERVATION_IMAGE_PATH", "").strip()
    image_path = Path(raw_image_path).expanduser() if raw_image_path else _default_image_path()
    if not image_path.exists():
        response = {
            "ok": False,
            "error": f"Observation image not found: {image_path}",
            "note": "Set ASK2ACT_STRETCH_OBSERVATION_IMAGE_PATH or ASK2ACT_STRETCH_OBSERVE_FORWARD_COMMAND.",
        }
    else:
        response = {
            "ok": True,
            "observation_id": f"stretch-{request.get('session_id', 'session')}-{int(time.time())}",
            "mime_type": mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
            "image_path": str(image_path),
            "source": "default_observe_hook",
            "note": (
                "Default hook is serving a local image. Replace this hook or set "
                "ASK2ACT_STRETCH_OBSERVE_FORWARD_COMMAND to capture from the real robot camera."
            ),
        }
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
