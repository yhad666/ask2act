from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _forward_if_configured(request_path: Path, response_path: Path) -> bool:
    command = os.getenv("ASK2ACT_STRETCH_EXECUTE_FORWARD_COMMAND", "").strip()
    if not command:
        return False
    completed = subprocess.run(
        ["bash", "-lc", f"{command} {request_path} {response_path}"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "execute forward command failed")
    return True


def _run_bundled_executor(request_path: Path, response_path: Path) -> bool:
    if os.getenv("ASK2ACT_STRETCH_USE_BUNDLED_EXECUTOR", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "dispatch_grasp.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(request_path), str(response_path)],
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0:
        return True
    raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "bundled executor failed")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: execute_hook.py REQUEST_JSON RESPONSE_JSON")

    request_path = Path(argv[1]).expanduser()
    response_path = Path(argv[2]).expanduser()
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    try:
        if _forward_if_configured(request_path, response_path):
            return 0
        if _run_bundled_executor(request_path, response_path):
            return 0
    except Exception as exc:
        response = {
            "ok": False,
            "execution_status": "failed",
            "error": str(exc),
            "note": "Bundled robot executor failed before fallback behavior.",
        }
        response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        return 0

    artifact_dir = Path(
        os.getenv("ASK2ACT_STRETCH_SERVER_ARTIFACT_ROOT", str(Path(__file__).resolve().parents[1] / "artifacts"))
    ).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"latest_execute_request_{stamp}.json"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if payload.get("dry_run"):
        response = {
            "ok": True,
            "execution_status": "dry_run_accepted",
            "artifact_path": str(artifact_path),
            "note": "Dry-run request recorded on the Stretch side.",
        }
    else:
        response = {
            "ok": False,
            "execution_status": "not_implemented",
            "artifact_path": str(artifact_path),
            "error": (
                "Default execute hook does not move the robot. Replace this hook or set "
                "ASK2ACT_STRETCH_EXECUTE_FORWARD_COMMAND before sending non-dry-run requests."
            ),
        }

    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
