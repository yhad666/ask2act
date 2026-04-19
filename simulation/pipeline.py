from __future__ import annotations

import os
import sys
from pathlib import Path


SIMULATION_ROOT = Path(__file__).resolve().parent
STRETCH_VENV_PYTHON = SIMULATION_ROOT / "stretch_mujoco" / ".venv" / "bin" / "python"

if STRETCH_VENV_PYTHON.exists():
    current = Path(sys.executable)
    target = STRETCH_VENV_PYTHON
    if current != target:
        os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])

if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

from ask2act_grasp.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
