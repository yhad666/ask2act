# Ask2Act Simulation

`simulation/` now has a clean primary path for the single-cup grasp pipeline under `ask2act_grasp/`.

## Structure

- `stretch_mujoco/`: fresh upstream simulator checkout, kept out of git history
- `ask2act_grasp/`: new modular pipeline package
- `scripts/`: bootstrap, setup, and runnable entrypoints
- `pipeline.py`: top-level launcher so the pipeline can be started with `python pipeline.py`
- `logs/`: runtime artifacts only

## Main Flow

```bash
cd simulation
./scripts/bootstrap_stretch_mujoco.sh
./scripts/run_single_cup_head_world_demo.sh
```

Or, if your ROS2 / viewer environment is already sourced:

```bash
cd simulation
python pipeline.py --show-viewer-ui
```

## Package Layout

```text
ask2act_grasp/
├── config/
├── execution/
├── grasp/
├── perception/
├── planning/
├── scene/
├── tests/
└── utils/
```

## Current Planner Notes

- The planner interface is ready for MoveIt2, but the active backend is a geometry-aware fallback planner that keeps the same module boundary.
- Contact-GraspNet is supported through a wrapper when the repo is present at the configured path; otherwise the package falls back to a point-cloud-driven heuristic generator so the pipeline remains runnable while CGN is being wired in.

## Lab Machine Handoff

Recommended target machine:

- native Linux
- NVIDIA GPU
- ROS 2 Humble already installed or easy to source
- enough RAM / SSD headroom for MuJoCo, ROS2, and Contact-GraspNet checkpoints

Bootstrap on a fresh machine:

```bash
git clone https://github.com/yhad666/ask2act.git
cd ask2act/simulation
./scripts/bootstrap_stretch_mujoco.sh
```

This does the following:

- clones a fresh upstream `stretch_mujoco` checkout into `simulation/stretch_mujoco`
- creates the local `.venv` through `uv sync`
- installs the extra Python config dependency `PyYAML`

Run the current pipeline:

```bash
cd simulation
python3 pipeline.py --headless
```

Useful diagnostics:

```bash
cd simulation
python3 scripts/diag_head_motion.py --camera-profile none --timeout-s 20
python3 scripts/diag_head_motion.py --camera-profile head --timeout-s 20
```

## Current Status

Working now:

- fresh upstream `stretch_mujoco` bootstrap
- modular `ask2act_grasp/` package structure
- generated runtime scene for table + free-body cup
- top-level runnable `python pipeline.py`
- headless EGL startup
- unit tests for scene generation, point cloud generation, fallback grasp generation, and grasp selection

Known blockers:

- head motion is reliable with cameras disabled
- head motion degrades sharply when RGB-D rendering is enabled in this VM
- because head alignment does not consistently finish under current VM load, the head point cloud can still come back empty
- MoveIt2 and Contact-GraspNet are not yet fully integrated end-to-end; the codebase currently exposes those interfaces and uses a geometry-aware fallback planner and fallback grasp generator when needed

## What To Check On The Lab Machine

The first checks to run on the stronger Linux box are:

1. `python3 scripts/diag_head_motion.py --camera-profile none --timeout-s 20`
2. `python3 scripts/diag_head_motion.py --camera-profile head --timeout-s 20`
3. `python3 pipeline.py --headless`

If step 2 reaches the requested head angles on the lab machine, the current main blocker in this VM is confirmed to be environment performance rather than the control stack itself.
