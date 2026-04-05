# Simulation Workspace

This directory packages the simulation-side work for `ask2act`.

It is designed to live at:

- `ask2act/simulation`

and expects the official upstream Stretch MuJoCo repository to be checked out at:

- `ask2act/simulation/stretch_mujoco`

## Included Here

- `scripts/`: reusable shell entrypoints for setup, smoke tests, data capture, motion validation, and scene snapshots
- `sim_grasping/`: Python utilities, tabletop scene template, and validation helpers
- `notes/`: Chinese validation reports and scene inventory
- `logs/sim_validation/`: sample outputs, screenshots, and motion summary artifacts

## Quick Start

1. Clone the upstream simulator into `simulation/stretch_mujoco`
2. Run `scripts/setup_stretch_mujoco.sh`
3. Run `scripts/run_stretch_mujoco_acceptance.sh`

## Upstream Dependency

This repo does not vendor the full upstream `hello-robot/stretch_mujoco` clone.

Use:

- `scripts/bootstrap_stretch_mujoco.sh`

or manually clone:

```bash
cd simulation
git clone --recurse-submodules https://github.com/hello-robot/stretch_mujoco.git
./scripts/setup_stretch_mujoco.sh
```

## Notes About Paths

The scripts in this folder are path-relative and already adapted to run from the `simulation/` root.

Some reports under `notes/` still mention the original local absolute paths from the development machine for traceability. In this repository, use the corresponding paths under `simulation/`.
