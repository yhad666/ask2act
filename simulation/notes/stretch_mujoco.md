# Stretch MuJoCo Notes

This workspace uses the official `hello-robot/stretch_mujoco` repository in:

- `~/robot/stretch_mujoco`

Current intent:

- keep ROS 2 and MoveIt already-installed state intact
- use a separate `uv` environment for the simulator
- verify minimum viable simulation in headless mode first
- keep real-robot fleet configuration out of the simulation workflow

Primary scripts:

- `~/robot/scripts/source_sim.sh`
- `~/robot/scripts/setup_stretch_mujoco.sh`
- `~/robot/scripts/smoke_test_sim.sh`
