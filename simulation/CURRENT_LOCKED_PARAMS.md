# Current Locked Params

This file records the parameters that are currently considered usable for the `ask2act` single-cup simulation pipeline.

If a future chat or agent continues this work, treat these values as the default locked baseline. Do not change them casually unless there is a clear reason and the change is documented.

## Head Camera

- `head_pan_rad = -1.57`
- `head_tilt_rad = -0.55`

These are loaded from:

- [scene_config.yaml](/home/yhad/robot/ask2act/simulation/ask2act_grasp/config/scene_config.yaml)

They are commanded in:

- [head_alignment.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/perception/head_alignment.py)

Important:

- The current pipeline does not infer head pan/tilt from perception.
- The head is explicitly commanded to these values before capture.
- If the robot does not visually reach these values in simulation, debug the simulator/control path first. Do not silently "fix" it by changing the configured angles.

## Cup Z Handling

The cup world pose is currently treated as known in the single-cup oracle path.

- `cup_position_m = [0.00, -0.60, 0.85]` in scene config
- `table_position_m = [0.0, -0.66, 0.76]`
- `table_size_m = [0.30, 0.22, 0.02]`
- `table_top_z_m = table_position_m[2] + table_size_m[2] = 0.78`

The current single-cup grasp target does not estimate cup height from RGB-D.
It computes the grasp height from known scene geometry:

- `cup_grasp_z = table_top_z_m + cup_height_m * oracle_grasp_height_ratio`

Relevant files:

- [grasp_executor.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/execution/grasp_executor.py)
- [motion_planner.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/planning/motion_planner.py)

## Current Grasp Height Baseline

These values are loaded from:

- [grasp_config.yaml](/home/yhad/robot/ask2act/simulation/ask2act_grasp/config/grasp_config.yaml)

Current baseline:

- `oracle_grasp_height_ratio = 0.30`
- `oracle_approach_height_offset_m = 0.015`
- `oracle_postgrasp_lift_delta_m = 0.10`
- `oracle_side_grasp_wrist_pitch_rad = 0.01`
- `oracle_arm_backoff_m = 0.18`
- `oracle_final_arm_delta_m = 0.04`
- `oracle_tucked_wrist_yaw_rad = 2.50`

Interpretation:

- `0.0` means grasp at the cup bottom / tabletop reference.
- `1.0` means grasp at the cup top.
- `0.30` means clearly below the cup midline.

If grasping is still too high, try reducing:

- `oracle_grasp_height_ratio` to `0.26`

If approach is too high before final closure, try reducing:

- `oracle_approach_height_offset_m` to `0.010`

If the gripper stops short of the cup, try:

- reducing `oracle_arm_backoff_m` to `0.16`
- or increasing `oracle_final_arm_delta_m` to `0.05`

If the wrist clips the table edge during the early phase, keep:

- `oracle_tucked_wrist_yaw_rad` in the tucked range first
- only rotate back toward the grasp yaw after the lift is already safe

## Start Pose Baseline

The current startup tuck is set in:

- [grasp_executor.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/execution/grasp_executor.py)

Current baseline:

- `gripper = 0.045`
- `wrist_roll = 0.0`
- `wrist_yaw = oracle_tucked_wrist_yaw_rad`
- `wrist_pitch = 0.0`
- `arm = 0.0`

These are intended to create a conservative retracted pose before head alignment and any grasp motion.
The startup phase should tuck and retract first, not lift first.

## Guidance For Future Chats

Before changing anything, first check:

1. Is the issue caused by a wrong target value, or by the simulator failing to reach the commanded value?
2. Is the issue in known-geometry grasping logic, or in perception?
3. Is the issue in arm extension / wrist orientation, rather than in grasp height?

Prefer this debugging order:

1. Confirm commanded vs actual `head_pan/head_tilt`
2. Confirm arm is truly retracted at startup
3. Confirm lift clears the table before extension
4. Only then tune grasp height ratio and wrist pitch
