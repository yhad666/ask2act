# Ask2Act Robot Hardware, Experimental Platform, and Runtime Parameters

Generated: 2026-06-11

This report summarizes the physical robot platform, the A6000-side perception and
planning service, the robot-side transport server, and the main runtime
parameters used by the Ask2Act real-robot experiments. The values below are
drawn from committed configuration and implementation files, especially:

- `services/a6000_web/a6000_real.env`
- `real/stretch_transport/robot_server.env`
- `real/stretch_transport/head_camera_extrinsics.json`
- `real/stretch_transport/README.md`
- `services/a6000_web/server.py`
- `services/a6000_web/grasp_runtime.py`
- `simulation/ask2act_grasp/stretch3_specs.py`
- `simulation/ask2act_grasp/config/grasp_config.yaml`
- `simulation/ask2act_grasp/config/scene_config.yaml`

Runtime UI controls can override some head-pose, gripper, and top-down grasp
tuning values during an experiment. For paper writing, treat this document as a
configuration/provenance summary rather than a claim that every value was
immutable across all trials.

## 1. Physical Robot Platform

The online experiments used a Hello Robot Stretch SE3 / Stretch 3-class mobile
manipulator. In the deployment configuration the robot is addressed on the local
network as:

```text
stretch-se3-3056.local
```

The robot-side server exposes a ZeroMQ REP endpoint:

```text
tcp://0.0.0.0:5557
```

The A6000-side web service connects to it through:

```text
tcp://stretch-se3-3056.local:5557
```

The robot-side execution interface supports two key commands:

- `observe`: move to a camera-safe pose, capture RGB-D data, and return image,
  depth, intrinsics, extrinsics, and metadata.
- `execute_grasp`: execute the waypoint trajectory generated on the A6000-side
  planner.

The robot-side transport layer intentionally does not invent new grasps. It
executes an already planned trajectory and returns execution feedback.

## 2. Manipulator and End-Effector Hardware Model

The motion planner uses Stretch 3-style kinematic and actuator limits from
`simulation/ask2act_grasp/stretch3_specs.py`.

Joint limits:

| Joint | Limit |
| --- | --- |
| lift | 0.00 to 1.10 m |
| arm | 0.00 to 0.52 m |
| wrist_yaw | -1.39 to 4.42 rad |
| wrist_pitch | -1.571 to 0.56 rad |
| wrist_roll | -3.14 to 3.14 rad |
| head_pan | -4.04 to 1.73 rad |
| head_tilt | -1.53 to 0.79 rad |

Nominal velocity limits used by the planner/spec file:

| Motion | Value |
| --- | ---: |
| lift | 0.15 m/s |
| arm | 0.40 m/s |
| mobile base linear | 0.30 m/s |
| mobile base rotation | 1.90 rad/s |
| wrist yaw | 2.98 rad/s |

Force/torque limits recorded in the hardware spec:

| Joint or action | Value |
| --- | ---: |
| lift up | 38.1 N |
| lift down | 55.8 N |
| arm extend | 56.3 N |
| arm retract | 39.5 N |
| wrist yaw torque | 4.56 N m |
| wrist pitch torque | 10.6 N m |
| wrist roll torque | 4.1 N m |

Gripper geometry/model parameters:

| Parameter | Value |
| --- | ---: |
| finger length | 0.257404 m |
| wrist-to-grasp-center distance | 0.271673 m |
| modeled maximum aperture | 0.09 m |
| gripper weight with gripper | 0.695 kg |
| simulation command range | -0.376 to 0.56 |

These values support the top-down IK target solve, the pregrasp and final grasp
offsets, and the FK validation checks used before dispatch.

## 3. Compute and Network Architecture

The online stack is split between a GPU workstation/service and the robot.

### A6000-side service

The A6000-side service runs the FastAPI web server:

```text
services.a6000_web.server:app
host = 0.0.0.0
port = 7862
```

It is launched by `services/a6000_web/run_real_service.sh`, which loads
`services/a6000_web/a6000_real.env`. Its responsibilities are:

1. Receive operator trial setup through the online experiment UI.
2. Request a live RGB-D observation from the Stretch transport server.
3. Run GroundingDINO candidate generation.
4. Build the annotated candidate overlay.
5. Run the VLM target-resolution / clarification protocol.
6. Apply backend question ranking for `proposed_efe`.
7. Gate execution on operator-confirmed target correctness.
8. Run SAM/bbox target point-cloud extraction and geometric grasp planning.
9. Dispatch the planned waypoint trajectory to the robot-side transport server.
10. Record execution and operator confirmation outcomes.

The A6000 service points the VLM client at a local OpenAI-compatible vLLM
endpoint:

```text
ASK2ACT_VLLM_BASE_URL = http://127.0.0.1:8000/v1
ASK2ACT_VLLM_MODEL = mimo-vl
```

The detector and SAM are pinned to `cuda:1` in the recorded real-runtime
configuration:

```text
ASK2ACT_DETECTOR_DEVICE = cuda:1
ASK2ACT_SAM_DEVICE = cuda:1
```

The environment comments note that GPU 0 is kept for the MiMo server while
GroundingDINO and SAM use GPU 1.

### Robot-side service

The Stretch-side service is launched by:

```text
bash real/stretch_transport/run_robot_server.sh
```

It loads `real/stretch_transport/robot_server.env`, listens on ZeroMQ, controls
the physical robot, captures RealSense observations, and records observation and
execution artifacts under:

```text
real/stretch_transport/artifacts/
```

Main robot-side timeouts:

| Parameter | Value |
| --- | ---: |
| command timeout | 120 s |
| observe timeout | 90 s |
| execute timeout | 115 s |
| execute deadline | 90 s |

Main A6000-to-robot client timeouts:

| Parameter | Value |
| --- | ---: |
| observe timeout | 120000 ms |
| execute timeout | 130000 ms |

## 4. Camera, Observation, and Extrinsics

The real-robot observation pipeline uses the robot head camera, configured as an
Intel RealSense D435i device in `robot_server.env`:

```text
STRETCH_D435I_SERIAL = 239122073910
STRETCH_CAMERA_WIDTH = 1280
STRETCH_CAMERA_HEIGHT = 720
STRETCH_CAMERA_FPS = 15
STRETCH_CAMERA_DEPTH = 1
```

For video preview, the robot server uses:

```text
STRETCH_VIDEO_WIDTH = 1280
STRETCH_VIDEO_HEIGHT = 720
STRETCH_VIDEO_FPS = 10
```

The capture script moves the manipulator to a camera-safe pose, opens the
RealSense stream with `pyrealsense2`, captures RGB and aligned depth, and returns:

- RGB image data
- depth image data
- camera intrinsics
- camera extrinsics
- current robot state
- capture metadata

The A6000-side DINO detector rotates the observation clockwise by 90 degrees in
the recorded real-runtime configuration:

```text
ASK2ACT_DINO_ROTATE_CLOCKWISE_90 = 1
```

The same rotation state is tracked in the real point-cloud pipeline so the
selected candidate bbox is mapped back to the corresponding depth-frame region.

### Head Pose

The default head pose for observation/video is:

```text
head_pan = -1.57 rad
head_tilt = -0.68 rad
```

Robot-side head-pose behavior:

| Parameter | Value |
| --- | --- |
| initialize head pose on server start | yes |
| head pose mode | every_observe |
| head pose required | yes |
| settle time | 2.0 s |
| tolerance | 0.15 rad |

The online UI can send manual head-pose commands through
`/api/online/head_pose`. Those changes can persist for future observations and
cause the transport server to recompute the camera extrinsics for the current
head pose.

### Camera Extrinsics

The default extrinsics source is:

```text
real/stretch_transport/head_camera_extrinsics.json
```

Recorded metadata:

```text
source = stretch_se3_urdf_chain
parent_frame = base_link
child_frame = camera_color_optical_frame
pan = -1.57 rad
tilt = -0.68 rad
```

Default camera-to-world matrix:

```text
[
  [ 0.06430943019208016,  0.9970807989814207,  -0.041160387400281266, -0.03233932026503949],
  [ 0.5379218511889357,  -0.06937625791474039, -0.8401351182108838,   -0.04049435509116056],
  [-0.8405381485702113,   0.03188753895048482, -0.5408130967883847,    1.2762098391718435],
  [ 0.0,                  0.0,                  0.0,                   1.0]
]
```

The actual online-trial metadata stores the observation-specific intrinsics and
extrinsics with each captured scene, so paper figures should cite trial metadata
when exact camera calibration is needed.

## 5. Detection and VLM Runtime Parameters

GroundingDINO candidate generation is configured with:

| Parameter | Value |
| --- | ---: |
| box threshold | 0.38 |
| text threshold | 0.30 |
| NMS IoU | 0.50 in detector default |
| max candidates per image | 100 in detector default |
| detector device | cuda:1 |

Detection prompts are built from instruction noun/attribute phrases and
dot-separated category terms. The current runtime can also run broader object
recall for partial prompts when the VLM determines that the single detected
candidate may be insufficient.

The VLM generation settings in the real-runtime environment include:

| Parameter | Value |
| --- | ---: |
| max generation tokens | 4096 |
| think hint enabled | 1 |
| VLM model id | `mimo-vl` |

Both offline and online trials cap clarification at:

```text
ASK2ACT_OFFLINE_MAX_ROUNDS = 6
ASK2ACT_ONLINE_MAX_ROUNDS = 6
```

## 6. Real-Robot Grasp Planning Pipeline

The online execution pipeline uses:

```text
ASK2ACT_PIPELINE_MODE = real_pointcloud
```

After target resolution and operator confirmation, the pipeline:

1. Loads the latest RGB-D observation and camera calibration.
2. Maps the resolved candidate bbox into the depth frame.
3. Optionally predicts a Segment Anything mask from the bbox.
4. Builds a target point cloud from the SAM mask or bbox crop.
5. Filters points using table/object height constraints.
6. Estimates a geometric top-down grasp from the target point cloud.
7. Solves a Stretch top-down motion plan with SimpleIK when available.
8. Dispatches the resulting waypoint trajectory to the robot.

SAM is enabled in the real-runtime configuration:

| Parameter | Value |
| --- | --- |
| use SAM mask | 1 |
| SAM model type | `vit_b` |
| SAM checkpoint | `/home/haoandong/workspace/project/sam/sam_vit_b_01ec64.pth` |
| SAM device | cuda:1 |

Table and object-height filters:

| Parameter | Value |
| --- | --- |
| table top z | auto |
| table clearance margin | 0.005 m |
| max object z above table | 0.22 m |

For tall objects, the top-down grasp height is capped by:

```text
grasp_z = max(center_z, object_top_z - max_top_grasp_delta_m)
```

with:

```text
max_top_grasp_delta_m = 0.07
```

The real runtime requires the calibrated/explicit planning path rather than
silently falling back to a rough approximate top-down grasp:

```text
ASK2ACT_REAL_ALLOW_APPROXIMATE_TOPDOWN_FALLBACK = 0
ASK2ACT_GRASP_APPROXIMATE_FALLBACK = false in grasp_config.yaml
```

## 7. Top-Down Grasp and Compensation Parameters

The real point-cloud planner uses a top-down geometric grasp. The gripper open
command is overridden to:

```text
ASK2ACT_GEOMETRIC_TOP_DOWN_GRIPPER_OPEN_CMD_OVERRIDE = 0.56
```

Top-down grasp z mode:

```text
ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_Z_MODE = center
ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_Z_TOP_CLEARANCE_M = 0.0
```

Regular-object rubber-contact correction:

| Axis | Value |
| --- | ---: |
| X | 0.000 m |
| Y | -0.045 m |
| Z | 0.000 m |

Slender/fork-object rubber-contact correction:

| Axis | Value |
| --- | ---: |
| X | 0.000 m |
| Y | -0.010 m |
| Z | 0.045 m |

Regular side-bias tuning:

| Parameter | Value |
| --- | ---: |
| side x bias | 0.020 m |
| side deadband | 0.050 m |
| right extra x | 0.010 m |
| left/center y | 0.005 m |
| right y | -0.005 m |

Slender-object side-bias tuning:

| Parameter | Value |
| --- | ---: |
| side x bias | 0.010 m |
| side deadband | 0.050 m |
| right extra x | 0.005 m |
| left/center y | 0.000 m |
| right y | 0.000 m |

Slender object classification and wrist-yaw behavior:

| Parameter | Value |
| --- | ---: |
| slender aspect ratio threshold | 2.5 |
| slender max height | 0.065 m |
| slender grip clearance | 0.018 m |
| slender min open width | 0.032 m |
| wrist-yaw open-width threshold | 0.040 m |
| force wrist yaw for slender | 1 |
| slender long-axis bias | 0.000 m |

This is the behavior used for utensil-like objects such as forks/spoons: if the
target is slender or the requested opening is below the threshold, the planner
aligns wrist yaw to the geometric grip angle instead of using the cup-like
default yaw.

Approach and recovery settings:

| Parameter | Value |
| --- | ---: |
| pregrasp clearance | 0.15 m |
| postgrasp lift | 0.30 m |
| return base rotation to start | 1 |
| SimpleIK max FK error | 0.025 m |

## 8. Base Reach and Prepositioning

The motion planner can request base motion when the arm-axis target is outside
the comfortable reach band.

Base reach translation:

| Parameter | Value |
| --- | ---: |
| enabled | 1 |
| max translate | 0.16 m |
| reach margin | 0.02 m |

Base preposition request:

| Parameter | Value |
| --- | ---: |
| goal distance | 0.56 m |
| correction fraction | 0.55 |
| lateral max | 0.06 m |
| longitudinal max | 0.06 m |
| lateral deadband | 0.035 m |
| longitudinal deadband | 0.040 m |
| replan after base reach | 1 |
| max base reach attempts | 2 |
| fail-fast oversized base reach | 1 |
| max total arm-axis correction | 0.15 m |
| target lock ambiguity margin | 0.02 m |

When the planner returns a `preposition_only` result, the robot executes the
base move, reobserves, and replans before attempting the final grasp.

Robot-side base motion limits:

| Parameter | Value |
| --- | ---: |
| forward base translation enabled | 1 |
| max forward translation | 0.10 m |
| arm-axis translation enabled | 1 |
| max arm-axis translation | 0.18 m |
| translation tolerance | 0.02 m |
| base rotation tolerance | 0.035 rad |

## 9. Robot-Side Execution and Home/Safety Behavior

The online experiment deliberately separates target selection from physical
execution. Execution is disabled by default until the operator confirms the
resolved target:

```text
ASK2ACT_AUTO_EXECUTE_ON_RESOLVE = 0
```

If the resolved target does not match the expected target, the online API marks
the trial as `skipped_wrong_target` and does not execute a physical grasp. This
is why the cleaned online experiment has zero wrong-object grasp attempts by
design.

Robot home/camera-safe behavior:

| Parameter | Value |
| --- | --- |
| home on observe | 1 |
| home on execute start | 1 |
| home on execute end | 1 |
| home required | 1 |
| home lift | 0.50 m |
| observe start lift | 0.45 m |
| execute end lift | 0.45 m |
| home arm | 0.00 m |
| home wrist yaw | 0.00 rad |
| home wrist pitch | -1.57 rad |
| home wrist roll | 0.00 rad |
| observe settle | 1.2 s |
| home settle | 2.0 s |
| verify observe home | 1 |
| verify joints | lift, arm |

Gripper behavior in the recorded robot runtime:

| Parameter | Value |
| --- | ---: |
| command mode | real_pct |
| real open command | 100.0 |
| real close command | -80.0 |
| open accept percent | 80.0 |
| close accept percent | 70.0 |
| close verify | 0 |
| close settle | 0.25 s |
| release gripper on execute end | 1 |
| execute-end release wait timeout | 0.0 s |
| execute-end release settle | 0.2 s |

The home routine does not force a gripper action during observe/start/end/failure
in the recorded robot-side config. The successful execution cleanup can release
the gripper after the motion.

## 10. Online Operator Controls

The online UI exposes controls for both experiment operation and real-robot
tuning. Relevant API endpoints include:

| Endpoint | Purpose |
| --- | --- |
| `/api/online/head_pose` | Set head pan/tilt and optionally persist the pose. |
| `/api/online/grasp_tuning` | Read or update top-down grasp tuning parameters. |
| `/api/online/experiments/{id}/video/start` | Start head-camera video preview. |
| `/api/online/experiments/{id}/video/stop` | Stop head-camera video preview. |
| `/api/online/experiments/{id}/video/status` | Query video preview status. |

The grasp tuning UI exposes:

- regular rubber X/Y/Z correction
- approximate fallback X/Y correction
- planner open command
- regular side x bias, side deadband, right extra x, left/center y, right y
- top-distance cap
- fork/slender rubber X/Y/Z correction
- fork/slender side x, right extra x, long-axis bias, left/center y, right y,
  and deadband
- Stretch gripper open, close, and release commands

The server applies planner tuning by updating process environment variables and
motion-planner module state. Stretch gripper command changes are also pushed to
the robot server through a runtime transport configuration call.

## 11. Simulation and Bring-Up Configuration

The repository also contains a simulation/bring-up configuration used for
pipeline development and validation:

```text
simulation/ask2act_grasp/config/scene_config.yaml
simulation/ask2act_grasp/config/grasp_config.yaml
```

Simulation camera/rendering settings:

| Parameter | Value |
| --- | --- |
| render width | 1280 |
| render height | 720 |
| z range | 0.2 to 1.1 m |
| head pan | -1.57 rad |
| head tilt | -0.55 rad |
| camera rate | 2 Hz |

Simulation tabletop scene defaults:

| Object | Value |
| --- | --- |
| table position | [0.0, -0.66, 0.76] |
| table size | [0.30, 0.22, 0.02] |
| cup position | [0.0, -0.60, 0.86] |
| cup radius | 0.035 m |
| cup height | 0.10 m |
| cup mass | 0.08 kg |

Grasp pipeline defaults in `grasp_config.yaml`:

| Parameter | Value |
| --- | --- |
| crop radius | 0.08 m |
| point-cloud subsample | 2048 |
| depth noise sigma | 0.001 |
| dropout ratio | 0.01 |
| pregrasp offset | 0.12 m |
| top-down pitch | -1.57 rad |
| max gripper width | 0.08 m |
| Contact-GraspNet optional | true |
| fallback generator | true |
| SimpleIK top-down | true |
| approximate fallback | false |

The online main experiment used the real-pointcloud pipeline and robot-side
transport described above. The simulation configuration is best cited as
bring-up and validation support unless a specific paper section is discussing
simulation.

## 12. Paper-Ready Platform Summary

The online Ask2Act experiments were run on a Stretch SE3 / Stretch 3-class mobile
manipulator equipped with a head-mounted Intel RealSense D435i RGB-D camera. A
GPU workstation hosted the FastAPI experiment server, GroundingDINO detector,
SAM segmentation, VLM clarification model, and real-pointcloud grasp planner.
The robot ran a separate ZeroMQ transport server that captured RGB-D
observations and executed only the waypoint trajectories produced by the
workstation-side planner. The system used a conservative execution gate:
operator confirmation of the resolved target was required before physical
motion, and trials with incorrect target resolution were skipped rather than
executed. For confirmed targets, the planner used SAM/bbox point-cloud
extraction, a geometric top-down grasp estimate, Stretch SimpleIK, calibrated
rubber-contact offsets, side-aware bias terms, and optional base prepositioning
before dispatching the motion to the robot.
