# Stretch-side Ask2Act Transport

This folder is the robot-side counterpart to `services/a6000_web/`.

It provides one small ZMQ `REP` server that the A6000 talks to directly:

- `observe`
- `execute_grasp`

The design goal is:

- one command to launch on Stretch after you pull the repo
- safe defaults for bring-up
- clear hook points for your real camera and motion code

## Files

- `robot_server.py`
  Stretch-side ZMQ server.
- `runtime.py`
  Shared request handling and hook execution.
- `hooks/observe_hook.py`
  Default observation hook. It now tries the bundled D435i capture script first.
- `hooks/execute_hook.py`
  Default execution hook. It now tries the bundled robot dispatch script first.
- `scripts/capture_observation.py`
  Bundled real-robot D435i capture path using `pyrealsense2`.
- `scripts/dispatch_grasp.py`
  Bundled robot-side dispatcher. It accepts dry-runs and can execute a waypoint trajectory with `stretch_body`.
- `scripts/generate_head_camera_extrinsics.py`
  Regenerates `head_camera_extrinsics.json` from the Stretch SE3 URDF chain for the configured head pan/tilt.
- `run_robot_server.sh`
  One-command launcher.
- `robot_server.env.example`
  Optional environment template.

## One-command launch on Stretch

After pulling the repo onto the robot:

```bash
cd /path/to/ask2act
bash real/stretch_transport/run_robot_server.sh
```

That is the one command to keep using once the environment is set up:

```bash
bash real/stretch_transport/run_robot_server.sh
```

The launcher auto-installs `pyzmq` on first boot if the selected Python does not
have it yet. If you want to disable that behavior, set:

```bash
export ASK2ACT_STRETCH_SERVER_AUTO_INSTALL=0
```

The server listens on:

```text
tcp://0.0.0.0:5557
```

by default.

The command hooks have explicit failure exits. The real config currently uses
90 seconds for observation, 115 seconds for execution, and an internal
90-second execution deadline before the dispatcher reports failure and tries to
return to the default pose.

## Safe defaults

If you launch with the bundled defaults:

- `observe` uses `hooks/observe_hook.py`
- the observe hook first tries `scripts/capture_observation.py`
- `execute_grasp` uses `hooks/execute_hook.py`
- the execute hook first tries `scripts/dispatch_grasp.py`
- the dispatch script accepts `dry_run=true`
- the dispatch script can execute a real waypoint trajectory when the request contains `grasp_plan.trajectory`

This makes the first bring-up practical: you can verify real observation immediately, and still keep execution safe until the A6000 is sending a real waypoint trajectory.

## Real-robot wiring

There are two easy ways to connect your actual Stretch code while keeping the same server entrypoint.

### Option A. Use the bundled robot scripts

Copy:

```bash
cp real/stretch_transport/robot_server.env.example real/stretch_transport/robot_server.env
```

Then minimally set the real camera serial:

```bash
ASK2ACT_STRETCH_D435I_SERIAL=239122073910
```

For the current tabletop flow, the bundled observation script can also force the
head D435i to a known tabletop pose before every observation:

```bash
ASK2ACT_STRETCH_INIT_HEAD_POSE_MODE=every_observe
ASK2ACT_STRETCH_INIT_HEAD_POSE_REQUIRED=1
ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD=-1.57
ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD=-0.68
ASK2ACT_STRETCH_INIT_HEAD_SETTLE_S=2.0
ASK2ACT_STRETCH_INIT_HEAD_TOLERANCE_RAD=0.15
ASK2ACT_STRETCH_HOME_POSE_ON_OBSERVE=0
ASK2ACT_STRETCH_HOME_GRIPPER_ON_OBSERVE=0
ASK2ACT_STRETCH_HOME_GRIPPER_ON_EXECUTE_START=0
ASK2ACT_STRETCH_HOME_GRIPPER_ON_EXECUTE_END=0
ASK2ACT_STRETCH_HOME_GRIPPER_ON_EXECUTE_FAILURE=0
```

For the online experiment, keep `ASK2ACT_STRETCH_INIT_HEAD_POSE_MODE=every_observe`.
That makes each trial actively point the head camera at the tabletop before
capturing RGB-D. `HOME_POSE_ON_OBSERVE=0` avoids repeated arm/lift/wrist homing
before every image, and the gripper flags leave the gripper unchanged during
observe/home and execute-start/end/failure homing. The grasp trajectory still
opens/closes the gripper when an actual grasp is executed. Wrong target
selections should be skipped on the A6000 `/online` console before any execute
request is sent to the robot.

After that, a single command is enough:

```bash
bash real/stretch_transport/run_robot_server.sh
```

### Option B. Forward to your own scripts

If your lab already has mature robot-side code, point the env file at it:

```bash
ASK2ACT_STRETCH_OBSERVE_FORWARD_COMMAND="python /abs/path/to/capture_observation.py"
ASK2ACT_STRETCH_EXECUTE_FORWARD_COMMAND="python /abs/path/to/dispatch_grasp.py"
```

Each forward command is still called with:

1. request JSON path
2. response JSON path

The forward script should write a JSON object to the response path.

### Option C. Edit the bundled hooks in place

If you prefer to keep everything in this repo, edit:

- `real/stretch_transport/hooks/observe_hook.py`
- `real/stretch_transport/hooks/execute_hook.py`

and replace the default file-backed / no-op behavior with your robot camera and motion calls.

## Request / response contracts

### Observe request from A6000

```json
{
  "op": "observe",
  "session_id": "SESSION_ID",
  "instruction": "Pick up my cup."
}
```

### Observe response back to A6000

Your hook can return any of these:

- `image_path`
- `image_base64`
- `image_data_url`

Recommended minimal shape:

```json
{
  "ok": true,
  "observation_id": "obs-001",
  "mime_type": "image/jpeg",
  "image_path": "/abs/path/to/latest_rgb.jpg"
}
```

### Execute request from A6000

```json
{
  "op": "execute_grasp",
  "session_id": "SESSION_ID",
  "instruction": "Pick up my cup.",
  "observation_id": "obs-001",
  "resolved_target": {
    "candidate_id": "cand_003",
    "bbox_xyxy": [120.5, 240.0, 360.5, 510.0],
    "mask_rle": null
  },
  "grasp_plan": {
    "pipeline_mode": "mock",
    "planner_backend": "mock-preview",
    "target_bbox_2d": [120, 240, 360, 510]
  },
  "dry_run": false
}
```

### Execute response back to A6000

```json
{
  "ok": true,
  "execution_status": "completed"
}
```

## Bundled real-robot behavior

### Observation

`scripts/capture_observation.py`:

- moves the manipulator to the configured default pose before capture; by default
  the wrist pitch is `-1.57` rad so the gripper points down instead of blocking
  the head camera view
- opens the head D435i through `pyrealsense2`
- captures one RGB frame
- captures depth by default as `.npy`, aligned to the color frame so RGB
  detections and depth crops share the same pixel coordinates
- uses the configured RealSense stream size for both RGB and depth, currently
  `1280x720` in `robot_server.env`
- saves it under `real/stretch_transport/artifacts/observations/`
- returns `image_path`; the transport server also inlines the depth npy as
  `depth_npy_base64` so the A6000 can plan from the same observation
- also returns `depth_npy_path`, `rgb_shape_hw`, `depth_shape_hw`,
  `depth_aligned_to_color`, and `camera_intrinsics`

It uses:

- `ASK2ACT_STRETCH_D435I_SERIAL`
- `ASK2ACT_STRETCH_CAMERA_WIDTH`
- `ASK2ACT_STRETCH_CAMERA_HEIGHT`
- `ASK2ACT_STRETCH_CAMERA_FPS`

### Head-camera video during online experiments

The robot server also supports experiment video through the same D435i stream:

- `start_head_camera_video` opens one high-resolution RGB-D RealSense pipeline.
- while video is active, `observe` reuses that same stream for the trial RGB-D
  frame instead of opening a second camera pipeline
- the experiment observation resolution remains controlled by
  `ASK2ACT_STRETCH_CAMERA_WIDTH` / `ASK2ACT_STRETCH_CAMERA_HEIGHT`
- video resolution defaults to the same size, currently `1280x720`, and is
  configured separately with `ASK2ACT_STRETCH_VIDEO_WIDTH` and
  `ASK2ACT_STRETCH_VIDEO_HEIGHT`
- `stop_head_camera_video` transfers the MP4 back to A6000; the temporary
  robot-side MP4 is deleted after transfer

For the online experiment, keep:

```bash
ASK2ACT_STRETCH_OBSERVE_FROM_VIDEO_STREAM=1
ASK2ACT_STRETCH_VIDEO_WIDTH=1280
ASK2ACT_STRETCH_VIDEO_HEIGHT=720
ASK2ACT_STRETCH_VIDEO_FPS=10
ASK2ACT_STRETCH_VIDEO_INIT_HEAD_POSE_ON_START=1
```

If video startup fails because `ffmpeg` or OpenCV is missing, leave video off
and run the experiment normally; the non-video observation path is unchanged.

### Execution

`scripts/dispatch_grasp.py`:

- moves the manipulator to the configured default pose before execution
- accepts `dry_run=true` without moving the robot
- records every request and response under `real/stretch_transport/artifacts/executions/`
- if `grasp_plan.trajectory` is present, it will try to execute the waypoints with `stretch_body`
- moves the manipulator back to the configured default pose after execution; by default the gripper is not opened after execution
- records before/after Stretch status for the default pose and each waypoint so
  failed or stalled grasps can be debugged from `execute_response_*.json`
- exits with a clear failure once `ASK2ACT_STRETCH_EXECUTE_DEADLINE_S` is
  exceeded, then attempts the configured failure default pose

Important:

- this bundled executor does not invent a grasp on the robot side
- it only executes a waypoint trajectory that the A6000 already produced
- if the A6000 sends no trajectory, the robot will reject live execution with a clear error

## What the robot-side scripts need to do

### Observation command

Read the request JSON and write a response JSON with:

- `ok`
- `observation_id`
- `mime_type`
- one of `image_path`, `image_base64`, `image_data_url`

### Execution command

Read the request JSON and write a response JSON with:

- `ok`
- `execution_status`
- optional `error`

## Suggested bring-up order

1. Launch `run_robot_server.sh` on Stretch with defaults.
2. On the A6000, run the ZMQ smoke test in `services/a6000_web/dev/stretch_zmq_smoke_test.py`.
3. Start the A6000 browser service.
4. Verify `fetch_observation` works.
5. Verify `dry_run=true` execute works.
6. Wire your real camera hook.
7. Wire your real execute hook.
8. Only then allow non-dry-run grasp dispatch.
