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
  Default observation hook. Serves a local image unless you wire a real capture command.
- `hooks/execute_hook.py`
  Default execution hook. Accepts dry-runs and records payloads; it rejects live execution until you wire a real command.
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

## Safe defaults

If you launch with the bundled defaults:

- `observe` uses `hooks/observe_hook.py`
- the observe hook serves a local image from the repo or from `ASK2ACT_STRETCH_OBSERVATION_IMAGE_PATH`
- `execute_grasp` uses `hooks/execute_hook.py`
- the execute hook accepts `dry_run=true`
- the execute hook records live requests to `real/stretch_transport/artifacts/` and returns `ok=false` for non-dry-run requests

This makes the first bring-up safe: you can verify the A6000 can reach Stretch without moving the robot.

## Real-robot wiring

There are two easy ways to connect your actual Stretch code while keeping the same server entrypoint.

### Option A. Forward to your own scripts

Copy:

```bash
cp real/stretch_transport/robot_server.env.example real/stretch_transport/robot_server.env
```

Then fill in:

```bash
ASK2ACT_STRETCH_OBSERVE_FORWARD_COMMAND="python /abs/path/to/capture_observation.py"
ASK2ACT_STRETCH_EXECUTE_FORWARD_COMMAND="python /abs/path/to/dispatch_grasp.py"
```

Each forward command is called with:

1. request JSON path
2. response JSON path

The forward script should write a JSON object to the response path.

### Option B. Edit the bundled hooks in place

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
