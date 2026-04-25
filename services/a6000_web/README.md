# Ask2Act A6000 Service

This service is now the main Ask2Act orchestrator on the A6000.

Responsibilities:

- browser UI
- fresh observation intake from Stretch
- GroundingDINO candidate generation
- VLM/EFE clarification loop
- local target-to-bbox handoff into the Ask2Act grasp pipeline
- robot command dispatch back to Stretch

Stretch stays minimal:

- sensing
- low-level execution
- transport endpoint only

## Launch

### 1. Start MiMo on the A6000

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 \
/home/haoandong/workspace/project/glm_ui/venv_vllm/bin/vllm serve \
/home/haoandong/workspace/project/mimo/models/MiMo-VL-7B-RL-2508 \
  --host 127.0.0.1 --port 8000 \
  --served-model-name mimo-vl \
  --dtype bfloat16 \
  --runner generate \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image": 1}' \
  --generation-config vllm \
  --disable-log-requests \
  --allowed-local-media-path /home/haoandong/workspace/project/mimo/test
```

### 2. Install dependencies

```bash
python -m pip install -r services/a6000_web/requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure the A6000 orchestrator

For no-robot validation:

```bash
export ASK2ACT_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export ASK2ACT_VLLM_MODEL=mimo-vl
export ASK2ACT_DETECTOR_DEVICE=cuda:1
export ASK2ACT_DINO_BOX_THRESHOLD=0.38
export ASK2ACT_GEN_MAX_TOKENS=8000
export ASK2ACT_THINK_HINT=1
export ASK2ACT_STRETCH_TRANSPORT=mock
export ASK2ACT_PIPELINE_MODE=mock
```

For direct Stretch integration:

```bash
export ASK2ACT_STRETCH_TRANSPORT=zmq
export ASK2ACT_STRETCH_ZMQ_ENDPOINT=tcp://STRETCH_HOST:5557
export ASK2ACT_STRETCH_TIMEOUT_MS=120000
export ASK2ACT_PIPELINE_MODE=real_pointcloud
export ASK2ACT_HEAD_CAMERA_EXTRINSICS_PATH=/abs/path/to/head_camera_extrinsics.json
export ASK2ACT_REAL_ALLOW_APPROXIMATE_TOPDOWN_FALLBACK=1
```

For the real-robot browser service, you can also copy:

```bash
cp services/a6000_web/a6000_real.env.example services/a6000_web/a6000_real.env
```

and then edit `services/a6000_web/a6000_real.env` once. After that, the A6000 service becomes a one-command launch:

```bash
bash services/a6000_web/run_real_service.sh
```

Optional:

```bash
export ASK2ACT_STRETCH_MOCK_IMAGE_PATH=/abs/path/to/demo.jpg
export ASK2ACT_PIPELINE_HEADLESS=1
export ASK2ACT_PIPELINE_SHOW_VIEWER=0
export ASK2ACT_PIPELINE_RUN_ROOT=/abs/path/to/run_logs
export ASK2ACT_SCENE_CONFIG_PATH=/abs/path/to/scene_config.yaml
export ASK2ACT_GRASP_CONFIG_PATH=/abs/path/to/grasp_config.yaml
export ASK2ACT_REAL_MIN_POINT_CLOUD_COUNT=30
```

### 4. Launch the A6000 web service

Use port `7862`.

```bash
bash services/a6000_web/run_real_service.sh
```

### 5. Launch the Stretch-side server

The Stretch server is now included in this repo:

```bash
bash real/stretch_transport/run_robot_server.sh
```

See [`real/stretch_transport/README.md`](../../real/stretch_transport/README.md) for the robot-side details.

### 6. Run the ZMQ smoke test before opening the UI

This checks the Stretch link without needing the browser flow:

```bash
PYTHONPATH=/home/haoandong/ask2act \
ASK2ACT_STRETCH_ZMQ_ENDPOINT=tcp://STRETCH_HOST:5557 \
python services/a6000_web/dev/stretch_zmq_smoke_test.py
```

## Browser Flow

1. Type one instruction.
2. Optionally attach a local image override for debugging.
3. Press `Start Session`.
4. The A6000 service:
   - fetches the newest observation from Stretch when no browser override is provided
   - extracts instruction phrases with spaCy
   - runs GroundingDINO candidate detection
   - starts the MiMo clarification loop
5. The UI shows one yes/no question at a time.
6. Each answer updates the candidate belief state.
7. Once resolved, press:
   - `Preview Payload` for a dry run
   - `Execute Grasp` to plan from the latest head D435i depth frame and dispatch the trajectory to Stretch

## Operator Flow

1. Start the Stretch transport server.
2. Start MiMo on the A6000.
3. Run the ZMQ smoke test once.
4. Start the A6000 web service on port `7862`.
5. Open the browser page.
6. Enter one instruction.
7. Answer clarification questions.
8. Preview or execute the final grasp handoff.

## API Routes

- `GET /`
  Browser UI.

- `GET /health`
  Service health and config summary.

- `POST /api/sessions/start`
  Start a new session.

  Request:

  ```json
  {
    "instruction": "Pick up my cup.",
    "fetch_observation": true,
    "observation_image_data_url": null
  }
  ```

- `GET /api/sessions/{session_id}`
  Fetch the current session state.

- `POST /api/sessions/{session_id}/step`
  Submit one yes/no clarification answer.

  ```json
  {
    "answer": "y"
  }
  ```

- `POST /api/sessions/{session_id}/execute`
  Run the local target handoff flow and dispatch to Stretch.

  ```json
  {
    "dry_run": false
  }
  ```

## Session State

Each in-memory session stores:

- `instruction`
- `instruction_phrases`
- `detection_prompt`
- `observation_id`
- `observation_source`
- `observation_image_data_url`
- `candidate_overlay_data_url`
- `candidates`
- `current_round`
- `current_questions`
- `current_question`
- `question_history`
- `resolved_target`
- `grasp_plan_result`
- `final_image_data_url`
- `execution_result`

## Stretch Observation Contract

The A6000 sends:

```json
{
  "op": "observe",
  "session_id": "SESSION_ID",
  "instruction": "Pick up my cup."
}
```

Expected Stretch reply:

```json
{
  "ok": true,
  "observation_id": "obs-001",
  "mime_type": "image/jpeg",
  "image_base64": "...",
  "depth_npy_base64": "...",
  "depth_scale_m_per_unit": 0.001,
  "camera_intrinsics": {
    "width": 1280,
    "height": 720,
    "fx": 913.0,
    "fy": 913.0,
    "cx": 640.0,
    "cy": 360.0
  }
}
```

The bundled robot-side default hook returns a local image and keeps the server safe for first bring-up. Replace it or set `ASK2ACT_STRETCH_OBSERVE_FORWARD_COMMAND` on Stretch to use the real camera.

## Stretch Execute Contract

The A6000 sends:

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

The bundled default execute hook accepts dry-runs and records live requests, but it intentionally rejects non-dry-run execution until you wire a real motion command on Stretch.

## Resolved Target Schema

```json
{
  "candidate_id": "cand_003",
  "bbox_xyxy": [120.5, 240.0, 360.5, 510.0],
  "mask_rle": null
}
```

## One-command summary

Stretch:

```bash
cd /path/to/ask2act
bash real/stretch_transport/run_robot_server.sh
```

A6000:

```bash
cd /path/to/ask2act
bash services/a6000_web/run_real_service.sh
```
