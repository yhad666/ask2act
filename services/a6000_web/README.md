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
  --prefix-caching-hash-algo sha256 \
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
export ASK2ACT_GEN_MAX_TOKENS=4096
export ASK2ACT_GEN_MAX_TOKENS_CAP=4096
export ASK2ACT_THINK_HINT=1
export ASK2ACT_STRETCH_TRANSPORT=mock
export ASK2ACT_PIPELINE_MODE=mock
```

For direct Stretch integration:

```bash
export ASK2ACT_STRETCH_TRANSPORT=zmq
export ASK2ACT_STRETCH_ZMQ_ENDPOINT=tcp://STRETCH_HOST:5557
export ASK2ACT_STRETCH_TIMEOUT_MS=120000
export ASK2ACT_STRETCH_OBSERVE_TIMEOUT_MS=120000
export ASK2ACT_STRETCH_EXECUTE_TIMEOUT_MS=130000
export ASK2ACT_PIPELINE_MODE=real_pointcloud
export ASK2ACT_HEAD_CAMERA_EXTRINSICS_PATH=/abs/path/to/head_camera_extrinsics.json
export ASK2ACT_REAL_ALLOW_APPROXIMATE_TOPDOWN_FALLBACK=0
# Keep this off for normal robot runs; it prevents coarse fallback execution
# when SimpleIK or its dependencies are missing.
export ASK2ACT_REAL_TABLE_TOP_Z_M=auto
export ASK2ACT_REAL_TABLE_CLEARANCE_MARGIN_M=0.005
export ASK2ACT_REAL_OBJECT_Z_MAX_ABOVE_TABLE_M=0.22
export ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_Z_MODE=center
export ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M=0.0
export ASK2ACT_GEOMETRIC_TOP_DOWN_PREGRASP_CLEARANCE_M=0.15
export ASK2ACT_GEOMETRIC_TOP_DOWN_POSTGRASP_LIFT_M=0.30
export ASK2ACT_GEOMETRIC_TOP_DOWN_RETURN_BASE_ROTATE_TO_START=1
export ASK2ACT_GEOMETRIC_TOP_DOWN_SIMPLEIK_MAX_FK_ERROR_M=0.025
export ASK2ACT_GEOMETRIC_TOP_DOWN_ENABLE_BASE_REACH_TRANSLATE=1
export ASK2ACT_GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MAX_M=0.16
export ASK2ACT_GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MARGIN_M=0.02
export ASK2ACT_REAL_REPLAN_AFTER_BASE_REACH=1
export ASK2ACT_REAL_BASE_REACH_REPLAN_MAX_ATTEMPTS=2
export ASK2ACT_AUTO_EXECUTE_ON_RESOLVE=0
export ASK2ACT_SESSION_RECORD_ROOT=/abs/path/to/session_records
export ASK2ACT_ONLINE_EXPERIMENT_ROOT=/abs/path/to/online_experiments
```

For online robot experiments, keep `ASK2ACT_AUTO_EXECUTE_ON_RESOLVE=0`.
The `/online` console performs an explicit target-correctness gate before any
physical execution, so wrong target selections are logged and skipped instead
of being sent to Stretch.

For real runs, `ASK2ACT_REAL_TABLE_TOP_Z_M=auto` estimates the tabletop/object
support height from the target bbox depth in the current observation. If you
measure the physical tabletop height in the robot base/world frame, you can set
that numeric value instead. The simulation default is 0.78 m, which is too high
for a low coffee table and can filter out all target points.

For real top-down grasps, the default contact target remains the geometric
point-cloud grasp height, while approximate fallback converts that contact
height into a wrist/lift command with the calibrated top-down gripper length.
The pregrasp/postgrasp lift clearances are 15 cm.
If the target is slightly beyond the arm limit, the real trajectory inserts a
small base reach adjustment. The A6000 executes that adjustment by itself,
fetches a new observation, reselects the target, replans from the new RGB-D
frame, and only then sends the final grasp trajectory. If the arm can already
reach, no base adjustment is sent.

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

The original live UI is available at `/`. The online experiment console is available at `/online`.

For the online experiment, use `/online`; it records 20 physical scenes × 3 prompts × 4 methods while fetching a fresh robot observation for every trial. Registered scene images are only references for the operator and are not reused by the trial.

The online console is intentionally operator-gated:

1. Register or load an online experiment.
2. Register the current physical scene.
3. Start one live trial; Stretch turns the head to the configured tabletop pose and captures a new RGB-D observation.
4. Answer clarification questions if the selected method asks them.
5. Enter the expected candidate display id from the overlay.
6. Press `Execute If Correct`; if target resolution picked the wrong candidate, the trial is logged and the robot does not move.
7. Only if the target is correct does the service dispatch the grasp trajectory, then you confirm correct grasp, grasp failure, or wrong-object grasp.

For the original single-session UI:

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
7. When only one candidate is detected, clarification is skipped.
8. Once resolved, use preview/execute from the UI. For online experiments keep `ASK2ACT_AUTO_EXECUTE_ON_RESOLVE=0` so execution is always gated by the operator.

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

- `POST /api/sessions/{session_id}/confirm`
  Record the operator-confirmed physical result after execution.

  ```json
  {
    "success": true,
    "note": "cup lifted cleanly",
    "reset_ready": true
  }
  ```

  Records are written under `ASK2ACT_SESSION_RECORD_ROOT` as one JSON file per
  trial plus a `session_records.jsonl` index.

## Offline Experiment Console

Open `GET /offline` for the offline large-scale experiment workflow. The
console lets you:

- create or load an experiment with an `experiment_type`
- capture/update a scene observation from Stretch, or upload a saved image
- run repeated prompts on the same scene image
- record `prompt_type` as `clear`, `ambiguous`, or `partial`
- choose the trial method: `proposed_efe`, `top_score`, `random_candidate`,
  `vlm_direct`, `first_question`, `random_question`, or `vlm_best_question`
- answer interactive yes/no questions without executing a grasp
- mark the outcome as `correct`, `wrong`, `unresolved`, `target_pruned`, or
  `aborted`
- watch aggregate metrics grouped by method and prompt type

Offline records are written under `ASK2ACT_OFFLINE_EXPERIMENT_ROOT` as
experiment-scoped `experiment.json`, `scenes/*/scene.json`, `trials/*.json`,
`trials.jsonl`, and `events.jsonl` files. This keeps offline evaluation data
separate from live grasp `session_records`.

Useful config:

```bash
export ASK2ACT_OFFLINE_EXPERIMENT_ROOT=/abs/path/to/offline_experiments
export ASK2ACT_OFFLINE_MAX_ROUNDS=6
```

## Online Robot Experiment Console

Open `GET /online` for the real-robot online evaluation workflow. This console
is separate from `/offline`: scene records are only metadata/reference images,
while every trial starts with a fresh live Stretch observation.

The intended main experiment is:

- 20 physical tabletop scenes
- 3 prompts per scene: `clear`, `ambiguous`, and `partial`
- 4 methods: `top_score`, `random_candidate`, `vlm_best_question`, and
  `proposed_efe`
- 240 total trials

Online trial flow:

1. Register the physical scene and scene type. A reference image is optional.
2. Optional: press `Start Video` once at the beginning of the run. When video
   is active, the robot-side server reuses the same high-resolution D435i stream
   for both trial observations and the MP4 recording, so it does not start a
   competing camera pipeline.
3. Enter the prompt, prompt type, method, and expected candidate display ID.
4. Press `Start Live Trial`. The A6000 requests a new Stretch RGB-D observation
   for that trial, so old offline scene images are not reused.
5. Answer clarification questions when the selected method asks them.
6. Press `Plan Dry Run` if you want a non-motion handoff check.
7. Press `Execute If Correct`. The service first compares the resolved target
   with the expected candidate. If the target is wrong or unresolved, it records
   the failure and skips the physical grasp.
8. After an executed grasp, record whether the physical grasp succeeded, whether
   the correct object was grasped, or whether the wrong object was grasped.
9. Press `Stop & Transfer` after the run; the MP4 is saved under the online
   experiment on A6000 and the robot-side temporary file is deleted.

The online metrics include target selection accuracy, physical grasp success,
correct-object grasp success, wrong-object grasp rate, task success rate, mean
questions for asked trials, and total time.

Useful online config:

```bash
export ASK2ACT_AUTO_EXECUTE_ON_RESOLVE=0
export ASK2ACT_ONLINE_EXPERIMENT_ROOT=/abs/path/to/online_experiments
export ASK2ACT_ONLINE_MAX_ROUNDS=6
```

For utensil-like thin objects, the geometric grasp planner has a conservative
PCA-based slender-object mode. The defaults are intentionally modest:

```bash
export ASK2ACT_GEOMETRIC_SLENDER_OBJECT_ASPECT_RATIO=2.5
export ASK2ACT_GEOMETRIC_SLENDER_OBJECT_MAX_HEIGHT_M=0.065
export ASK2ACT_GEOMETRIC_SLENDER_GRIP_CLEARANCE_M=0.018
export ASK2ACT_GEOMETRIC_SLENDER_MIN_OPEN_WIDTH_M=0.032
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
  "rgb_shape_hw": [720, 1280],
  "depth_shape_hw": [720, 1280],
  "depth_aligned_to_color": true,
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
