# Ask2Act Offline/Online Experimental Protocol and System Implementation

Generated: 2026-06-08T21:40:45

This report explains how the offline and online experiments were implemented technically, how the comparison methods differ, and how the robot system turns a natural-language command into a grasp attempt.

> Note: the full prompt shown here is the project prompt sent to the VLM by `services/a6000_web`. It is not the hidden system prompt of this Codex session.

## 1. Code and Data Map

| Component | Path | Role |
| --- | --- | --- |
| Web service/API | `services/a6000_web/server.py` | FastAPI endpoints for sessions, offline trials, online trials, audit UI, and robot execution. |
| Detection | `services/a6000_web/detection.py` | GroundingDINO prompt construction, detection, NMS, overlay rendering, recall fallback. |
| VLM clarification | `services/a6000_web/clarification.py` | VLM message construction, JSON extraction/repair, candidate state, question ranking, direct selection. |
| Schemas | `services/a6000_web/schemas.py` | Trial/method/prompt/online execution request schemas and candidate/result objects. |
| Offline/online storage | `services/a6000_web/offline_experiments.py` | Experiment, scene, trial JSON storage and metrics. |
| Real grasp runtime | `services/a6000_web/grasp_runtime.py` | RGB-D loading, SAM mask, point cloud, geometric grasp, dispatch payload. |
| Motion planner | `simulation/ask2act_grasp/planning/motion_planner.py` | Stretch top-down IK, wrist yaw, compensation, waypoint plan. |
| Robot dispatch | `real/stretch_transport/scripts/dispatch_grasp.py` | Runs Stretch waypoints on the robot server side. |
| Robot platform/config | `services/a6000_web/a6000_real.env`, `real/stretch_transport/robot_server.env` | A6000-side and robot-side real-runtime parameters. |
| Hardware model | `simulation/ask2act_grasp/stretch3_specs.py` | Stretch joint limits, velocities, forces, and gripper geometry used by the planner. |
| Older CP gate | `/home/haoandong/workspace/project/cp_gate` | Historical conformal/calibrated threshold scripts for DINO score filtering. |

## 2. Offline Experiment Protocol

The offline experiment used saved tabletop observations rather than live robot execution. Each trial followed this path:

1. Load a saved scene image from `services/a6000_web/artifacts/offline_experiments/top_cups_01/scenes/<scene_id>/observation.jpg`.
2. Send the prompt to the same detection and VLM resolution pipeline used by the web service.
3. Record target-resolution outcome as `correct`, `wrong`, or `unresolved`.
4. Audit and clean the trial-level data, then balance the final main analysis to 1512 records: 216 records for each of 7 methods.

Offline methods:

| Method | Interaction | Implementation |
| --- | --- | --- |
| `top_score` | no | Selects the highest GroundingDINO score candidate. |
| `random_candidate` | no | Selects one candidate uniformly using a seeded trial id. |
| `vlm_direct` | no | Asks the VLM to directly output a final `decision` JSON without questions. |
| `first_question` | yes | Uses the first legal VLM question as the clarification question. |
| `random_question` | yes | Randomly chooses one legal VLM question from the generated four. |
| `vlm_best_question` | yes | Uses the VLM's self-ranked best question, id 1. |
| `proposed_efe` | yes | Backend ranks the four legal questions by balanced split plus diversity penalties. |

The common method dispatcher is implemented in `server.py`:

```python
1107: 
1108: def _apply_offline_question_method(session: SessionState, method: str, *, trial_id: str) -> None:
1109:     if session.resolved_target is not None or session.status != "awaiting_answer":
1110:         return
1111:     protocol = session.last_protocol_json or {}
1112:     if method == "proposed_efe":
1113:         session.current_questions = clarifier.rank_questions_for_mode(protocol, session, mode=method)
1114:         if session.current_questions:
1115:             session.current_question = session.current_questions[0]
1116:         return
1117: 
1118:     questions = clarifier.rank_questions_for_mode(protocol, session, mode=method)
1119:     if not questions:
1120:         return
1121:     session.current_questions = questions
1122:     if method in {"first_question", "vlm_best_question"}:
1123:         session.current_question = questions[0]
1124:     elif method == "random_question":
1125:         seed = f"{trial_id}:{session.current_round}:{len(session.question_history)}"
1126:         viable = [
1127:             question
1128:             for question in questions
1129:             if int(question.count.get("total") or 0) <= 1
1130:             or (int(question.count.get("y") or 0) > 0 and int(question.count.get("n") or 0) > 0)
1131:         ]
1132:         session.current_question = random.Random(seed).choice(viable or questions)
1133: 
1134: 
1135: def _apply_offline_trial_method(
1136:     session: SessionState,
1137:     method: str,
1138:     *,
1139:     trial_id: str,
1140:     object_recall_image_data_url: str | None = None,
1141: ) -> None:
1142:     if session.status == "failed_no_candidates":
1143:         return
1144:     if session.resolved_target is not None:
1145:         return
1146:     if method == "top_score":
1147:         if not session.candidates:
1148:             return
1149:         session.resolved_target = _candidate_to_resolved(max(session.candidates, key=lambda item: item.score))
1150:         session.plausible_candidate_ids = [session.resolved_target.candidate_id]
1151:         session.eliminated_candidate_ids = [
1152:             candidate.candidate_id for candidate in session.candidates if candidate.candidate_id != session.resolved_target.candidate_id
1153:         ]
1154:         session.last_removed_candidate_ids = list(session.eliminated_candidate_ids)
1155:         session.candidate_state_history.append(
1156:             {
1157:                 "source": "offline_top_score",
1158:                 "round": session.current_round,
1159:                 "head": "decision",
1160:                 "has_protocol_state": False,
1161:                 "plausible": list(session.plausible_candidate_ids),
1162:                 "eliminated": list(session.eliminated_candidate_ids),
1163:                 "last_removed": list(session.last_removed_candidate_ids),
1164:             }
1165:         )
1166:         session.status = "resolved"
1167:         session.current_question = None
1168:         session.current_questions = []
1169:         session.last_protocol_json = {
1170:             "Head": "decision",
1171:             "Task_ID": 1,
1172:             "Grasp": "yes",
1173:             "Target": {"name": session.resolved_target.candidate_id},
1174:             "Reason": "Offline top-score baseline selected the highest GroundingDINO score.",
1175:         }
1176:     elif method == "random_candidate":
1177:         if not session.candidates:
1178:             return
1179:         candidate = random.Random(trial_id).choice(session.candidates)
1180:         session.resolved_target = _candidate_to_resolved(candidate)
1181:         session.plausible_candidate_ids = [candidate.candidate_id]
1182:         session.eliminated_candidate_ids = [
1183:             item.candidate_id for item in session.candidates if item.candidate_id != candidate.candidate_id
1184:         ]
1185:         session.last_removed_candidate_ids = list(session.eliminated_candidate_ids)
1186:         session.candidate_state_history.append(
1187:             {
1188:                 "source": "offline_random_candidate",
1189:                 "round": session.current_round,
1190:                 "head": "decision",
1191:                 "has_protocol_state": False,
1192:                 "plausible": list(session.plausible_candidate_ids),
1193:                 "eliminated": list(session.eliminated_candidate_ids),
1194:                 "last_removed": list(session.last_removed_candidate_ids),
1195:             }
1196:         )
1197:         session.status = "resolved"
1198:         session.current_question = None
1199:         session.current_questions = []
1200:         session.last_protocol_json = {
1201:             "Head": "decision",
1202:             "Task_ID": 1,
1203:             "Grasp": "yes",
1204:             "Target": {"name": candidate.candidate_id},
1205:             "Reason": "Offline random-candidate baseline selected uniformly from the CP-gated candidates.",
1206:         }
1207:     elif method == "vlm_direct":
1208:         clarifier.direct_select(session)
1209:     else:
1210:         if session.status == "detected" and session.last_protocol_json is None:
1211:             clarifier.initialize_session(session)
1212:         if session.status == "needs_object_recall":
1213:             if object_recall_image_data_url and _run_partial_object_recall(session, object_recall_image_data_url):
1214:                 clarifier.initialize_session(session)
1215:                 if session.status == "needs_object_recall":
1216:                     session.status = "failed_object_recall"
1217:                     session.error_message = "VLM requested object recall again after the broader GroundingDINO prompt."
1218:                     session.current_question = None
1219:                     session.current_questions = []
1220:                     return
1221:             else:
1222:                 session.status = "failed_object_recall"
1223:                 session.error_message = "VLM requested object recall, but the broader GroundingDINO prompt did not add candidates."
```

Offline trial startup is implemented as:

```python
3481: def start_offline_trial(experiment_id: str, request: OfflineTrialStartRequest):
3482:     try:
3483:         scene = offline_store.scene_view(experiment_id, request.scene_id)
3484:         method = request.method
3485:         single_candidate_audit = request.prompt_type == "partial" and method in INTERACTIVE_CLARIFICATION_METHODS
3486:         trial_id = f"trial_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
3487:         session = create_session(
3488:             StartSessionRequest(
3489:                 instruction=request.prompt,
3490:                 fetch_observation=False,
3491:                 observation_image_data_url=scene["observation_image_data_url"],
3492:                 observation_id=scene.get("observation_id") or f"{experiment_id}:{request.scene_id}",
3493:             ),
3494:             initialize_clarification=False,
3495:             auto_execute=False,
3496:             allow_single_candidate_auto_resolve=not single_candidate_audit,
3497:             enable_attribute_object_recall=single_candidate_audit,
3498:         )
3499:         session.prompt_type = request.prompt_type
3500:         session.question_mode = method
3501:         session.single_candidate_audit = single_candidate_audit and len(session.candidates) == 1 and session.resolved_target is None
3502:         clarifier.ensure_candidate_state(session)
3503:         SESSIONS[session.session_id] = session
3504:         OFFLINE_TRIAL_SESSIONS[trial_id] = session.session_id
3505:         trial = {
3506:             "trial_id": trial_id,
3507:             "experiment_id": experiment_id,
3508:             "experiment_type": offline_store.read_experiment(experiment_id).get("experiment_type"),
3509:             "scene_id": request.scene_id,
3510:             "scene_type": scene.get("scene_type"),
3511:             "object_categories": scene.get("object_categories") or [],
3512:             "prompt": request.prompt,
3513:             "prompt_type": request.prompt_type,
3514:             "method": method,
3515:             "expected_candidate_id": request.expected_candidate_id,
3516:             "expected_display_id": request.expected_display_id,
3517:             "notes": request.notes or "",
3518:             "started_at_epoch_s": time.time(),
3519:             "scene_observation_path": scene.get("observation_path"),
3520:         }
3521:         _update_offline_trial_from_session(experiment_id, trial, session)
3522:         offline_store.append_event(experiment_id, {"event": "trial_started", "trial_id": trial_id, "method": method})
3523:         try:
3524:             _apply_offline_trial_method(
3525:                 session,
3526:                 method,
3527:                 trial_id=trial_id,
3528:                 object_recall_image_data_url=scene["observation_image_data_url"],
3529:             )
3530:             if session.status == "awaiting_answer" and len(session.question_history) >= OFFLINE_MAX_ROUNDS:
3531:                 session.status = "offline_max_rounds"
3532:                 session.current_question = None
3533:             _update_offline_trial_from_session(experiment_id, trial, session)
3534:         except Exception as exc:
3535:             _mark_offline_vlm_failure(experiment_id, trial, session, exc, event="trial_vlm_start_failed")
3536:         return _offline_trial_view(experiment_id, trial_id)
3537:     except Exception as exc:
3538:         raise HTTPException(status_code=500, detail=str(exc)) from exc
3539: 
3540: 
3541: @app.get("/api/offline/experiments/{experiment_id}/trials/{trial_id}")
3542: def get_offline_trial(experiment_id: str, trial_id: str):
3543:     try:
3544:         return _offline_trial_view(experiment_id, trial_id)
3545:     except Exception as exc:
3546:         raise HTTPException(status_code=404, detail=str(exc)) from exc
3547: 
3548: 
3549: @app.post("/api/offline/experiments/{experiment_id}/trials/{trial_id}/step")
3550: def step_offline_trial(experiment_id: str, trial_id: str, request: OfflineTrialStepRequest):
```

## 3. Online Experiment Protocol

The online main experiment used live Stretch observations and physical grasp execution. The cleaned main dataset is `main_02`, balanced to 228 records: 57 trials per method and 19 trials per method × prompt type.

Online methods:

| Method | Uses questions | Robot execution gate |
| --- | ---: | --- |
| `top_score` | no | Execute only if operator confirms the selected target is correct. |
| `random_candidate` | no | Execute only if operator confirms the selected target is correct. |
| `vlm_best_question` | yes | Execute only after resolved target is confirmed correct. |
| `proposed_efe` | yes | Execute only after resolved target is confirmed correct. |

The online gate is intentionally conservative: if the resolved target is wrong, no physical grasp is executed. This saves robot wear and makes wrong-object grasp rate zero in the cleaned main experiment.

Online trial startup and execution gate:

```python
2953: @app.post("/api/online/experiments/{experiment_id}/trials/start")
2954: def start_online_trial(experiment_id: str, request: OnlineTrialStartRequest):
2955:     try:
2956:         if request.method not in ONLINE_METHODS:
2957:             raise HTTPException(status_code=400, detail=f"unsupported online method: {request.method}")
2958:         scene = _online_scene_view(experiment_id, request.scene_id, include_image=False, public=False)
2959:         method = request.method
2960:         single_candidate_audit = request.prompt_type == "partial" and method in INTERACTIVE_CLARIFICATION_METHODS
2961:         trial_id = f"online_trial_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
2962:         session = create_session(
2963:             StartSessionRequest(
2964:                 instruction=request.prompt,
2965:                 fetch_observation=True,
2966:                 observation_image_data_url=None,
2967:                 observation_id=None,
2968:             ),
2969:             initialize_clarification=False,
2970:             auto_execute=False,
2971:             allow_single_candidate_auto_resolve=not single_candidate_audit,
2972:             enable_attribute_object_recall=single_candidate_audit,
2973:         )
2974:         session.prompt_type = request.prompt_type
2975:         session.question_mode = method
2976:         session.single_candidate_audit = single_candidate_audit and len(session.candidates) == 1 and session.resolved_target is None
2977:         clarifier.ensure_candidate_state(session)
2978:         SESSIONS[session.session_id] = session
2979:         ONLINE_TRIAL_SESSIONS[trial_id] = session.session_id
2980:         trial = {
2981:             "trial_id": trial_id,
2982:             "experiment_id": experiment_id,
2983:             "experiment_type": online_store.read_experiment(experiment_id).get("experiment_type"),
2984:             "scene_id": request.scene_id,
2985:             "scene_type": scene.get("scene_type"),
2986:             "object_categories": scene.get("object_categories") or [],
2987:             "prompt": request.prompt,
2988:             "prompt_type": request.prompt_type,
2989:             "method": method,
2990:             "expected_candidate_id": None,
2991:             "expected_display_id": None,
2992:             "notes": request.notes or "",
2993:             "started_at_epoch_s": time.time(),
2994:             "live_observation": True,
2995:             "scene_reference_observation_path": scene.get("observation_path"),
2996:             "grasp_attempted": False,
2997:             "wrong_target_grasp_prevented": False,
2998:         }
2999:         _update_online_trial_from_session(experiment_id, trial, session)
3000:         online_store.append_event(experiment_id, {"event": "online_trial_started", "trial_id": trial_id, "method": method})
3001:         try:
3002:             _apply_offline_trial_method(
3003:                 session,
3004:                 method,
3005:                 trial_id=trial_id,
3006:                 object_recall_image_data_url=session.observation_image_data_url,
3007:             )
3008:             if session.status == "awaiting_answer" and len(session.question_history) >= ONLINE_MAX_ROUNDS:
3009:                 session.status = "online_max_rounds"
3010:                 session.current_question = None
3011:             _update_online_trial_from_session(experiment_id, trial, session)
3012:         except Exception as exc:
3013:             _mark_online_vlm_failure(experiment_id, trial, session, exc, event="online_trial_vlm_start_failed")
3014:         return _online_trial_view(experiment_id, trial_id)
3015:     except HTTPException:
3016:         raise
3017:     except Exception as exc:
3018:         raise HTTPException(status_code=500, detail=str(exc)) from exc
3019: 
3020: 
3021: @app.get("/api/online/experiments/{experiment_id}/trials/{trial_id}")
3022: def get_online_trial(experiment_id: str, trial_id: str):
3023:     try:
3024:         return _online_trial_view(experiment_id, trial_id)
3025:     except Exception as exc:
3026:         raise HTTPException(status_code=404, detail=str(exc)) from exc
3027: 
3028: 
3029: @app.post("/api/online/experiments/{experiment_id}/trials/{trial_id}/step")
3030: def step_online_trial(experiment_id: str, trial_id: str, request: OnlineTrialStepRequest):
3031:     try:
3032:         with _online_trial_lock(trial_id):
3033:             trial = online_store.read_trial(experiment_id, trial_id)
3034:             session_id = trial.get("session_id") or ONLINE_TRIAL_SESSIONS.get(trial_id)
3035:             session = SESSIONS.get(str(session_id))
3036:             if session is None:
3037:                 raise HTTPException(status_code=409, detail="active session is not in memory; start a new online trial or abort this one")
3038:             client_q_count = request.question_count
3039:             if client_q_count is not None and len(session.question_history) > int(client_q_count):
3040:                 return _online_trial_view(experiment_id, trial_id)
3041:             if session.status != "awaiting_answer" or session.current_question is None:
3042:                 if client_q_count is not None and len(session.question_history) >= int(client_q_count):
3043:                     return _online_trial_view(experiment_id, trial_id)
3044:                 raise HTTPException(status_code=409, detail="trial is not waiting for a clarification answer")
3045:             try:
3046:                 clarifier.answer_current_question(session, request.answer)
3047:             except Exception as exc:
3048:                 _mark_online_vlm_failure(experiment_id, trial, session, exc, event="online_trial_vlm_step_failed")
3049:                 return _online_trial_view(experiment_id, trial_id)
3050:             if session.resolved_target is not None:
3051:                 finalize_resolved_target(session)
3052:             elif len(session.question_history) >= ONLINE_MAX_ROUNDS:
3053:                 session.status = "online_max_rounds"
3054:                 session.current_question = None
3055:                 session.current_questions = []
3056:             else:
3057:                 _apply_offline_question_method(session, str(trial.get("method") or "proposed_efe"), trial_id=trial_id)
3058:             _update_online_trial_from_session(experiment_id, trial, session)
3059:             online_store.append_event(
3060:                 experiment_id,
3061:                 {"event": "online_trial_answered", "trial_id": trial_id, "answer": request.answer, "request_id": request.request_id},
3062:             )
3063:             return _online_trial_view(experiment_id, trial_id)
3064:     except HTTPException:
3065:         raise
3066:     except Exception as exc:
3067:         raise HTTPException(status_code=500, detail=str(exc)) from exc
3068: 
3069: 
3070: @app.post("/api/online/experiments/{experiment_id}/trials/{trial_id}/execute")
3071: def execute_online_trial(experiment_id: str, trial_id: str, request: OnlineTrialExecuteRequest):
3072:     try:
3073:         trial = online_store.read_trial(experiment_id, trial_id)
3074:         session_id = trial.get("session_id") or ONLINE_TRIAL_SESSIONS.get(trial_id)
3075:         session = SESSIONS.get(str(session_id)) if session_id else None
3076:         if session is None:
3077:             raise HTTPException(status_code=409, detail="active session is not in memory; cannot execute safely")
3078:         if request.expected_candidate_id:
3079:             trial["expected_candidate_id"] = request.expected_candidate_id
3080:         if request.expected_display_id is not None:
3081:             trial["expected_display_id"] = request.expected_display_id
3082:         if not trial.get("expected_candidate_id") and trial.get("expected_display_id") is None:
3083:             raise HTTPException(status_code=400, detail="expected candidate/display id is required before any online execution gate")
3084:         _update_online_trial_from_session(experiment_id, trial, session)
3085:         trial = online_store.read_trial(experiment_id, trial_id)
3086:         target_outcome = trial.get("target_selection_outcome")
3087:         if target_outcome is None:
3088:             raise HTTPException(
3089:                 status_code=400,
3090:                 detail="expected target is required before execution; set expected_display_id or expected_candidate_id",
3091:             )
3092:         if session.resolved_target is None or target_outcome == "unresolved":
3093:             trial.update(
3094:                 {
3095:                     "status": "finished",
3096:                     "outcome": "unresolved",
3097:                     "grasp_attempted": False,
3098:                     "wrong_target_grasp_prevented": False,
3099:                     "operator_note": request.note or "No resolved target; grasp skipped.",
3100:                     "finished_at_epoch_s": time.time(),
3101:                 }
3102:             )
3103:             trial["total_time_s"] = trial["finished_at_epoch_s"] - float(trial.get("started_at_epoch_s") or trial["finished_at_epoch_s"])
3104:             online_store.write_trial(experiment_id, trial)
3105:             return _online_trial_view(experiment_id, trial_id)
3106:         if target_outcome != "correct":
3107:             trial.update(
3108:                 {
3109:                     "status": "finished",
3110:                     "outcome": "skipped_wrong_target",
3111:                     "grasp_attempted": False,
3112:                     "wrong_target_grasp_prevented": True,
3113:                     "physical_grasp_success": False,
3114:                     "correct_object_grasp_success": False,
3115:                     "wrong_object_grasp": False,
3116:                     "operator_note": request.note or "Resolved target did not match expected candidate; grasp intentionally skipped.",
3117:                     "finished_at_epoch_s": time.time(),
3118:                 }
3119:             )
3120:             trial["total_time_s"] = trial["finished_at_epoch_s"] - float(trial.get("started_at_epoch_s") or trial["finished_at_epoch_s"])
3121:             online_store.write_trial(experiment_id, trial)
3122:             online_store.append_event(
3123:                 experiment_id,
3124:                 {"event": "online_wrong_target_grasp_skipped", "trial_id": trial_id, "resolved_candidate_id": trial.get("resolved_candidate_id")},
3125:             )
3126:             return _online_trial_view(experiment_id, trial_id)
3127: 
3128:         execute_resolved_session(session, dry_run=request.dry_run, raise_on_error=True)
3129:         execution_ok = bool(session.execution_result and session.execution_result.get("ok"))
3130:         if not execution_ok:
3131:             finished_at = time.time()
3132:             trial.update(
3133:                 {
3134:                     "status": "finished",
3135:                     "outcome": "execution_failed",
3136:                     "grasp_attempted": not bool(request.dry_run),
```

Online confirmation converts physical feedback into task outcome:

```python
3198:             raise HTTPException(status_code=409, detail="this online trial did not execute a physical grasp")
3199:         correct_object_success = (
3200:             bool(request.correct_object_grasp_success)
3201:             if request.correct_object_grasp_success is not None
3202:             else (bool(request.physical_grasp_success) and not bool(request.wrong_object_grasp))
3203:         )
3204:         outcome = "correct" if correct_object_success else ("wrong" if request.wrong_object_grasp else "grasp_failed")
3205:         trial.update(
3206:             {
3207:                 "status": "finished",
3208:                 "outcome": outcome,
3209:                 "physical_grasp_success": bool(request.physical_grasp_success),
3210:                 "correct_object_grasp_success": correct_object_success,
3211:                 "wrong_object_grasp": bool(request.wrong_object_grasp),
3212:                 "confirmation_note": request.note or "",
3213:                 "reset_ready": bool(request.reset_ready),
3214:                 "finished_at_epoch_s": time.time(),
3215:             }
3216:         )
3217:         trial["total_time_s"] = trial["finished_at_epoch_s"] - float(trial.get("started_at_epoch_s") or trial["finished_at_epoch_s"])
3218:         if session is not None:
3219:             session.confirmation_result = {
3220:                 "success": correct_object_success,
3221:                 "physical_grasp_success": bool(request.physical_grasp_success),
3222:                 "wrong_object_grasp": bool(request.wrong_object_grasp),
3223:                 "note": request.note or "",
3224:                 "confirmed_at_epoch_s": trial["finished_at_epoch_s"],
3225:                 "reset_ready": bool(request.reset_ready),
3226:             }
3227:             session.status = "confirmed_success" if correct_object_success else "confirmed_failure"
3228:             finalize_resolved_target(
```

## 4. CP / Calibrated DINO Thresholding From the Earlier Project

The older `~/workspace/project/cp_gate` code implements a calibrated score threshold for GroundingDINO candidates. It is best understood as a score-filtering/calibration module around DINO rather than the VLM clarification method itself.

Data collection and inference used a broad DINO prompt:

```text
cup . bottle . spoon . fork . knife . plate .
```

The CP/calibration scripts use low baseline DINO thresholds to collect candidate scores, annotate false positives, then choose a final score threshold `tau_final`.

Formula used by `compute_tau_balanced.py`:

```text
S_FP = scores of detections labeled as false positives
S_TP = scores of all other detections
tau_fp = Quantile_{1 - eps_fp}(S_FP)
tau_tp = Quantile_{eps_tp}(S_TP)
tau_final = max(tau_fp, tau_tp)
keep detection iff score >= tau_final
```

Core implementation:

```python
0012:         return None
0013:     a.sort()
0014:     k = int(np.ceil(q * a.size)) - 1
0015:     k = max(0, min(k, a.size - 1))
0016:     return float(a[k])
0017: 
0018: 
0019: def main():
0020:     ap = argparse.ArgumentParser()
0021:     ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
0022:     ap.add_argument("--csv", default="work/annotations/detections_labeled.csv",
0023:                     help="标注后的 CSV（含 is_fp）")
0024:     ap.add_argument("--eps_fp", type=float, default=0.02,
0025:                     help="FP 端误差率 ε_FP（默认 0.02 → 98%）")
0026:     ap.add_argument("--eps_tp", type=float, default=0.02,
0027:                     help="TP 端误差率 ε_TP（默认 0.02 → 98%）")
0028:     ap.add_argument("--out", default="work/tau/tau_balanced.json")
0029:     args = ap.parse_args()
0030: 
0031:     eps_fp = args.eps_fp
0032:     eps_tp = args.eps_tp
0033: 
0034:     csv_path = os.path.join(args.data_root, args.csv)
0035:     fp_scores = []
0036:     tp_scores = []
0037: 
0038:     if not os.path.exists(csv_path):
0039:         raise FileNotFoundError(f"CSV 找不到: {csv_path}")
0040: 
0041:     with open(csv_path, newline="", encoding="utf-8") as f:
0042:         rd = csv.DictReader(f)
0043: 
0044:         for r in rd:
0045:             try:
0046:                 s = float(r["score"])
0047:             except:
0048:                 continue
0049: 
0050:             # 🌟 规则：
0051:             #  - is_fp == 1 → FP
0052:             #  - 其它全部视为 TP（包括 is_tp 空）
0053:             if r.get("is_fp", "").strip() == "1":
0054:                 fp_scores.append(s)
0055:             else:
0056:                 tp_scores.append(s)
0057: 
0058:     if not fp_scores:
0059:         raise ValueError("❌ 没有 FP 样本（is_fp=1）。至少要标些 FP 才能算阈值。")
0060: 
0061:     # τ_FP: FP 的分位（1 - eps_fp）
0062:     tau_fp = finite_quantile(fp_scores, 1 - eps_fp)
0063: 
0064:     # τ_TP: TP 的分位（eps_tp）
0065:     tau_tp = finite_quantile(tp_scores, eps_tp) if tp_scores else None
0066: 
0067:     # 最终阈值：取更严格的那个
0068:     tau_final = max(tau_fp, tau_tp) if tau_tp is not None else tau_fp
0069: 
0070:     out_path = os.path.join(args.data_root, args.out)
0071:     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
0072: 
0073:     json.dump(
0074:         {
0075:             "tau_final": tau_final,
0076:             "tau_fp": tau_fp,
0077:             "tau_tp": tau_tp,
0078:             "eps_fp": eps_fp,
```

Inference with `tau_final`:

```python
0108:     ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
0109:                     help="一般是 project/calib_data")
0110:     ap.add_argument("--img_dir", default=None,
0111:                     help="要推理的图片根目录，默认= data_root")
0112:     ap.add_argument("--prompt", default="cup . bottle . spoon . fork . knife . plate .")
0113:     ap.add_argument("--tau", default="work/tau/tau_balanced.json",
0114:                     help="compute_tau_balanced 输出的 JSON，相对 data_root")
0115:     ap.add_argument("--no_exif_fix", action="store_true",
0116:                     help="关闭 EXIF 方向修正")
0117:     args = ap.parse_args()
0118: 
0119:     # 1) 读取 τ
0120:     tau_path = os.path.join(args.data_root, args.tau)
0121:     with open(tau_path, "r", encoding="utf-8") as f:
0122:         tau = float(json.load(f)["tau_final"])
0123:     print(f"✅ 使用 τ_final = {tau:.4f} 进行过滤")
0124: 
0125:     # 2) 输入/输出路径
0126:     img_root = args.img_dir or args.data_root
0127:     out_dir = os.path.join(args.data_root, "work/infer_vis")
0128: 
0129:     # 3) 初始化 DINO
0130:     device = "cuda" if torch.cuda.is_available() else "cpu"
0131:     print("device:", device)
0132:     proc = GroundingDinoProcessor.from_pretrained(MODEL_ID)
0133:     model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
0134: 
0135:     # 4) 收集图片
0136:     imgs = []
0137:     for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
0138:         imgs += glob.glob(os.path.join(img_root, "**", e), recursive=True)
0139:     imgs.sort()
0140:     if not imgs:
0141:         print("⚠️ 没有找到任何图片，请检查 --img_dir 或 data_root。")
0142:         return
0143: 
0144:     # 5) 逐张推理 + 过滤 + 画图
0145:     for p in imgs:
0146:         try:
0147:             img = Image.open(p).convert("RGB")
0148:         except Exception as e:
0149:             print("[open failed]", p, e)
0150:             continue
0151: 
0152:         if not args.no_exif_fix:
0153:             img = ImageOps.exif_transpose(img)
0154: 
0155:         w, h = img.size
0156:         inp = proc(images=img, text=args.prompt, return_tensors="pt").to(device)
0157:         with torch.no_grad():
0158:             out = model(**inp)
0159: 
0160:         r = postproc(proc, out, inp["input_ids"], h, w)
0161: 
0162:         labels_raw = r.get("text_labels") or r.get("labels") or []
0163:         boxes = r.get("boxes", [])
0164:         scores = r.get("scores", [])
0165: 
0166:         if len(boxes) == 0:
0167:             continue
0168: 
0169:         boxes_np = to_numpy_safe(boxes)
```

Evaluation computes object recall, precision, F1, and false positives per image:

```python
0083:     ap.add_argument("--tau_grid", default=TAU_GRID_JSON)
0084:     ap.add_argument("--out_prefix", default=OUT_PREFIX)
0085:     args = ap.parse_args()
0086: 
0087:     data_root = args.data_root
0088:     test_csv_path = os.path.join(data_root, args.test_csv)
0089: 
0090:     per_image = load_per_image_stats(test_csv_path)
0091:     if not per_image:
0092:         print("❌ 没有任何带 gt_total 的图像，无法评估。")
0093:         return
0094: 
0095:     images = sorted(per_image.keys())
0096:     global_N = sum(per_image[img]["N"] for img in images)
0097:     print(f"✅ 有效 test 图像数: {len(images)}, 真实物体总数 N = {global_N}")
0098: 
0099:     tau_grid_path = args.tau_grid
0100:     if not os.path.isabs(tau_grid_path):
0101:         tau_grid_path = os.path.join(data_root, tau_grid_path)
0102:     tau_records = load_tau_grid(tau_grid_path)
0103:     print(f"📐 tau_grid 里共有 {len(tau_records)} 条 (eps_fp, eps_tp) 记录。")
0104: 
0105:     results = []
0106: 
0107:     for rec in tau_records:
0108:         eps_fp = float(rec.get("eps_fp", 0.0))
0109:         eps_tp = float(rec.get("eps_tp", 0.0))
0110:         tau_final = float(rec["tau_final"])
0111: 
0112:         TP_total = 0
0113:         FP_total = 0
0114: 
0115:         for img in images:
0116:             info = per_image[img]
0117:             tp_scores = info["tp_scores"]
0118:             fp_scores = info["fp_scores"]
0119: 
0120:             TP_keep = int(np.sum(tp_scores >= tau_final))
0121:             FP_keep = int(np.sum(fp_scores >= tau_final))
0122: 
0123:             TP_total += TP_keep
0124:             FP_total += FP_keep
0125: 
0126:         recall_obj = TP_total / global_N if global_N > 0 else 0.0
0127:         denom = TP_total + FP_total
0128:         if denom > 0:
0129:             precision = TP_total / denom
0130:         else:
0131:             precision = 1.0
```

In the current A6000 runtime, GroundingDINO thresholds are controlled through environment variables such as `ASK2ACT_DINO_BOX_THRESHOLD` and `ASK2ACT_DINO_TEXT_THRESHOLD`; the old CP scripts document the calibration method and can be reused to set those thresholds or reintroduce a separate calibrated post-filter.

## 5. GroundingDINO Candidate Generation

Detection begins by extracting noun phrases from the instruction. The resulting terms are joined with periods because GroundingDINO expects category-like phrases separated by `.`.

Core prompt construction:

```python
0082:     "rubber",
0083:     "steel",
0084:     "wood",
0085:     "wooden",
0086: }
0087: SIZE_ATTRIBUTE_WORDS = {"big", "large", "little", "medium", "small", "smaller", "larger"}
0088: 
0089: 
0090: def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
0091:     b64 = base64.b64encode(image_bytes).decode("ascii")
0092:     return f"data:{mime_type};base64,{b64}"
0093: 
0094: 
0095: def pil_to_png_bytes(image: Image.Image) -> bytes:
0096:     buffer = io.BytesIO()
0097:     image.save(buffer, format="PNG")
0098:     return buffer.getvalue()
0099: 
0100: 
0101: def to_numpy_safe(value):
0102:     if isinstance(value, torch.Tensor):
0103:         return value.detach().cpu().numpy()
0104:     return np.asarray(value)
0105: 
0106: 
0107: def iou_xyxy(a, b):
0108:     x1 = max(a[0], b[0])
0109:     y1 = max(a[1], b[1])
0110:     x2 = min(a[2], b[2])
0111:     y2 = min(a[3], b[3])
0112:     iw = max(0.0, x2 - x1)
0113:     ih = max(0.0, y2 - y1)
0114:     inter = iw * ih
0115:     area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
```

Phrase extraction returns either spaCy noun phrases or a fallback keyword phrase:

```python
0240:     @staticmethod
0241:     def _extract_fallback(text: str) -> Tuple[List[str], List[str]]:
0242:         text = _normalize_text(text)
0243:         words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())
0244:         filtered = [word for word in words if word not in FALLBACK_STOPWORDS]
0245:         if not filtered:
0246:             return [], []
0247:         phrase = " ".join(filtered)
0248:         return [phrase], [filtered[-1]]
0249: 
0250:     def extract_np_like_phrases(self, text: str) -> List[str]:
0251:         text = (text or "").strip()
0252:         if not text:
0253:             return []
0254:         if self._load_nlp():
0255:             phrases, _ = self._extract_with_spacy(text)
0256:             return phrases
0257:         phrases, _ = self._extract_fallback(text)
0258:         return phrases
0259: 
0260:     def extract_terms(self, text: str, base_terms: Sequence[str]) -> Tuple[List[str], List[str]]:
0261:         text = (text or "").strip()
0262:         phrases: List[str]
0263:         nouns: List[str]
0264:         if text and self._load_nlp():
0265:             phrases, nouns = self._extract_with_spacy(text)
0266:         else:
0267:             phrases, nouns = self._extract_fallback(text)
0268:         if phrases:
```

Detection fallbacks were added to handle cases such as `yellow cup`, `left fork`, or a partial prompt where the initial DINO result missed matching candidates:

```python
0360:                     max(0.0, min(origin_x, max_x)),
0361:                     max(0.0, min(origin_y, max_y)),
0362:                 )
0363:             )
0364: 
0365:     own_rect = (left, top, right, bottom)
0366:     avoid_rects = list(avoid_bboxes or [])
0367: 
0368:     def score(option: Tuple[int, float, float]) -> Tuple[int, float, float, int]:
0369:         idx, origin_x, origin_y = option
0370:         label_rect = (origin_x, origin_y, origin_x + label_w, origin_y + label_h)
0371:         own_overlap = _rect_overlap_area(label_rect, own_rect)
0372:         avoid_overlap = sum(_rect_overlap_area(label_rect, rect) for rect in avoid_rects)
0373:         return (1 if own_overlap > 0 else 0, avoid_overlap, own_overlap, idx)
0374: 
0375:     _, best_x, best_y = min(candidates, key=score)
0376:     return int(round(best_x)), int(round(best_y))
0377: 
0378: 
0379: @dataclass
0380: class DetectionResult:
0381:     prepared_image_bytes: bytes
0382:     image_data_url: str
0383:     overlay_data_url: str
0384:     candidates: List[Candidate]
0385:     phrases: List[str]
0386:     detection_prompt: str
0387: 
0388: 
0389: class GroundingDinoDetector:
0390:     def __init__(
0391:         self,
0392:         phrase_extractor: InstructionPhraseExtractor,
0393:         model_id: str = "IDEA-Research/grounding-dino-base",
0394:         base_terms: Sequence[str] = DEFAULT_BASE_TERMS,
0395:         box_threshold: float | None = None,
0396:         text_threshold: float | None = None,
0397:         nms_iou: float = 0.50,
0398:         max_per_image: int = 100,
0399:         rotate_clockwise_90: bool | None = None,
0400:         device: str | None = None,
0401:     ) -> None:
0402:         self.phrase_extractor = phrase_extractor
0403:         self.model_id = model_id
0404:         self.base_terms = tuple(base_terms)
0405:         self.box_threshold = float(os.getenv("ASK2ACT_DINO_BOX_THRESHOLD", "0.38")) if box_threshold is None else box_threshold
0406:         self.text_threshold = (
0407:             float(os.getenv("ASK2ACT_DINO_TEXT_THRESHOLD", "0.30")) if text_threshold is None else text_threshold
0408:         )
0409:         self.nms_iou = nms_iou
0410:         self.max_per_image = max_per_image
0411:         self.rotate_clockwise_90 = (
0412:             os.getenv("ASK2ACT_DINO_ROTATE_CLOCKWISE_90", "1").strip().lower() in {"1", "true", "yes", "on"}
0413:             if rotate_clockwise_90 is None
0414:             else rotate_clockwise_90
0415:         )
0416:         self.device = self._resolve_device(device)
0417:         self.processor = None
0418:         self.model = None
0419: 
0420:     @staticmethod
0421:     def _resolve_device(device: str | None) -> str:
0422:         if device:
0423:             return device
0424: 
0425:         env_device = os.getenv("ASK2ACT_DETECTOR_DEVICE", "").strip()
0426:         if env_device:
0427:             return env_device
0428: 
0429:         if torch.cuda.is_available():
0430:             if torch.cuda.device_count() > 1:
0431:                 return "cuda:1"
0432:             return "cuda:0"
0433: 
0434:         return "cpu"
0435: 
0436:     def _ensure_model(self) -> None:
0437:         if self.processor is not None and self.model is not None:
0438:             return
0439:         self.processor = GroundingDinoProcessor.from_pretrained(self.model_id)
0440:         self.model = GroundingDinoForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
0441: 
0442:     def _prepare_image(self, image_bytes: bytes) -> Image.Image:
0443:         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
0444:         image = ImageOps.exif_transpose(image)
0445:         if self.rotate_clockwise_90:
0446:             image = image.transpose(Image.ROTATE_270)
0447:         return image
0448: 
0449:     def _post_process(
0450:         self,
0451:         outputs,
0452:         input_ids,
0453:         height: int,
0454:         width: int,
0455:         *,
0456:         box_threshold: float | None = None,
0457:         text_threshold: float | None = None,
0458:     ):
0459:         box_threshold = self.box_threshold if box_threshold is None else box_threshold
0460:         text_threshold = self.text_threshold if text_threshold is None else text_threshold
0461:         for kwargs in (
0462:             dict(threshold=box_threshold, text_threshold=text_threshold),
0463:             dict(box_threshold=box_threshold, text_threshold=text_threshold),
0464:         ):
0465:             try:
0466:                 return self.processor.post_process_grounded_object_detection(
0467:                     outputs=outputs,
0468:                     input_ids=input_ids,
0469:                     target_sizes=[(height, width)],
0470:                     **kwargs,
0471:                 )[0]
0472:             except TypeError:
```

Every candidate is represented as:

```python
0009: class Candidate(BaseModel):
0010:     candidate_id: str
0011:     display_id: Optional[int] = None
0012:     label: str
0013:     score: float
0014:     bbox_xyxy: List[float]
0015:     mask_rle: Optional[str] = None
0016: 
0017: 
0018: class ScoredQuestion(BaseModel):
```

The candidate overlay uses `display_id` only for humans. The VLM is forbidden from asking about display ids, candidate ids, tags, or numbers.

## 6. VLM JSON Protocol

The VLM sees the annotated candidate overlay image plus one JSON payload. The payload contains:

- `Head`: `start` or `answer`.
- `Task_ID` and `Round`.
- `Target_Instruction`.
- `Prompt_Type`.
- `Question_Mode`.
- `Candidate_State` with plausible/eliminated ids.
- `Candidates`, each containing `display_id`, `visual_tag`, `label`, `bbox`, ranks, and score.
- `Asked` history after each human answer.

Payload construction:

```python
0436:             centers[candidate.candidate_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
0437: 
0438:         left_to_right = {
0439:             candidate.candidate_id: rank
0440:             for rank, candidate in enumerate(
0441:                 sorted(candidates, key=lambda item: (centers[item.candidate_id][0], centers[item.candidate_id][1])),
0442:                 start=1,
0443:             )
0444:         }
0445:         top_to_bottom = {
0446:             candidate.candidate_id: rank
0447:             for rank, candidate in enumerate(
0448:                 sorted(candidates, key=lambda item: (centers[item.candidate_id][1], centers[item.candidate_id][0])),
0449:                 start=1,
0450:             )
0451:         }
0452: 
0453:         payload: Dict[str, Any] = {}
0454:         for candidate in candidates:
0455:             x1, y1, x2, y2 = [float(value) for value in candidate.bbox_xyxy]
0456:             cx, cy = centers[candidate.candidate_id]
0457:             display_id = cls._candidate_display_id(candidate)
0458:             payload[candidate.candidate_id] = {
0459:                 "display_id": display_id,
0460:                 "visual_tag": str(display_id),
0461:                 "label": candidate.label,
0462:                 "bbox": rounded([x1, y1, x2, y2]),
0463:                 "bbox_center": rounded([cx, cy]),
0464:                 "left_to_right_rank": left_to_right[candidate.candidate_id],
0465:                 "top_to_bottom_rank": top_to_bottom[candidate.candidate_id],
0466:                 "score": round(float(candidate.score), 3),
0467:             }
0468:         return payload
0469: 
0470:     @classmethod
0471:     def ensure_candidate_state(cls, session: SessionState) -> None:
0472:         all_ids = [candidate.candidate_id for candidate in session.candidates]
0473:         all_id_set = set(all_ids)
0474:         eliminated = {candidate_id for candidate_id in session.eliminated_candidate_ids if candidate_id in all_id_set}
0475:         if not session.plausible_candidate_ids and not eliminated:
0476:             plausible = set(all_ids)
0477:         else:
0478:             plausible = {
0479:                 candidate_id
0480:                 for candidate_id in session.plausible_candidate_ids
0481:                 if candidate_id in all_id_set and candidate_id not in eliminated
0482:             }
0483:         session.plausible_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in plausible]
0484:         session.eliminated_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in eliminated]
0485:         session.last_removed_candidate_ids = [
0486:             candidate_id for candidate_id in session.last_removed_candidate_ids if candidate_id in all_id_set
0487:         ]
0488: 
0489:     @classmethod
0490:     def _candidate_state_payload(cls, session: SessionState) -> Dict[str, Any]:
0491:         cls.ensure_candidate_state(session)
0492:         return {
0493:             "plausible": list(session.plausible_candidate_ids),
0494:             "eliminated": list(session.eliminated_candidate_ids),
0495:             "last_removed": list(session.last_removed_candidate_ids),
0496:         }
0497: 
0498:     @staticmethod
0499:     def _question_mode_instruction(question_mode: str | None) -> str:
0500:         mode = (question_mode or "proposed_efe").strip()
0501:         if mode == "vlm_best_question":
0502:             return (
0503:                 "Put your own best next yes/no question as Question id=1. "
0504:                 "Choose it by considering answer consistency, visibility, balanced split, and diversity. "
0505:                 "Questions id=2..4 are alternates."
0506:             )
0507:         if mode == "first_question":
0508:             return (
0509:                 "Put a simple natural first clarification question as Question id=1. "
0510:                 "It should be valid and visible, but it is a first-question baseline, not optimized by backend EFE."
0511:             )
0512:         if mode == "random_question":
0513:             return (
0514:                 "Return four distinct useful questions from different visual categories when possible; "
0515:                 "the backend will choose one randomly for this baseline."
0516:             )
0517:         if mode == "proposed_efe":
0518:             return (
0519:                 "Return four diverse valid questions. The backend will score them by balanced yes/no split "
0520:                 "with a small diversity preference."
0521:             )
0522:         return "Follow the Ask2Act protocol and return valid diverse clarification questions."
0523: 
0524:     @classmethod
0525:     def _base_payload(
0526:         cls,
0527:         task_id: int,
0528:         round_idx: int,
0529:         image_id: str,
0530:         target: str,
0531:         candidates: List[Candidate],
0532:         *,
0533:         question_mode: str | None = None,
0534:         prompt_type: str | None = None,
0535:         candidate_state: Dict[str, Any] | None = None,
0536:         single_candidate_audit: bool = False,
0537:     ) -> Dict[str, Any]:
0538:         payload = {
0539:             "Task_ID": task_id,
0540:             "Round": round_idx,
0541:             "Image": image_id,
0542:             "Target_Instruction": target,
0543:             "Candidates": cls._candidate_payload(candidates),
0544:         }
0545:         if question_mode:
0546:             payload["Question_Mode"] = question_mode
0547:             payload["Question_Mode_Instruction"] = cls._question_mode_instruction(question_mode)
0548:         if prompt_type:
0549:             payload["Prompt_Type"] = prompt_type
0550:         if candidate_state is not None:
0551:             payload["Candidate_State"] = candidate_state
0552:         if single_candidate_audit:
0553:             payload["Single_Candidate_Audit"] = True
0554:             payload["Single_Candidate_Audit_Instruction"] = (
0555:                 "This is a partial-description trial with only one detector candidate. "
0556:                 "Inspect whether the annotated image suggests other plausible unboxed objects. "
0557:                 "Do not accept the single candidate only because it is alone; accept only if it visibly matches the instruction."
0558:             )
0559:         return payload
0560: 
0561:     @classmethod
0562:     def _start_payload(
0563:         cls,
0564:         task_id: int,
0565:         image_id: str,
0566:         target: str,
0567:         candidates: List[Candidate],
0568:         **kwargs,
0569:     ) -> Dict[str, Any]:
0570:         return {"Head": "start", **cls._base_payload(task_id, 1, image_id, target, candidates, **kwargs)}
0571: 
0572:     @classmethod
0573:     def _answer_payload(
0574:         cls,
0575:         task_id: int,
0576:         round_idx: int,
0577:         image_id: str,
0578:         target: str,
0579:         candidates: List[Candidate],
0580:         asked_history: Dict[str, Any],
0581:         **kwargs,
0582:     ) -> Dict[str, Any]:
0583:         payload = {"Head": "answer", **cls._base_payload(task_id, round_idx, image_id, target, candidates, **kwargs)}
0584:         payload["Asked"] = asked_history
0585:         return payload
0586: 
0587:     @staticmethod
0588:     def _score_questions(protocol: Dict[str, Any], *, sort_by_efe: bool = True) -> List[ScoredQuestion]:
0589:         questions = protocol.get("Question", [])
```

The output must end in exactly one JSON object. The backend extracts the final valid protocol object even if the model first emits optional `<think>` reasoning:

```python
0031:         raise ValueError("Empty data URL")
0032:     _, sep, payload = data_url.partition(",")
0033:     if not sep or not payload:
0034:         raise ValueError("Invalid data URL")
0035:     return base64.b64decode(payload)
0036: 
0037: 
0038: def _find_all_complete_json_object_spans(text: str) -> List[Tuple[int, int]]:
0039:     if not text:
0040:         return []
0041:     in_str = False
0042:     escape = False
0043:     stack = []
0044:     spans = []
0045:     start_idx = None
0046:     for idx, char in enumerate(text):
0047:         if in_str:
0048:             if escape:
0049:                 escape = False
0050:             elif char == "\\":
0051:                 escape = True
0052:             elif char == '"':
0053:                 in_str = False
0054:             continue
0055:         if char == '"':
0056:             in_str = True
0057:             continue
0058:         if char == "{":
0059:             if not stack:
0060:                 start_idx = idx
0061:             stack.append("{")
0062:         elif char == "}":
0063:             if stack:
0064:                 stack.pop()
0065:                 if not stack and start_idx is not None:
0066:                     spans.append((start_idx, idx))
0067:                     start_idx = None
0068:     return spans
0069: 
0070: 
0071: def extract_protocol_json(text: str) -> Dict[str, Any]:
0072:     text = (text or "").strip()
0073:     if not text:
0074:         raise ValueError("Empty model output")
0075: 
0076:     # Prefer the final answer region after the model's optional reasoning block.
0077:     # If the model includes JSON-like scratch content in <think>, this keeps us
0078:     # focused on the protocol JSON that should actually drive the robot.
0079:     search_texts = []
0080:     think_end = text.lower().rfind("</think>")
0081:     if think_end >= 0:
0082:         search_texts.append(text[think_end + len("</think>") :].strip())
0083:     search_texts.append(text)
0084: 
0085:     parsed: List[Dict[str, Any]] = []
0086:     for search_text in search_texts:
```

If the VLM output is truncated or illegal, the backend performs bounded repair calls and finally falls back to a legal backend protocol rather than letting robot control depend on invalid text:

```python
0340: 
0341:     def _generate_protocol(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
0342:         response = self._chat(messages=messages)
0343:         try:
0344:             raw, protocol = self._extract_response_protocol(response)
0345:         except Exception as first_error:
0346:             follow = (
0347:                 "Your previous response was unusable for the robot because it did not end in valid Ask2Act JSON. "
0348:                 f"Reason: {type(first_error).__name__}.\n"
0349:                 "If it was truncated or only contained <think>, discard that reasoning and produce the final object now.\n"
0350:                 "Output ONLY one JSON object now. Do NOT output <think>, markdown, explanations, or code fences.\n"
0351:                 'The object Head must be "probose" or "decision", and it must include Task_ID.\n'
0352:                 "Use the exact schema from the system prompt. Start with { and end with }.\n"
0353:             )
0354:             second = self._chat(
0355:                 messages=messages + [{"role": "user", "content": follow}],
0356:                 max_tokens=self._repair_token_budget(1800),
0357:             )
0358:             try:
0359:                 raw_second, protocol = self._extract_response_protocol(second)
0360:                 raw = raw_second
0361:             except Exception as second_error:
0362:                 final_repair = (
0363:                     "Still invalid. Return ONLY the final Ask2Act protocol JSON. "
0364:                     f"Reason: {type(second_error).__name__}.\n"
0365:                     "No <think>. No prose. No markdown. No code fence.\n"
0366:                     "Start with { and end with }."
0367:                 )
0368:                 third = self._chat(
0369:                     messages=messages + [{"role": "user", "content": final_repair}],
0370:                     max_tokens=self._repair_token_budget(1600),
0371:                 )
0372:                 raw_third, protocol = self._extract_response_protocol(third)
0373:                 raw = raw_third
0374: 
0375:         if self._has_forbidden_question_reference(protocol):
0376:             repair = (
0377:                 "REWRITE the final protocol JSON.\n"
0378:                 "Your previous questions mentioned candidate numbers/tags/marks/ids. That is forbidden.\n"
0379:                 "Ask only about visible object properties such as color, left/right position, relative position, "
0380:                 "size, or shape. Do not mention numbers, marks, tags, display IDs, candidate IDs, or bbox values.\n"
0381:                 "Prefer balanced visible-attribute splits when possible.\n"
0382:                 "Keep the same protocol schema, Candidate_State, count fields, yes_candidates, and no_candidates. "
0383:                 "Output only one JSON object; no <think>."
0384:             )
0385:             repaired = self._chat(
0386:                 messages=messages + [{"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}]
0387:                 + [{"role": "user", "content": repair}],
0388:                 max_tokens=self._repair_token_budget(1200),
0389:             )
0390:             raw_repaired, repaired_protocol = self._extract_response_protocol(repaired)
0391:             if self._has_forbidden_question_reference(repaired_protocol):
0392:                 raise ValueError("VLM proposed a question about candidate numbers/tags after repair")
0393:             raw = raw_repaired
0394:             protocol = repaired_protocol
0395: 
0396:         if not self._has_candidate_state(protocol):
0397:             repair = (
0398:                 "REWRITE the final protocol JSON to include bookkeeping Candidate_State.\n"
0399:                 "Do not change the user-facing question text unless necessary.\n"
0400:                 'Add "Candidate_State": {"plausible": [candidate_id, ...], "eliminated": [candidate_id, ...]}.\n'
0401:                 "Candidate_State must use candidate_id strings from Candidates, not display numbers. "
0402:                 "Output only one JSON object; no <think>, markdown, prose, or code fence."
0403:             )
0404:             repaired = self._chat(
0405:                 messages=messages + [{"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}]
0406:                 + [{"role": "user", "content": repair}],
0407:                 max_tokens=self._repair_token_budget(1400),
0408:             )
0409:             try:
0410:                 raw_repaired, repaired_protocol = self._extract_response_protocol(repaired)
0411:                 if self._has_candidate_state(repaired_protocol):
0412:                     return raw_repaired, repaired_protocol
0413:             except Exception:
0414:                 pass
0415: 
0416:         return raw, protocol
0417: 
0418:     @staticmethod
0419:     def _candidate_display_id(candidate: Candidate) -> int | str:
0420:         if candidate.display_id is not None:
0421:             return candidate.display_id
0422:         suffix = candidate.candidate_id.rsplit("_", 1)[-1]
0423:         try:
0424:             return int(suffix)
0425:         except ValueError:
0426:             return candidate.candidate_id
0427: 
0428:     @classmethod
0429:     def _candidate_payload(cls, candidates: List[Candidate]) -> Dict[str, Any]:
0430:         def rounded(values: List[float] | Tuple[float, ...], ndigits: int = 1) -> List[float]:
0431:             return [round(float(value), ndigits) for value in values]
0432: 
```

Question ranking for `proposed_efe`:

```python
1040:         y_count = int(question.count.get("y") or 0)
1041:         n_count = int(question.count.get("n") or 0)
1042:         if total <= 0:
1043:             balance = 1.0
1044:         else:
1045:             balance = abs(y_count - n_count) / max(total, 1)
1046: 
1047:         known_questions = set()
1048:         previous_categories: List[str] = []
1049:         if session is not None:
1050:             for turn in session.question_history:
1051:                 known_questions.add(cls._normalized_question_text(turn.text))
1052:                 previous_categories.append(cls.question_category(turn.text))
1053: 
1054:         category = cls.question_category(question.text)
1055:         category_prior = {
1056:             "spatial": 0.00,
1057:             "relation": 0.02,
1058:             "shape_material": 0.03,
1059:             "size": 0.04,
1060:             "color": 0.08,
1061:             "other": 0.10,
1062:         }.get(category, 0.10)
1063:         repeat_category_penalty = 0.12 * previous_categories.count(category)
1064:         duplicate_penalty = 1.0 if cls._normalized_question_text(question.text) in known_questions else 0.0
1065:         no_split_penalty = 1.0 if total > 1 and (y_count <= 0 or n_count <= 0) else 0.0
1066:         isolate_penalty = 0.08 if total > 2 and min(y_count, n_count) == 1 else 0.0
1067:         score = balance + category_prior + repeat_category_penalty + duplicate_penalty + no_split_penalty + isolate_penalty
1068:         return score, question.efe_score, question.id
1069: 
1070:     @classmethod
1071:     def rank_questions_for_mode(
1072:         cls,
1073:         protocol: Dict[str, Any],
1074:         session: SessionState | None,
1075:         *,
1076:         mode: str | None,
1077:     ) -> List[ScoredQuestion]:
1078:         if session is None:
1079:             questions = cls._score_questions(protocol, sort_by_efe=False)
1080:         else:
1081:             used_texts = {cls._normalized_question_text(turn.text) for turn in session.question_history}
1082:             questions = []
1083:             for raw_question in protocol.get("Question") or []:
1084:                 if not isinstance(raw_question, dict):
1085:                     continue
1086:                 question = cls._question_from_protocol_question(
1087:                     raw_question,
1088:                     session,
1089:                     question_id=len(questions) + 1,
1090:                 )
1091:                 if question is None:
1092:                     continue
1093:                 normalized = cls._normalized_question_text(question.text)
1094:                 if normalized in used_texts:
1095:                     continue
1096:                 questions.append(question)
1097:                 used_texts.add(normalized)
1098:             if len(questions) < 4:
1099:                 questions.extend(
1100:                     cls._fallback_questions(
1101:                         session,
1102:                         start_id=len(questions) + 1,
1103:                         exclude_texts={cls._normalized_question_text(question.text) for question in questions},
1104:                     )
1105:                 )
1106:             questions = questions[:4]
1107:         if mode == "proposed_efe":
1108:             return sorted(questions, key=lambda question: cls._rank_question_key(question, session))
1109:         return questions
```

The rank key combines split balance and small penalties:

```text
balance = |y - n| / total
rank_score = balance + category_prior + repeat_category_penalty + duplicate_penalty + no_split_penalty + isolate_penalty
smaller rank_score is better
```

This approximates the EFE intuition: a good question should reduce uncertainty by splitting the plausible candidates evenly, while also staying natural and diverse.

## 7. Full Project VLM System Prompt

```text
You are a vision-language module inside a robot grasping system.
There is exactly ONE target object described by "Target_Instruction".
Your job is to propose YES/NO questions to identify the target.

The first user message contains an <image> token followed by ONE JSON object.
Later answer messages contain JSON and refer to the same image.
The image is the GroundingDINO candidate overlay, not a plain camera image:
- every candidate bbox is drawn on the image
- each bbox has a nearby tag and object label, e.g. "1 cup"
- the numeric tag equals Candidates[<candidate_name>].display_id / visual_tag

You MAY include <think>...</think> before the final JSON.
Use <think> to reason like the original Ask2Act loop: identify visible attributes,
apply every human answer, maintain the plausible candidate set, then choose questions.
Keep <think> concise, but correctness is more important than being extremely short.
If the token budget is limited, skip <think> and output the final JSON directly.
You MUST end your message with EXACTLY ONE FINAL JSON object.
Do NOT output markdown, prose, or code fences outside the optional <think> and final JSON.

=================================================
I/O HEADS

INPUT Heads:
- "start": only once at the beginning. Contains the initial Candidates list.
- "answer": repeated each round. Contains the same Candidates list plus the full history of asked questions and the human answers.

Every input with Head="start" is a fresh independent trial. Ignore any prior
prompt, image, candidate set, question, answer, or target unless it appears in
the current message sequence for this trial.

OUTPUT Heads:
- "probose": after "start", propose 4 questions for Round 1 when multiple candidates remain plausible.
- "decision": after each "answer", decide whether to grasp; if not, propose the next round questions.
  You may also output "decision" after "start" only when Candidate_State filtering leaves exactly one plausible candidate.
  For partial single-candidate audit only, you may output "decision" after "start" with Grasp="no" and
  Need_Object_Recall=true when the image shows another unboxed object that could match Target_Instruction.

=================================================
TASK_ID AND ROUND RULES (HARD)

Always copy "Task_ID" exactly from input to output.

Input Head="start" with Round=1 -> Output Head="probose" MUST have Round=1.

Input Head="answer" with Round=r:
- If you output Grasp="yes": Output Round MUST be r.
- If you output Grasp="no" and output new questions: Output Round MUST be r+1.

=================================================
FORBIDDEN TOPICS (HARD)

You MUST NEVER ask any question about:
- detection score, confidence, probability, threshold, logits, model outputs
- numeric bounding box coordinates, IoU, NMS, thresholds
- candidate numbers, visual tags, marks, display IDs, candidate IDs, or image sequence numbers
Treat candidate "score" as hidden metadata. Never mention it.

=================================================
CANDIDATE AND ANSWER CONSISTENCY (HARD)

You may only choose among the provided GroundingDINO Candidates.
Use the annotated numeric tags in the image to connect visible objects to candidate names:
- display_id / visual_tag is only for human-visible reference
- final Target.name MUST be the candidate key such as "cand_002", never "2"
- Do NOT mention display_id / visual_tag / numbers / marked objects in questions to the human.
- Human-facing questions must describe object properties only: color, left/right position, relative position, size, shape, handle/lid, or other visual attributes.

After every human answer, update the internal plausible candidate set using ALL Asked history.
The human answers are authoritative. Do not ignore them and do not rely on a fixed feature order.
The plausible candidate set must monotonically shrink: once a candidate is ruled out by any answer, it stays ruled out.
For every proposed question, count.total MUST equal the size of the current plausible candidate set after all Asked history.
You MUST NOT output Grasp="yes" unless exactly one candidate remains plausible.
You MUST NEVER select a candidate that contradicts any previous human answer.
Every output MUST include machine-readable Candidate_State:
- "plausible": candidate IDs that remain possible after applying all answers so far
- "eliminated": candidate IDs ruled out after applying all answers so far
Candidate_State and yes_candidates/no_candidates are for backend bookkeeping only. You may output candidate IDs inside Candidate_State,
yes_candidates/no_candidates, and Target.name,
but you must never mention candidate IDs, display tags, or numbers in human-facing question text.
Examples:
- If the human answered YES to "Is the target cup orange?", do not select a non-orange cup.
- If the human answered NO to "Is the target cup red?", do not select a red cup.
- If the human answered NO to "Is the target cup the left orange?", and there are two orange cups, the left orange is no longer plausible.
- If the human answered NO to purple, NO to yellow, and YES to orange, the plausible set is only orange cups.
  If multiple orange cups remain, ask about their relative position, not another broad color question.
- If the human answered NO to "Is the target cup green?", eliminate ALL plausible green cups, not just one green cup.
- If the human answered YES to "Is the target cup green?", eliminate ALL non-green plausible cups.
- Apply the same grouped elimination rule to size, material, lid/handle, left/right, top/bottom, and relative-position questions.

Reasoning checklist for <think>:
- Note each candidate's visible color and relative position using candidate keys internally.
- Apply all previous answers and write the current plausible candidate keys internally; the list should get smaller after informative answers.
- Compare your updated plausible set against input Candidate_State. Never re-add a candidate listed as eliminated.
- If exactly one candidate remains, output Grasp="yes" for that candidate.
- If multiple candidates remain, propose questions that distinguish among those remaining candidates.
- Never let a balanced count override answer consistency.

Question quality:
- Prefer questions about visible color, left/right relation, relative position, size, shape, handle/lid, or material.
- Good examples: "Is the target cup orange?", "Is the target the right orange cup?", "Is the target left of the blue cup?"
- Bad examples: "Is the target the cup marked 3?", "Is the target candidate 6?", "Is the target the object with tag 2?"
- Make the human-facing question short and natural. Prefer wording like "Is it the yellow cup?",
  "Is it on the left side?", or "Does it have a handle?"
- Avoid robotic/backend wording such as "current plausible candidates", "candidate set", "target object",
  "visible objects", or "left half of the visible objects" in the question text.
- Avoid "front row", "back row", or "row" questions unless rows are visually obvious and split the current plausible candidates cleanly.
- Do not repeat or paraphrase a question already answered.
- Do not ask a question whose answer is already known from Asked history.
- Avoid asking mostly color questions. Across four questions, use at least three categories when the scene supports it:
  color, size, material/shape/lid/handle, left/right/top/bottom, and relative position.
- Color can be used when it is discriminative, but do not ask more than two color-only questions unless color is the only visible distinction.

Question selection:
- The robot scores your proposed questions from count.y/count.n and prefers the most balanced split.
- First ensure every proposed question is relevant to the CURRENT plausible candidates after all answers.
- Then prefer visible-trait questions that split those plausible candidates close to half/half while staying diverse.
- Avoid broad questions whose answer is already implied by history.
- When color is already known but multiple same-color objects remain, ask left/right, upper/lower,
  closer/farther, or relative-to-another-object questions among that same-color group.
- Avoid isolating one candidate unless only two candidates remain, or no balanced visible trait exists.
- Your four proposed questions should be distinct useful partitions of the same current plausible set.

Question_Mode:
- proposed_efe: return four diverse valid questions; the backend will choose by balanced split with diversity preference.
- first_question: put a simple natural first clarification question as Question id=1.
- random_question: make all four questions useful and varied because the backend will randomly choose one.
- vlm_best_question: Question id=1 MUST be your own best next question after considering visibility, answer consistency,
  split balance, and diversity. Questions id=2..4 are alternates.

Partial single-candidate audit:
- If Prompt_Type="partial", Single_Candidate_Audit=true, and only one candidate is provided, do not accept it merely
  because it is the only candidate.
- If the visible candidate matches Target_Instruction and no other visible unboxed object appears to match,
  output a "decision" with Grasp="yes" for that candidate. Do not ask questions.
- If another visible unboxed object appears to match Target_Instruction, output a "decision" with Grasp="no",
  Need_Object_Recall=true, and Recall_Target set to the broad object noun, such as "cup". Do not ask questions yet.
- Do not use Need_Object_Recall merely because the scene contains other objects. Use it only when another unboxed
  object could match the instruction, e.g. a second red cup when Target_Instruction says "red cup".

Partial broad-candidate recall:
- If Prompt_Type="partial" and the Candidates are broader than Target_Instruction, first filter by the instruction.
  Example: if Target_Instruction says "red cup" but Candidates are labeled "cup", inspect the image and keep only
  candidates that visibly could be the red cup.
- Put non-matching broad-recall candidates directly in Candidate_State.eliminated. Do not include eliminated
  candidates in yes_candidates/no_candidates.
- If exactly one instruction-matching candidate remains after this filtering, output a "decision" with Grasp="yes".
- If multiple instruction-matching candidates remain, output exactly 4 legal questions whose count and
  yes_candidates/no_candidates are computed over only those remaining plausible candidates.

=================================================
WHAT YOU MUST OUTPUT FOR EACH QUESTION (HARD)

For EACH proposed question, you MUST output:

"count": { "total": <int>, "y": <int>, "n": <int> }
"yes_candidates": ["<candidate_name>", ...]
"no_candidates": ["<candidate_name>", ...]

Meaning:
- total = number of candidates that are still plausible GIVEN the full Asked history.
- y = how many of those plausible candidates would remain if the human answers YES to this question.
- n = how many would remain if the human answers NO to this question.
- yes_candidates = exactly the candidate IDs counted by y.
- no_candidates = exactly the candidate IDs counted by n.

Constraints:
- total >= 1
- 0 <= y <= total, 0 <= n <= total
- y + n MUST equal total (within integer arithmetic).
- yes_candidates and no_candidates MUST be disjoint and their union MUST equal the current plausible Candidate_State.
- len(yes_candidates) MUST equal y. len(no_candidates) MUST equal n.
- yes_candidates/no_candidates MUST NOT include eliminated or unknown candidate IDs.
- The question text MUST match the yes/no candidate split and MUST NOT contradict any previous human answer.
- Output integers (not strings).

Do NOT output candidate lists inside human-facing questions or prose.
You MUST output the backend Candidate_State object separately in the JSON.

=================================================
INPUT FORMAT

Head="start"
{
  "Head": "start",
  "Task_ID": <int>,
  "Round": 1,
  "Image": <string>,
  "Target_Instruction": <string>,
  "Prompt_Type": "clear" or "ambiguous" or "partial" or omitted,
  "Question_Mode": "proposed_efe" or "first_question" or "random_question" or "vlm_best_question" or omitted,
  "Single_Candidate_Audit": <bool, optional>,
  "Candidate_State": {
    "plausible": ["<candidate_name>", ...],
    "eliminated": ["<candidate_name>", ...],
    "last_removed": ["<candidate_name>", ...]
  },
  "Candidates": {
    "<name>": {
      "display_id": <int or string>,
      "visual_tag": <string>,
      "label": <string>,
      "bbox": [x_min,y_min,x_max,y_max],
      "bbox_center": [x_center,y_center],
      "left_to_right_rank": <int>,
      "top_to_bottom_rank": <int>,
      "score": <float>
    },
    ...
  }
}

Head="answer"
{
  "Head": "answer",
  "Task_ID": <int>,
  "Round": <int>,
  "Image": <string>,
  "Target_Instruction": <string>,
  "Prompt_Type": "clear" or "ambiguous" or "partial" or omitted,
  "Question_Mode": "proposed_efe" or "first_question" or "random_question" or "vlm_best_question" or omitted,
  "Single_Candidate_Audit": <bool, optional>,
  "Candidate_State": {
    "plausible": ["<candidate_name>", ...],
    "eliminated": ["<candidate_name>", ...],
    "last_removed": ["<candidate_name>", ...]
  },
  "Candidates": {
    "<name>": {
      "display_id": <int or string>,
      "visual_tag": <string>,
      "label": <string>,
      "bbox": [x_min,y_min,x_max,y_max],
      "bbox_center": [x_center,y_center],
      "left_to_right_rank": <int>,
      "top_to_bottom_rank": <int>,
      "score": <float>
    },
    ...
  },
  "Asked": {
    "Round 1": [ { "id": <int>, "text": <string>, "answer": "y" or "n" }, ... ],
    "Round 2": [ { "id": <int>, "text": <string>, "answer": "y" or "n" }, ... ],
    ...
  }
}

=================================================
OUTPUT FORMAT

A) After Head="start": output Head="probose" unless Candidate_State filtering leaves exactly one plausible candidate
{
  "Head": "probose",
  "Task_ID": <int>,
  "Round": 1,
  "Target_Instruction": <string>,
  "Candidate_State": {
    "plausible": ["<candidate_name>", ...],
    "eliminated": ["<candidate_name>", ...]
  },
  "Question": [
    { "id": 1, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] },
    { "id": 2, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] },
    { "id": 3, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] },
    { "id": 4, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] }
  ]
}

B) After Head="answer": output Head="decision"

For partial single-candidate audit after Head="start", request broader object recall when needed:
{
  "Head": "decision",
  "Task_ID": <int>,
  "Round": 1,
  "Grasp": "no",
  "Need_Object_Recall": true,
  "Recall_Target": "<broad object noun such as cup>",
  "Candidate_State": {
    "plausible": ["<current_single_candidate_name>"],
    "eliminated": []
  },
  "Reason": "<short reason, e.g. another unboxed red cup may match>"
}

STOP:
{
  "Head": "decision",
  "Task_ID": <int>,
  "Round": <int>,
  "Grasp": "yes",
  "Candidate_State": {
    "plausible": ["<selected_candidate_name>"],
    "eliminated": ["<candidate_name>", ...]
  },
  "Target": {
    "name": "<candidate_name>",
    "label": "<string>",
    "bbox": [x_min, y_min, x_max, y_max],
    "score": <float>
  }
}

CONTINUE:
{
  "Head": "decision",
  "Task_ID": <int>,
  "Round": <int + 1>,
  "Grasp": "no",
  "Candidate_State": {
    "plausible": ["<candidate_name>", ...],
    "eliminated": ["<candidate_name>", ...]
  },
  "Question": [
    { "id": 1, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] },
    { "id": 2, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] },
    { "id": 3, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] },
    { "id": 4, "text": <string>, "count": { "total": <int>, "y": <int>, "n": <int> }, "yes_candidates": ["<candidate_name>", ...], "no_candidates": ["<candidate_name>", ...] }
  ]
}

=================================================
FINAL REMINDERS

- End with exactly ONE protocol JSON object.
- Output JSON only (no markdown, no extra text after JSON).
- The final JSON is mandatory; never spend the whole output budget on <think>.
- Always include count.total/count.y/count.n for each question.
- Always include yes_candidates/no_candidates for each question.
- Always include Candidate_State with plausible and eliminated candidate_id arrays.
```

## 8. Real-Robot Grasping Pipeline

After target resolution, online execution uses the selected candidate bbox and the current RGB-D observation.

Pipeline:

1. Load RGB, depth, intrinsics, and camera extrinsics from Stretch observation metadata.
2. Map the selected candidate bbox to the depth frame, including rotated-image handling.
3. Optionally predict a Segment Anything mask from the bbox.
4. Generate a target point cloud using the mask; if mask point cloud has too few points, fall back to bbox crop.
5. Estimate a geometric grasp from target points.
6. Convert the grasp to Stretch top-down motion targets using SimpleIK when available.
7. Dispatch the waypoint trajectory to the Stretch transport server.

SAM/mask and point-cloud selection path:

```python
0883:             target_bbox_xyxy=list(resolved_target.bbox_xyxy),
0884:             point_cloud_count=None,
0885:             selected_grasp_score=resolved_target.score,
0886:             trajectory_waypoint_count=0,
0887:             pipeline_run_dir=None,
0888:             success=True,
0889:             note="Preview-only local plan. Set ASK2ACT_PIPELINE_MODE=local_sim to invoke the current Ask2Act grasp stack.",
0890:         )
0891:         dispatch_payload = {
0892:             "pipeline_mode": "mock",
0893:             "planner_backend": "mock-preview",
0894:             "target_bbox_2d": list(bbox_tuple),
0895:             "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
0896:             "resolved_target": resolved_target.model_dump(),
0897:             "dry_run": dry_run,
0898:         }
0899:         return {
0900:             "ok": True,
0901:             "dry_run": dry_run,
0902:             "pipeline_mode": "mock",
0903:             "plan_summary": plan_summary.model_dump(),
0904:             "pipeline_result": None,
0905:             "dispatch_payload": dispatch_payload,
0906:         }
0907: 
0908:     def _local_sim_plan(self, resolved_target: ResolvedTarget, dry_run: bool) -> Dict[str, Any]:
0909:         if dry_run:
0910:             return self._mock_plan(resolved_target, dry_run=True)
0911: 
0912:         from simulation.ask2act_grasp.pipeline import run_pipeline
0913: 
0914:         bbox_tuple = self._bbox_to_int_tuple(resolved_target.bbox_xyxy)
0915:         run_dir = self._default_run_dir()
0916:         result = run_pipeline(
0917:             scene_config_path=self.scene_config_path or None,
0918:             grasp_config_path=self.grasp_config_path or None,
0919:             run_dir=run_dir,
0920:             target_bbox_2d=bbox_tuple,
0921:             headless=self.headless,
0922:             show_viewer_ui=self.show_viewer_ui,
0923:         )
0924:         plan_summary = GraspPlanResult(
0925:             pipeline_mode="local_sim",
0926:             planner_backend=result.planner_backend,
0927:             target_bbox_xyxy=list(resolved_target.bbox_xyxy),
0928:             point_cloud_count=result.point_cloud_count,
0929:             selected_grasp_score=result.selected_grasp_score,
0930:             trajectory_waypoint_count=len(result.trajectory or []),
0931:             pipeline_run_dir=str(run_dir),
0932:             success=result.success,
0933:             note=result.error,
0934:         )
0935:         dispatch_payload = {
0936:             "pipeline_mode": "local_sim",
0937:             "planner_backend": result.planner_backend,
0938:             "target_bbox_2d": list(bbox_tuple),
0939:             "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
0940:             "trajectory": result.trajectory,
0941:             "run_dir": str(run_dir),
0942:             "resolved_target": resolved_target.model_dump(),
0943:         }
0944:         pipeline_result = {
0945:             "success": result.success,
0946:             "scene_xml_path": result.scene_xml_path,
0947:             "point_cloud_count": result.point_cloud_count,
0948:             "selected_grasp_score": result.selected_grasp_score,
0949:             "planner_backend": result.planner_backend,
0950:             "trajectory": result.trajectory,
0951:             "intermediate": result.intermediate,
0952:             "error": result.error,
0953:         }
0954:         return {
0955:             "ok": bool(result.success),
0956:             "dry_run": False,
0957:             "pipeline_mode": "local_sim",
0958:             "plan_summary": plan_summary.model_dump(),
0959:             "pipeline_result": pipeline_result,
0960:             "dispatch_payload": dispatch_payload,
0961:         }
0962: 
0963:     def _real_pointcloud_plan(
0964:         self,
0965:         resolved_target: ResolvedTarget,
0966:         observation_metadata: Dict[str, Any] | None,
0967:         dry_run: bool,
0968:         rotate_clockwise_90: bool,
0969:     ) -> Dict[str, Any]:
0970:         self._ensure_grasp_import_path()
0971: 
0972:         import numpy as np
0973:         from ask2act_grasp.perception.point_cloud_gen import PointCloudGenerator
0974:         from ask2act_grasp.planning.motion_planner import MotionPlanner
0975:         from ask2act_grasp.types import GraspCandidate
0976:         from ask2act_grasp.utils.config_loader import load_grasp_config, load_scene_config
```

Fallback geometric grasp estimate:

```python
0760:                 "rotated_observation_clockwise_90": bool(rotate_clockwise_90),
0761:                 "camera_intrinsics": intrinsics,
0762:                 "camera_extrinsics": extrinsics,
0763:                 "current_state_for_planning": current_state,
0764:             },
0765:             "error": None,
0766:             "note": "Base preposition only; reobserve and replan before grasping the same target.",
0767:         }
0768:         (run_dir / "real_pointcloud_plan.json").write_text(
0769:             json.dumps(self._to_jsonable(pipeline_result), indent=2),
0770:             encoding="utf-8",
0771:         )
0772:         plan_summary = GraspPlanResult(
0773:             pipeline_mode="real_pointcloud",
0774:             planner_backend=motion_plan.backend,
0775:             target_bbox_xyxy=list(resolved_target.bbox_xyxy),
0776:             point_cloud_count=int(point_count),
0777:             selected_grasp_score=float(resolved_target.score),
0778:             trajectory_waypoint_count=len(trajectory),
0779:             pipeline_run_dir=str(run_dir),
0780:             success=True,
0781:             note="SimpleIK requested base preposition; execute this move, reobserve, then replan the grasp.",
0782:         )
0783:         dispatch_payload = {
0784:             "pipeline_mode": "real_pointcloud",
0785:             "planner_backend": motion_plan.backend,
0786:             "target_bbox_2d": list(applied_bbox_tuple),
0787:             "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
0788:             "trajectory": trajectory,
0789:             "run_dir": str(run_dir),
0790:             "resolved_target": resolved_target.model_dump(),
0791:             "geometric_grasp": self._to_jsonable(geometric_grasp),
0792:             "motion_plan_metadata": self._to_jsonable(motion_plan.metadata),
0793:             "simple_ik": simple_ik_status,
0794:             "selected_pointcloud_option": selected_option_name,
0795:             "pointcloud_diagnostics_path": str(run_dir / "real_pointcloud_diagnostics.json"),
0796:             "preposition_only": True,
0797:         }
0798:         return {
0799:             "ok": True,
0800:             "dry_run": bool(dry_run),
0801:             "pipeline_mode": "real_pointcloud",
0802:             "plan_summary": plan_summary.model_dump(),
0803:             "pipeline_result": self._to_jsonable(pipeline_result),
0804:             "dispatch_payload": dispatch_payload,
0805:         }
0806: 
0807:     @staticmethod
0808:     def _fallback_geometric_grasp(points_xyz, table_top_z_m: float, max_gripper_width_m: float) -> Dict[str, Any]:
0809:         import numpy as np
0810: 
0811:         points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
0812:         if len(points) < 3:
0813:             raise RuntimeError("Need at least three target points for geometric grasp fallback")
0814:         z_top = float(np.percentile(points[:, 2], 95.0))
0815:         z_bottom = float(np.percentile(points[:, 2], 5.0))
0816:         center_z = float((z_top + z_bottom) / 2.0)
0817:         slice_mask = np.abs(points[:, 2] - center_z) < 0.015
0818:         slice_pts = points[slice_mask] if int(np.count_nonzero(slice_mask)) >= 10 else points
0819:         xy = np.asarray(slice_pts[:, :2], dtype=np.float64)
0820:         center_xy = np.median(xy, axis=0)
0821:         centered = xy - center_xy[None, :]
0822:         if len(centered) >= 3:
0823:             cov = np.cov(centered.T)
0824:             eigvals, eigvecs = np.linalg.eigh(cov)
```

Top-down target solving applies side biases, rubber contact offsets, wrist yaw, SimpleIK, and FK validation:

```python
0480:             base_world_translation[1] = float(current_state.get("base_y", 0.0))
0481:         lateral_x_m = float(desired_rubber_xyz[0] - base_world_translation[0])
0482:         tuning = self._topdown_tuning(use_slender_tuning=use_slender_tuning)
0483:         side_x_bias_applied_m = 0.0
0484:         side_y_bias_applied_m = tuning["left_center_y"]
0485:         side_bias_region = "center"
0486:         if abs(lateral_x_m) > tuning["side_deadband"]:
0487:             side_x_bias_applied_m = float(math.copysign(tuning["side_x"], lateral_x_m))
0488:             if lateral_x_m < 0.0:
0489:                 side_bias_region = "right"
0490:                 side_x_bias_applied_m -= tuning["right_extra_x"]
0491:                 side_y_bias_applied_m = tuning["right_y"]
0492:             else:
0493:                 side_bias_region = "left"
0494:             desired_rubber_xyz[0] += side_x_bias_applied_m
0495:         desired_rubber_xyz[1] += side_y_bias_applied_m
0496:         gripper_open_cmd = self._topdown_gripper_open_cmd(requested_open_width)
0497:         wrist_to_grasp_center_local = np.asarray(topdown_wrist_to_grasp_center_offset_m(), dtype=float)
0498:         grasp_center_to_rubber_local = np.asarray(topdown_grasp_center_to_rubber_offset_m(gripper_open_cmd), dtype=float)
0499:         rubber_local_correction = np.array(
0500:             [
0501:                 tuning["rubber_x"],
0502:                 tuning["rubber_y"],
0503:                 tuning["rubber_z"],
0504:             ],
0505:             dtype=float,
0506:         )
0507:         grasp_center_to_rubber_local = grasp_center_to_rubber_local + rubber_local_correction
0508: 
0509:         base_rotate_guess = float(
0510:             np.clip(
0511:                 math.atan2(float(desired_rubber_xyz[0]), max(-float(desired_rubber_xyz[1]), 1e-3)),
0512:                 -self.BASE_ROTATE_LIMIT_RAD,
0513:                 self.BASE_ROTATE_LIMIT_RAD,
0514:             )
0515:         )
0516:         use_geometric_yaw = bool(force_wrist_yaw) or (
0517:             float(requested_open_width) < GEOMETRIC_TOP_DOWN_WRIST_YAW_OPEN_WIDTH_THRESHOLD_M
0518:         )
0519:         current_yaw = 0.0 if current_state is None else float(current_state.get("wrist_yaw", 0.0))
0520: 
0521:         def wrist_yaw_for_base(base_rotate_rad: float) -> float:
0522:             if not use_geometric_yaw:
0523:                 return 0.0
0524:             # grip_angle_rad is expressed in the world/table XY frame. Stretch's
0525:             # actual top-down jaw yaw is base_rotate + wrist_yaw, so command the
0526:             # wrist relative to the base rotation.
0527:             return self._select_wrist_yaw(
0528:                 self._normalize_angle(float(grip_angle_rad) - float(base_rotate_rad)),
0529:                 current_yaw=current_yaw,
0530:                 allow_pi_flip=True,
0531:             )
0532: 
0533:         desired_wrist_yaw = wrist_yaw_for_base(base_rotate_guess)
0534:         ik_result = None
0535:         wrist_model_error_local = np.zeros(3, dtype=float)
0536:         desired_wrist_yaw_pos = np.zeros(3, dtype=float)
0537:         wrist_to_grasp_center_world = np.zeros(3, dtype=float)
0538:         grasp_center_to_rubber_world = np.zeros(3, dtype=float)
0539:         wrist_model_error_world = np.zeros(3, dtype=float)
0540:         desired_grasp_center_xyz = desired_rubber_xyz.copy()
0541:         base_rotate = base_rotate_guess
0542:         lift_val = 0.0
0543:         arm_val = 0.0
0544:         for _ in range(5):
0545:             desired_wrist_yaw = wrist_yaw_for_base(base_rotate_guess)
0546:             total_yaw = base_rotate_guess + desired_wrist_yaw
0547:             wrist_to_grasp_center_world = rotate_topdown_offset_to_world_m(wrist_to_grasp_center_local, total_yaw)
0548:             grasp_center_to_rubber_world = rotate_topdown_offset_to_world_m(grasp_center_to_rubber_local, total_yaw)
0549:             wrist_model_error_world = rotate_topdown_offset_to_world_m(wrist_model_error_local, total_yaw)
0550:             desired_grasp_center_xyz = desired_rubber_xyz - grasp_center_to_rubber_world
0551:             desired_wrist_yaw_pos = (
0552:                 desired_grasp_center_xyz
0553:                 - base_world_translation
0554:                 - wrist_to_grasp_center_world
0555:                 - wrist_model_error_world
0556:             )
0557: 
0558:             ik_result = self.simple_ik.ik_rotary_base(desired_wrist_yaw_pos.tolist())
0559:             if ik_result is None:
0560:                 print(f"WARNING: SimpleIK returned None for target {desired_wrist_yaw_pos.tolist()}", flush=True)
0561:                 return None
0562: 
0563:             self.simple_ik.clip_with_joint_limits(ik_result)
0564:             base_rotate = float(ik_result["joint_mobile_base_rotation"])
0565:             lift_val = float(ik_result["joint_lift"])
0566:             arm_val = float(ik_result["joint_arm_l0"])
0567:             next_wrist_model_error_local = np.asarray(topdown_simpleik_wrist_model_error_m(lift_val, arm_val), dtype=float)
0568:             if (
0569:                 abs(self._normalize_angle(base_rotate - base_rotate_guess)) < 1e-5
0570:                 and np.linalg.norm(next_wrist_model_error_local - wrist_model_error_local) < 1e-6
0571:             ):
0572:                 wrist_model_error_local = next_wrist_model_error_local
0573:                 break
0574:             wrist_model_error_local = next_wrist_model_error_local
0575:             base_rotate_guess = base_rotate
0576: 
0577:         desired_wrist_yaw = wrist_yaw_for_base(base_rotate)
0578:         total_yaw = base_rotate + desired_wrist_yaw
0579:         wrist_to_grasp_center_world = rotate_topdown_offset_to_world_m(wrist_to_grasp_center_local, total_yaw)
0580:         grasp_center_to_rubber_world = rotate_topdown_offset_to_world_m(grasp_center_to_rubber_local, total_yaw)
0581:         wrist_model_error_world = rotate_topdown_offset_to_world_m(wrist_model_error_local, total_yaw)
0582:         desired_grasp_center_xyz = desired_rubber_xyz - grasp_center_to_rubber_world
0583:         desired_wrist_yaw_pos = (
0584:             desired_grasp_center_xyz
0585:             - base_world_translation
0586:             - wrist_to_grasp_center_world
0587:             - wrist_model_error_world
0588:         )
0589: 
0590:         pregrasp_clearance_m = self._geometric_topdown_pregrasp_clearance_m(self.grasp_config)
0591:         pregrasp_rubber_xyz = desired_rubber_xyz.copy()
0592:         pregrasp_rubber_xyz[2] += pregrasp_clearance_m
0593:         pregrasp_grasp_center_xyz = pregrasp_rubber_xyz.copy()
0594:         pregrasp_ik = None
0595:         pregrasp_lift = min(
0596:             STRETCH3_JOINT_LIMITS["lift"][1],
0597:             lift_val + pregrasp_clearance_m,
0598:         )
0599:         pregrasp_arm = arm_val
0600:         pregrasp_wrist_pos = np.zeros(3, dtype=float)
0601:         pregrasp_error_local = wrist_model_error_local.copy()
0602:         pregrasp_base_rotate_guess = base_rotate
0603:         for _ in range(5):
0604:             pregrasp_total_yaw = pregrasp_base_rotate_guess + desired_wrist_yaw
0605:             pregrasp_wrist_to_grasp_center_world = rotate_topdown_offset_to_world_m(
0606:                 wrist_to_grasp_center_local,
0607:                 pregrasp_total_yaw,
0608:             )
0609:             pregrasp_grasp_center_to_rubber_world = rotate_topdown_offset_to_world_m(
0610:                 grasp_center_to_rubber_local,
0611:                 pregrasp_total_yaw,
0612:             )
0613:             pregrasp_error_world = rotate_topdown_offset_to_world_m(pregrasp_error_local, pregrasp_total_yaw)
0614:             pregrasp_grasp_center_xyz = pregrasp_rubber_xyz - pregrasp_grasp_center_to_rubber_world
0615:             pregrasp_wrist_pos = (
0616:                 pregrasp_grasp_center_xyz
0617:                 - base_world_translation
0618:                 - pregrasp_wrist_to_grasp_center_world
0619:                 - pregrasp_error_world
0620:             )
0621:             pregrasp_ik = self.simple_ik.ik_rotary_base(pregrasp_wrist_pos.tolist())
0622:             if pregrasp_ik is None:
0623:                 break
0624:             self.simple_ik.clip_with_joint_limits(pregrasp_ik)
0625:             pregrasp_lift = float(pregrasp_ik["joint_lift"])
0626:             pregrasp_arm = float(pregrasp_ik["joint_arm_l0"])
0627:             pregrasp_base_rotate = float(pregrasp_ik["joint_mobile_base_rotation"])
0628:             next_pregrasp_error_local = np.asarray(
0629:                 topdown_simpleik_wrist_model_error_m(pregrasp_lift, pregrasp_arm),
0630:                 dtype=float,
0631:             )
0632:             if (
0633:                 abs(self._normalize_angle(pregrasp_base_rotate - pregrasp_base_rotate_guess)) < 1e-5
0634:                 and np.linalg.norm(next_pregrasp_error_local - pregrasp_error_local) < 1e-6
0635:             ):
0636:                 pregrasp_error_local = next_pregrasp_error_local
0637:                 break
0638:             pregrasp_error_local = next_pregrasp_error_local
0639:             pregrasp_base_rotate_guess = pregrasp_base_rotate
0640:         if pregrasp_ik is not None:
0641:             self.simple_ik.clip_with_joint_limits(pregrasp_ik)
0642:             pregrasp_lift = float(pregrasp_ik["joint_lift"])
0643:             pregrasp_arm = float(pregrasp_ik["joint_arm_l0"])
0644: 
0645:         fk_check = np.asarray(
0646:             self.simple_ik.fk_rotary_base(
0647:                 {
0648:                     "joint_mobile_base_rotation": base_rotate,
0649:                     "joint_lift": lift_val,
0650:                     "joint_arm_l0": arm_val,
```

Tall-object top-down height is capped by the top-delta rule:

```text
grasp_z = max(center_z, object_top_z - max_top_grasp_delta_m)
```

Utensil/fork/spoon behavior depends on whether the geometric grasp is classified as slender or the requested open width is below the wrist-yaw threshold. In that case the planner tries to align the wrist yaw to the geometric grip angle instead of using a cup-like default.

## 9. Robot Hardware, Platform, and Runtime Parameters

A separate paper-ready platform summary is available at:

```text
outputs/research_reports/robot_hardware_platform_and_parameters.md
```

The online experiments used a Stretch SE3 / Stretch 3-class mobile manipulator with a head-mounted Intel RealSense D435i RGB-D camera. The robot-side service ran a ZeroMQ transport server on port 5557 and executed only waypoint trajectories produced by the A6000-side planner.

Key runtime configuration:

| Area | Configuration |
| --- | --- |
| A6000 web service | FastAPI on `0.0.0.0:7862`, launched through `services/a6000_web/run_real_service.sh`. |
| VLM endpoint | OpenAI-compatible local endpoint `http://127.0.0.1:8000/v1`, model id `mimo-vl`. |
| Detector/SAM GPU | `cuda:1` for GroundingDINO and SAM in the recorded real-runtime env. |
| Robot transport | A6000 connects to `tcp://stretch-se3-3056.local:5557`; robot binds `tcp://0.0.0.0:5557`. |
| Camera | RealSense D435i, 1280 x 720 RGB-D capture, 15 fps capture configuration, 10 fps video preview. |
| Default head pose | pan `-1.57 rad`, tilt `-0.68 rad`, with online UI override support. |
| Real grasp mode | `ASK2ACT_PIPELINE_MODE=real_pointcloud`; SAM mask enabled with bbox fallback logic. |
| Execution safety | `ASK2ACT_AUTO_EXECUTE_ON_RESOLVE=0`; operator target confirmation required before physical grasp. |
| Online max rounds | `ASK2ACT_ONLINE_MAX_ROUNDS=6`. |

The grasp planner uses Stretch joint limits and gripper geometry from `simulation/ask2act_grasp/stretch3_specs.py`, SimpleIK top-down solving, calibrated rubber-contact offsets, side-aware target biases, slender-object wrist-yaw alignment, and optional base preposition/reobserve/replan behavior.

## 10. End-to-End System Summary

The whole system is a target-resolution-first robot pipeline:

```text
camera/RGB-D observation
  -> optional CP-calibrated / thresholded GroundingDINO candidate set
  -> annotated candidate overlay
  -> VLM JSON protocol for direct selection or clarification questions
  -> backend candidate-state update and EFE-style question ranking
  -> operator answers / target confirmation
  -> skip execution if target is wrong
  -> SAM/bbox point-cloud crop for confirmed target
  -> geometric grasp estimate
  -> Stretch top-down motion plan
  -> physical grasp execution
  -> operator grasp evaluation
```

The offline experiment isolates target-resolution accuracy and question efficiency. The online experiment measures whether those target-resolution gains transfer to full physical correct-object grasp success while protecting the robot from unnecessary wrong-target grasps.
