from __future__ import annotations

import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.a6000_web import server as web_server
from services.a6000_web.grasp_runtime import LocalGraspRuntime
from services.a6000_web.offline_experiments import OfflineExperimentStore
from services.a6000_web.schemas import Candidate, ResolvedTarget, SessionState


def _session(*, session_id: str = "session-online", resolved_candidate_id: str = "candidate_1") -> SessionState:
    candidates = [
        Candidate(candidate_id="candidate_1", display_id=1, label="cup", score=0.9, bbox_xyxy=[10, 10, 40, 40]),
        Candidate(candidate_id="candidate_2", display_id=2, label="cup", score=0.8, bbox_xyxy=[50, 10, 80, 40]),
    ]
    resolved = next(item for item in candidates if item.candidate_id == resolved_candidate_id)
    return SessionState(
        session_id=session_id,
        instruction="pick up my cup",
        instruction_phrases=["cup"],
        detection_prompt="cup.",
        observation_id="obs-online",
        observation_source="test",
        observation_image_bytes=b"test-image",
        observation_image_data_url="data:image/png;base64,dGVzdA==",
        candidate_overlay_data_url="data:image/png;base64,dGVzdA==",
        candidates=candidates,
        vlm_messages=[],
        resolved_target=ResolvedTarget(**resolved.model_dump()),
        status="resolved",
    )


def _install_online_trial(tmp_path, monkeypatch, *, resolved_candidate_id: str = "candidate_1") -> tuple[TestClient, str, str]:
    store = OfflineExperimentStore(tmp_path)
    monkeypatch.setattr(web_server, "online_store", store)
    web_server.SESSIONS.clear()
    web_server.ONLINE_TRIAL_SESSIONS.clear()
    web_server.ONLINE_TRIAL_LOCKS.clear()

    experiment_id = "online-flow"
    trial_id = "trial-online-flow"
    session_id = "session-online-flow"
    store.create_experiment(experiment_id=experiment_id, name="online flow", experiment_type="online_pilot")
    store.save_scene_metadata(
        experiment_id=experiment_id,
        scene_id="scene_001",
        scene_type="pilot",
        object_categories=["cup"],
        notes="",
    )
    store.write_trial(
        experiment_id,
        {
            "trial_id": trial_id,
            "scene_id": "scene_001",
            "scene_type": "pilot",
            "prompt": "pick up my cup",
            "prompt_type": "ambiguous",
            "method": "proposed_efe",
            "session_id": session_id,
            "started_at_epoch_s": time.time(),
            "grasp_attempted": False,
            "wrong_target_grasp_prevented": False,
        },
    )
    web_server.SESSIONS[session_id] = _session(session_id=session_id, resolved_candidate_id=resolved_candidate_id)
    web_server.ONLINE_TRIAL_SESSIONS[trial_id] = session_id
    return TestClient(web_server.app), experiment_id, trial_id


def test_online_execute_requires_expected_target(tmp_path, monkeypatch):
    client, experiment_id, trial_id = _install_online_trial(tmp_path, monkeypatch)

    response = client.post(f"/api/online/experiments/{experiment_id}/trials/{trial_id}/execute", json={})

    assert response.status_code == 400
    assert "expected" in response.json()["detail"]
    trial = web_server.online_store.read_trial(experiment_id, trial_id)
    assert not trial.get("grasp_attempted")


def test_online_start_ignores_expected_target_until_execute(tmp_path, monkeypatch):
    store = OfflineExperimentStore(tmp_path)
    monkeypatch.setattr(web_server, "online_store", store)
    web_server.SESSIONS.clear()
    web_server.ONLINE_TRIAL_SESSIONS.clear()
    web_server.ONLINE_TRIAL_LOCKS.clear()
    store.create_experiment(experiment_id="online-no-leak", name="online no leak", experiment_type="online_pilot")
    store.save_scene_metadata(
        experiment_id="online-no-leak",
        scene_id="scene_001",
        scene_type="pilot",
        object_categories=["cup"],
        notes="",
    )

    def fake_create_session(*args, **kwargs):
        return _session(session_id="session-no-leak")

    def noop_method(*args, **kwargs):
        return None

    monkeypatch.setattr(web_server, "create_session", fake_create_session)
    monkeypatch.setattr(web_server, "_apply_offline_trial_method", noop_method)
    client = TestClient(web_server.app)

    response = client.post(
        "/api/online/experiments/online-no-leak/trials/start",
        json={
            "scene_id": "scene_001",
            "prompt": "pick up my cup",
            "prompt_type": "ambiguous",
            "method": "proposed_efe",
            "expected_display_id": 1,
            "expected_candidate_id": "candidate_1",
        },
    )

    assert response.status_code == 200
    trial = response.json()["trial"]
    assert trial.get("expected_display_id") is None
    assert trial.get("expected_candidate_id") is None


def test_online_wrong_target_is_skipped_without_robot_execution(tmp_path, monkeypatch):
    client, experiment_id, trial_id = _install_online_trial(tmp_path, monkeypatch, resolved_candidate_id="candidate_2")
    called = {"execute": False}

    def fail_if_called(*args, **kwargs):
        called["execute"] = True
        raise AssertionError("robot execution should not be called for a wrong target")

    monkeypatch.setattr(web_server, "execute_resolved_session", fail_if_called)

    response = client.post(
        f"/api/online/experiments/{experiment_id}/trials/{trial_id}/execute",
        json={"expected_display_id": 1},
    )

    assert response.status_code == 200
    trial = response.json()["trial"]
    assert trial["outcome"] == "skipped_wrong_target"
    assert trial["wrong_target_grasp_prevented"] is True
    assert trial["grasp_attempted"] is False
    assert called["execute"] is False


def test_online_operator_can_mark_wrong_target_without_robot_execution(tmp_path, monkeypatch):
    client, experiment_id, trial_id = _install_online_trial(tmp_path, monkeypatch)

    response = client.post(
        f"/api/online/experiments/{experiment_id}/trials/{trial_id}/finish",
        json={"outcome": "skipped_wrong_target", "note": "operator says target is wrong"},
    )

    assert response.status_code == 200
    trial = web_server.online_store.read_trial(experiment_id, trial_id)
    assert trial["status"] == "finished"
    assert trial["outcome"] == "skipped_wrong_target"
    assert trial["target_selection_outcome"] == "wrong"
    assert trial["target_selection_correct"] is False
    assert trial["wrong_target_grasp_prevented"] is True
    assert trial["grasp_attempted"] is False


def test_online_execution_failure_is_not_marked_executed(tmp_path, monkeypatch):
    client, experiment_id, trial_id = _install_online_trial(tmp_path, monkeypatch)

    def mark_failed(session, *, dry_run, raise_on_error=True):
        session.status = "execution_failed"
        session.execution_result = {"ok": False, "dry_run": dry_run, "error": "simulated stretch failure"}

    monkeypatch.setattr(web_server, "execute_resolved_session", mark_failed)

    response = client.post(
        f"/api/online/experiments/{experiment_id}/trials/{trial_id}/execute",
        json={"expected_display_id": 1},
    )

    assert response.status_code == 200
    trial = response.json()["trial"]
    assert trial["status"] == "finished"
    assert trial["outcome"] == "execution_failed"
    assert trial["grasp_execution_ok"] is False
    assert trial["grasp_attempted"] is True


def test_online_head_camera_video_is_saved_on_a6000(tmp_path, monkeypatch):
    store = OfflineExperimentStore(tmp_path)
    monkeypatch.setattr(web_server, "online_store", store)
    monkeypatch.setattr(web_server, "ONLINE_EXPERIMENT_ROOT", tmp_path)
    store.create_experiment(experiment_id="online-video", name="online video", experiment_type="online_pilot")

    class FakeStretchTransport:
        mode = "zmq"

        def start_head_camera_video(self, *, experiment_id, video_id=None):
            return {
                "ok": True,
                "status": "recording",
                "video_id": video_id,
                "video_width": 1280,
                "video_height": 720,
                "fps": 10,
            }

        def stop_head_camera_video(self, *, experiment_id):
            payload = b"fake-mp4-bytes"
            return {
                "ok": True,
                "status": "stopped",
                "video_id": "fake_video",
                "filename": "fake_video.mp4",
                "mime_type": "video/mp4",
                "size_bytes": len(payload),
                "video_base64": base64.b64encode(payload).decode("ascii"),
                "metadata": {"frame_count": 3},
                "mode": "zmq",
            }

        def head_camera_video_status(self):
            return {"ok": True, "running": False, "status": "idle"}

    monkeypatch.setattr(web_server, "stretch_transport", FakeStretchTransport())
    client = TestClient(web_server.app)

    start = client.post("/api/online/experiments/online-video/video/start", json={})
    assert start.status_code == 200
    assert start.json()["video"]["video_width"] == 1280

    stop = client.post("/api/online/experiments/online-video/video/stop", json={})
    assert stop.status_code == 200
    video = stop.json()["video"]
    assert video["filename"] == "fake_video.mp4"
    video_path = tmp_path / "online-video" / "videos" / "fake_video.mp4"
    assert video_path.read_bytes() == b"fake-mp4-bytes"

    download = client.get(video["video_url"])
    assert download.status_code == 200
    assert download.content == b"fake-mp4-bytes"


def test_online_head_pose_command_is_forwarded(monkeypatch):
    calls = []

    class FakeStretchTransport:
        def set_head_camera_pose(self, *, head_pan_rad, head_tilt_rad, persist=True):
            calls.append((head_pan_rad, head_tilt_rad, persist))
            return {
                "ok": True,
                "commanded_head_pan_rad": head_pan_rad,
                "commanded_head_tilt_rad": head_tilt_rad,
                "actual_head_pan_rad": head_pan_rad,
                "actual_head_tilt_rad": head_tilt_rad,
                "persisted_for_future_observations": persist,
            }

    monkeypatch.setattr(web_server, "stretch_transport", FakeStretchTransport())
    client = TestClient(web_server.app)

    response = client.post(
        "/api/online/head_pose",
        json={"head_pan_rad": -1.5, "head_tilt_rad": -0.62, "persist": True},
    )

    assert response.status_code == 200
    assert calls == [(-1.5, -0.62, True)]
    assert response.json()["head_pose"]["actual_head_tilt_rad"] == -0.62


def test_sam_mask_predictor_uses_box_prompt_and_clips_to_bbox(monkeypatch):
    import numpy as np

    class FakePredictor:
        def __init__(self) -> None:
            self.image_shape = None
            self.box = None

        def set_image(self, image_np):
            self.image_shape = image_np.shape[:2]

        def predict(self, *, box, multimask_output):
            self.box = box.tolist()
            mask = np.zeros(self.image_shape, dtype=bool)
            mask[1:5, 1:5] = True
            mask[0, 0] = True
            return np.asarray([mask]), np.asarray([0.93], dtype=float), None

    fake = FakePredictor()
    runtime = LocalGraspRuntime(mode="real_pointcloud")
    monkeypatch.setenv("ASK2ACT_SAM_BOX_CLIP_EXPAND_PX", "0")
    monkeypatch.setenv("ASK2ACT_SAM_MIN_MASK_PIXELS", "1")
    monkeypatch.setattr(
        runtime,
        "_load_sam_predictor",
        lambda: (
            fake,
            {
                "enabled": True,
                "used": True,
                "source": "segment_anything",
                "model_type": "vit_b",
                "checkpoint": "fake.pth",
                "device": "cpu",
            },
        ),
    )

    result = runtime._predict_sam_mask_for_bbox(
        rgb_image=Image.new("RGB", (8, 8), color=(20, 20, 20)),
        depth_shape_hw=(8, 8),
        bbox_xyxy=(1, 1, 5, 5),
    )

    assert result["used"] is True
    assert result["score"] == pytest.approx(0.93)
    assert fake.box == [1.0, 1.0, 5.0, 5.0]
    assert result["mask"][0, 0] == np.bool_(False)
    assert result["mask_pixels"] == 16


def test_sam_mask_predictor_falls_back_when_checkpoint_missing(monkeypatch):
    runtime = LocalGraspRuntime(mode="real_pointcloud")
    monkeypatch.setenv("ASK2ACT_REAL_USE_SAM_MASK", "1")
    monkeypatch.delenv("ASK2ACT_SAM_CHECKPOINT", raising=False)

    result = runtime._predict_sam_mask_for_bbox(
        rgb_image=Image.new("RGB", (8, 8), color=(20, 20, 20)),
        depth_shape_hw=(8, 8),
        bbox_xyxy=(1, 1, 5, 5),
    )

    assert result["used"] is False
    assert result["source"] == "missing_checkpoint"


def test_robot_head_pose_override_drives_future_observations(tmp_path, monkeypatch):
    from real.stretch_transport.scripts import capture_observation

    override_path = tmp_path / "head_pose_override.json"
    override_path.write_text(
        json.dumps(
            {
                "head_pan_rad": -1.42,
                "head_tilt_rad": -0.51,
                "source": "manual_ui",
                "saved_at_epoch_s": 123.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ASK2ACT_STRETCH_HEAD_POSE_OVERRIDE_PATH", str(override_path))
    monkeypatch.setenv("ASK2ACT_STRETCH_HEAD_POSE_STAMP_PATH", str(tmp_path / "stamp.json"))
    monkeypatch.setenv("ASK2ACT_STRETCH_INIT_HEAD_POSE_MODE", "once_per_server")
    monkeypatch.setenv("ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD", "-1.57")
    monkeypatch.setenv("ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD", "-0.68")
    calls = []

    def fake_command_head_pose(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "commanded_head_pan_rad": kwargs["head_pan"],
            "commanded_head_tilt_rad": kwargs["head_tilt"],
            "actual_head_pan_rad": kwargs["head_pan"],
            "actual_head_tilt_rad": kwargs["head_tilt"],
            "pose_source": kwargs.get("pose_source"),
            "camera_extrinsics": capture_observation._camera_extrinsics_payload(
                kwargs["head_pan"], kwargs["head_tilt"]
            ),
        }

    monkeypatch.setattr(capture_observation, "_command_head_pose", fake_command_head_pose)

    result = capture_observation._ensure_initial_head_pose()

    assert calls
    assert calls[0]["head_pan"] == -1.42
    assert calls[0]["head_tilt"] == -0.51
    assert calls[0]["mode"] == "manual_override"
    assert calls[0]["pose_source"]["source"] == "manual_override"
    assert result["camera_extrinsics"]["head_pan_rad"] == -1.42
    assert result["camera_extrinsics"]["head_tilt_rad"] == -0.51


def test_camera_extrinsics_follow_head_angle():
    from real.stretch_transport.scripts import capture_observation

    left = capture_observation._camera_extrinsics_payload(-1.42, -0.51)
    default = capture_observation._camera_extrinsics_payload(-1.57, -0.68)

    assert left["head_pan_rad"] == -1.42
    assert left["head_tilt_rad"] == -0.51
    assert left["camera_to_world"] != default["camera_to_world"]


def test_camera_safe_observe_verify_uses_clearance_bounds(monkeypatch):
    from real.stretch_transport.scripts import capture_observation

    readings = {"lift": 0.50, "arm": 0.02}

    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_JOINTS", "lift,arm")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_TIMEOUT_S", "0.1")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_LIFT", "0.035")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_ARM", "0.025")
    monkeypatch.setattr(capture_observation, "_read_joint_position", lambda _robot, joint: readings[joint])

    result = capture_observation._verify_default_pose_for_camera(
        object(),
        {"lift": 0.45, "arm": 0.0},
    )

    assert result["ok"] is True
    assert result["pending_joints"] == []
    assert result["joints"]["lift"]["mode"] == "min"
    assert result["joints"]["arm"]["mode"] == "max"


def test_camera_safe_observe_verify_allows_soft_arm_clearance(monkeypatch):
    from real.stretch_transport.scripts import capture_observation

    readings = {"lift": 0.50, "arm": 0.055}

    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_JOINTS", "lift,arm")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_TIMEOUT_S", "0.1")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_LIFT", "0.035")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_ARM", "0.025")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_ARM_SOFT_MAX_M", "0.08")
    monkeypatch.setattr(capture_observation, "_read_joint_position", lambda _robot, joint: readings[joint])

    result = capture_observation._verify_default_pose_for_camera(
        object(),
        {"lift": 0.45, "arm": 0.0},
    )

    assert result["ok"] is True
    assert result["pending_joints"] == []
    assert result["joints"]["arm"]["mode"] == "soft_max"


def test_camera_safe_observe_verify_rejects_low_lift(monkeypatch):
    from real.stretch_transport.scripts import capture_observation

    readings = {"lift": 0.40, "arm": 0.0}

    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_JOINTS", "lift,arm")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_TIMEOUT_S", "0.1")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_LIFT", "0.035")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_ARM", "0.025")
    monkeypatch.setattr(capture_observation, "_read_joint_position", lambda _robot, joint: readings[joint])

    result = capture_observation._verify_default_pose_for_camera(
        object(),
        {"lift": 0.45, "arm": 0.0},
    )

    assert result["ok"] is False
    assert result["pending_joints"] == ["lift"]


def test_observe_default_pose_retries_arm_retract_after_soft_pass(monkeypatch):
    from real.stretch_transport.scripts import capture_observation

    class FakeJoint:
        def __init__(self, status=None):
            self.status = status or {}
            self.moves = []

        def move_to(self, *args):
            self.moves.append(args)

    class FakeRobot:
        def __init__(self):
            self.arm = FakeJoint({"pos": 0.055})
            self.lift = FakeJoint({"pos": 0.50})
            self.end_of_arm = FakeJoint({})
            self.base = FakeJoint({})
            self.head = FakeJoint({})
            self.push_count = 0

        def push_command(self):
            self.push_count += 1

        def pull_status(self):
            return None

    robot = FakeRobot()
    arm_readings = [0.055, 0.010]

    def fake_read(_robot, joint):
        if joint == "lift":
            return 0.50
        if joint == "arm":
            return arm_readings.pop(0) if arm_readings else 0.010
        return 0.0

    monkeypatch.setattr(capture_observation, "_read_joint_position", fake_read)
    monkeypatch.setattr(capture_observation.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_LIFT_OBSERVE_START_M", "0.45")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_SETTLE_OBSERVE_START_S", "0")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_ARM_RETRY", "1")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_ARM_RETRY_SETTLE_S", "0")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_TIMEOUT_S", "0.1")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_JOINTS", "lift,arm")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_ARM", "0.025")
    monkeypatch.setenv("ASK2ACT_STRETCH_HOME_VERIFY_ARM_SOFT_MAX_M", "0.08")

    result = capture_observation._command_default_pose(
        robot,
        include_gripper=False,
        reason="observe_start",
    )

    assert result["ok"] is True
    assert result["arm_retry"]["joint"] == "arm"
    assert result["verify"]["initial_verify"]["joints"]["arm"]["mode"] == "soft_max"
    assert result["verify"]["joints"]["arm"]["mode"] == "max"
    assert len(robot.arm.moves) == 2


def test_stretch_close_gripper_wait_is_nonblocking_by_default(monkeypatch):
    from real.stretch_transport.scripts import dispatch_grasp

    def fail_read(*_args, **_kwargs):
        raise AssertionError("close wait should not require reading final gripper closure")

    monkeypatch.setenv("ASK2ACT_STRETCH_GRIPPER_COMMAND_MODE", "real_pct")
    monkeypatch.setenv("ASK2ACT_STRETCH_GRIPPER_REAL_CLOSE_CMD", "-80.0")
    monkeypatch.setenv("ASK2ACT_STRETCH_GRIPPER_CLOSE_VERIFY", "0")
    monkeypatch.setenv("ASK2ACT_STRETCH_GRIPPER_CLOSE_SETTLE_S", "0.0")
    monkeypatch.setattr(dispatch_grasp, "_read_joint_position", fail_read)

    result = dispatch_grasp._wait_for_joint_target(
        object(),
        joint_name="stretch_gripper",
        target=-80.0,
        waypoint_name="close_gripper",
    )

    assert result["ok"] is True
    assert result["close_verify"] is False


def test_online_grasp_tuning_updates_planner_and_robot_runtime(monkeypatch):
    web_server.grasp_runtime._ensure_grasp_import_path()
    from ask2act_grasp.planning import motion_planner as mp

    original_rubber_y = mp.GEOMETRIC_TOP_DOWN_RUBBER_LOCAL_Y_CORRECTION_M
    original_open = mp.GEOMETRIC_TOP_DOWN_GRIPPER_OPEN_CMD_OVERRIDE
    original_side_x = mp.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_M
    original_side_deadband = mp.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_DEADBAND_M
    original_right_extra = mp.GEOMETRIC_TOP_DOWN_RIGHT_EXTRA_X_BIAS_M
    original_left_center_y = mp.GEOMETRIC_TOP_DOWN_LEFT_CENTER_Y_BIAS_M
    original_right_y = mp.GEOMETRIC_TOP_DOWN_RIGHT_Y_BIAS_M
    original_slender_rubber_y = mp.GEOMETRIC_TOP_DOWN_SLENDER_RUBBER_LOCAL_Y_CORRECTION_M
    original_slender_side_x = mp.GEOMETRIC_TOP_DOWN_SLENDER_SIDE_X_BIAS_M
    original_slender_right_extra = mp.GEOMETRIC_TOP_DOWN_SLENDER_RIGHT_EXTRA_X_BIAS_M
    original_slender_left_center_y = mp.GEOMETRIC_TOP_DOWN_SLENDER_LEFT_CENTER_Y_BIAS_M
    original_slender_right_y = mp.GEOMETRIC_TOP_DOWN_SLENDER_RIGHT_Y_BIAS_M
    original_max_top_delta = mp.GEOMETRIC_TOP_DOWN_MAX_TOP_GRASP_DELTA_M
    calls = []

    class FakeStretchTransport:
        def set_runtime_config(self, *, env):
            calls.append(env)
            return {"ok": True, "applied_env": {str(key): str(value) for key, value in env.items()}}

    monkeypatch.setattr(web_server, "stretch_transport", FakeStretchTransport())
    client = TestClient(web_server.app)

    try:
        response = client.post(
            "/api/online/grasp_tuning",
            json={
                "rubber_local_y_correction_m": -0.03,
                "side_x_bias_m": 0.02,
                "side_x_bias_deadband_m": 0.05,
                "right_extra_x_bias_m": 0.01,
                "left_center_y_bias_m": 0.005,
                "right_y_bias_m": -0.005,
                "slender_rubber_local_y_correction_m": -0.012,
                "slender_side_x_bias_m": 0.015,
                "slender_right_extra_x_bias_m": 0.006,
                "slender_left_center_y_bias_m": 0.002,
                "slender_right_y_bias_m": -0.002,
                "max_top_grasp_delta_m": 0.07,
                "gripper_open_cmd_override": 0.58,
                "stretch_gripper_real_open_cmd": 100.0,
                "stretch_gripper_real_close_cmd": -80.0,
                "stretch_release_gripper_cmd": 100.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tuning"]["rubber_local_y_correction_m"] == -0.03
        assert data["tuning"]["side_x_bias_m"] == 0.02
        assert data["tuning"]["side_x_bias_deadband_m"] == 0.05
        assert data["tuning"]["right_extra_x_bias_m"] == 0.01
        assert data["tuning"]["left_center_y_bias_m"] == 0.005
        assert data["tuning"]["right_y_bias_m"] == -0.005
        assert data["tuning"]["slender_rubber_local_y_correction_m"] == -0.012
        assert data["tuning"]["slender_side_x_bias_m"] == 0.015
        assert data["tuning"]["slender_right_extra_x_bias_m"] == 0.006
        assert data["tuning"]["slender_left_center_y_bias_m"] == 0.002
        assert data["tuning"]["slender_right_y_bias_m"] == -0.002
        assert data["tuning"]["max_top_grasp_delta_m"] == 0.07
        assert data["tuning"]["gripper_open_cmd_override"] == 0.58
        assert data["tuning"]["stretch_gripper_real_close_cmd"] == -80.0
        assert mp.GEOMETRIC_TOP_DOWN_RUBBER_LOCAL_Y_CORRECTION_M == -0.03
        assert mp.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_M == 0.02
        assert mp.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_DEADBAND_M == 0.05
        assert mp.GEOMETRIC_TOP_DOWN_RIGHT_EXTRA_X_BIAS_M == 0.01
        assert mp.GEOMETRIC_TOP_DOWN_LEFT_CENTER_Y_BIAS_M == 0.005
        assert mp.GEOMETRIC_TOP_DOWN_RIGHT_Y_BIAS_M == -0.005
        assert mp.GEOMETRIC_TOP_DOWN_SLENDER_RUBBER_LOCAL_Y_CORRECTION_M == "-0.012"
        assert mp.GEOMETRIC_TOP_DOWN_SLENDER_SIDE_X_BIAS_M == "0.015"
        assert mp.GEOMETRIC_TOP_DOWN_SLENDER_RIGHT_EXTRA_X_BIAS_M == "0.006"
        assert mp.GEOMETRIC_TOP_DOWN_SLENDER_LEFT_CENTER_Y_BIAS_M == "0.002"
        assert mp.GEOMETRIC_TOP_DOWN_SLENDER_RIGHT_Y_BIAS_M == "-0.002"
        assert mp.GEOMETRIC_TOP_DOWN_MAX_TOP_GRASP_DELTA_M == 0.07
        assert mp.GEOMETRIC_TOP_DOWN_GRIPPER_OPEN_CMD_OVERRIDE == "0.58"
        assert calls == [
            {
                "ASK2ACT_STRETCH_GRIPPER_REAL_OPEN_CMD": 100.0,
                "ASK2ACT_STRETCH_GRIPPER_REAL_CLOSE_CMD": -80.0,
                "ASK2ACT_STRETCH_RELEASE_GRIPPER_CMD": 100.0,
                "ASK2ACT_STRETCH_GRIPPER_CLOSE_VERIFY": 0,
            }
        ]
    finally:
        mp.GEOMETRIC_TOP_DOWN_RUBBER_LOCAL_Y_CORRECTION_M = original_rubber_y
        mp.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_M = original_side_x
        mp.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_DEADBAND_M = original_side_deadband
        mp.GEOMETRIC_TOP_DOWN_RIGHT_EXTRA_X_BIAS_M = original_right_extra
        mp.GEOMETRIC_TOP_DOWN_LEFT_CENTER_Y_BIAS_M = original_left_center_y
        mp.GEOMETRIC_TOP_DOWN_RIGHT_Y_BIAS_M = original_right_y
        mp.GEOMETRIC_TOP_DOWN_SLENDER_RUBBER_LOCAL_Y_CORRECTION_M = original_slender_rubber_y
        mp.GEOMETRIC_TOP_DOWN_SLENDER_SIDE_X_BIAS_M = original_slender_side_x
        mp.GEOMETRIC_TOP_DOWN_SLENDER_RIGHT_EXTRA_X_BIAS_M = original_slender_right_extra
        mp.GEOMETRIC_TOP_DOWN_SLENDER_LEFT_CENTER_Y_BIAS_M = original_slender_left_center_y
        mp.GEOMETRIC_TOP_DOWN_SLENDER_RIGHT_Y_BIAS_M = original_slender_right_y
        mp.GEOMETRIC_TOP_DOWN_MAX_TOP_GRASP_DELTA_M = original_max_top_delta
        mp.GEOMETRIC_TOP_DOWN_GRIPPER_OPEN_CMD_OVERRIDE = original_open


def test_oversized_base_reach_refuses_before_robot_motion(monkeypatch):
    session = _session()
    session.observation_raw_response = {"depth_aligned_to_color": True}
    called = {"dispatch": False}

    class FakeGraspRuntime:
        mode = "real_pointcloud"

        def plan_for_target(self, *args, **kwargs):
            return {
                "ok": True,
                "dispatch_payload": {
                    "trajectory": [
                        {
                            "name": "base_translate_for_reach",
                            "joint_targets": {"base_translate_arm_axis": 0.06},
                            "settle_s": 0.4,
                        }
                    ],
                    "motion_plan_metadata": {
                        "base_preposition": {
                            "reach_error_m": 0.26,
                            "longitudinal_deadband_m": 0.04,
                            "requested_base_translate_arm_axis_m": 0.06,
                        }
                    },
                },
                "plan_summary": {
                    "pipeline_mode": "real_pointcloud",
                    "planner_backend": "real_base_reach_preposition_required",
                    "target_bbox_xyxy": [10, 10, 40, 40],
                    "point_cloud_count": 100,
                    "selected_grasp_score": 0.9,
                    "trajectory_waypoint_count": 1,
                    "pipeline_run_dir": "",
                    "success": True,
                    "note": "base preposition only",
                },
            }

    class FakeStretchTransport:
        def dispatch_grasp(self, *args, **kwargs):
            called["dispatch"] = True
            raise AssertionError("base should not move for oversized preposition")

    monkeypatch.setattr(web_server, "grasp_runtime", FakeGraspRuntime())
    monkeypatch.setattr(web_server, "stretch_transport", FakeStretchTransport())
    monkeypatch.setattr(web_server, "REAL_REPLAN_AFTER_BASE_REACH", True)
    monkeypatch.setattr(web_server, "REAL_BASE_REACH_REPLAN_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(web_server, "REAL_BASE_REACH_FAIL_FAST_OVERSIZED", True)
    monkeypatch.setattr(web_server, "finalize_resolved_target", lambda *args, **kwargs: None)
    monkeypatch.setenv("ASK2ACT_REAL_BASE_REACH_MAX_TOTAL_ARM_AXIS_M", "0.15")

    with pytest.raises(Exception) as exc_info:
        web_server.execute_resolved_session(session, dry_run=False, raise_on_error=True)

    assert "exceeding the configured safe cumulative preposition budget" in str(exc_info.value)
    assert called["dispatch"] is False
