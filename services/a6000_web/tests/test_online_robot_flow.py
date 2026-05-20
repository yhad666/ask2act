from __future__ import annotations

import base64
import time

import pytest
from fastapi.testclient import TestClient

from services.a6000_web import server as web_server
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
