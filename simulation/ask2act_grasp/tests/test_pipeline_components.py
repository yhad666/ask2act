from __future__ import annotations

from pathlib import Path

import numpy as np

from ask2act_grasp.grasp.grasp_generator import ContactGraspNetWrapper
from ask2act_grasp.grasp.grasp_selector import GraspSelector
from ask2act_grasp.perception.point_cloud_gen import PointCloudGenerator
from ask2act_grasp.scene.scene_setup import SceneSetup
from ask2act_grasp.types import GraspCandidate
from ask2act_grasp.utils.config_loader import load_grasp_config, load_scene_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_scene_setup_writes_runtime_xml(tmp_path):
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    scene_path = SceneSetup(scene_config).write_scene(tmp_path / "scene.xml")
    text = scene_path.read_text()
    assert "target_cup" in text
    assert "freejoint" in text


def test_point_cloud_generation_filters_below_table():
    generator = PointCloudGenerator()
    depth = np.array([[0.82, 0.84], [0.88, 0.0]], dtype=float)
    k = np.array([[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]])
    result = generator.generate(
        depth_image=depth,
        camera_intrinsics=k,
        camera_extrinsics=np.eye(4),
        table_top_z_m=0.81,
        table_margin_m=0.01,
        z_min_m=0.2,
        z_max_m=1.1,
    )
    assert result.filtered_point_count == 2


def test_fallback_grasp_generator_produces_candidates():
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.enable_contact_graspnet = False
    generator = ContactGraspNetWrapper(grasp_config)
    points = np.array(
        [
            [0.00, -0.34, 0.87],
            [0.01, -0.33, 0.86],
            [-0.01, -0.35, 0.88],
        ],
        dtype=float,
    )
    candidates = generator.generate(points)
    assert len(candidates) >= 1
    assert candidates[0].score >= 0.5


def test_grasp_selector_prefers_top_down_feasible_candidate():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    selector = GraspSelector(scene_config, grasp_config)
    good_pose = np.eye(4)
    good_pose[:3, 2] = np.array([0.0, 0.0, -1.0])
    good_pose[:3, 3] = np.array([0.0, -0.34, 0.88])
    bad_pose = np.eye(4)
    bad_pose[:3, 2] = np.array([0.0, 0.0, 1.0])
    bad_pose[:3, 3] = np.array([0.0, -0.34, 0.88])
    selected = selector.select_best(
        [
            GraspCandidate(good_pose, 0.7, 0.06, "test"),
            GraspCandidate(bad_pose, 0.9, 0.06, "test"),
        ],
        robot_state={"stretch_gripper": 0.04},
    )
    assert selected is not None
    assert selected.score == 0.7

