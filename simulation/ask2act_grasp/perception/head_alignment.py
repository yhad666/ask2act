from __future__ import annotations

import time

import numpy as np
from stretch_mujoco.enums.actuators import Actuators

from ask2act_grasp.types import HeadAlignmentConfig, HeadObservation
from ask2act_grasp.utils.tf_utils import make_transform


HEAD_CAMERA_LINK_FALLBACKS = ("realsense", "link_head_tilt")
HEAD_CAMERA_LOCAL_TRANSFORM = make_transform((0.0, 0.015, 0.0), (1.57, -1.57, 0.0))


class HeadAligner:
    def __init__(self, head_config: HeadAlignmentConfig) -> None:
        self.head_config = head_config

    def _resolve_head_camera_pose(self, sim) -> tuple[np.ndarray, str]:
        for link_name in HEAD_CAMERA_LINK_FALLBACKS:
            try:
                return np.asarray(sim.get_link_pose(link_name), dtype=float) @ HEAD_CAMERA_LOCAL_TRANSFORM, link_name
            except Exception:
                continue
        return np.eye(4, dtype=float), "identity_fallback"

    def align_and_capture(self, sim) -> HeadObservation:
        sim.move_to(Actuators.head_pan, float(self.head_config.head_pan_rad))
        sim.wait_until_at_setpoint(Actuators.head_pan, timeout=30.0)
        sim.move_to(Actuators.head_tilt, float(self.head_config.head_tilt_rad))
        sim.wait_until_at_setpoint(Actuators.head_tilt, timeout=30.0)
        time.sleep(float(self.head_config.settle_seconds))

        camera_status = sim.pull_camera_data()
        rgb = np.asarray(camera_status.cam_d435i_rgb, dtype=np.uint8)
        depth = np.asarray(camera_status.cam_d435i_depth, dtype=float)
        k_matrix = np.asarray(camera_status.cam_d435i_K, dtype=float).reshape(3, 3)
        camera_pose, source = self._resolve_head_camera_pose(sim)
        return HeadObservation(
            rgb_image=rgb,
            depth_image=depth,
            camera_intrinsics=k_matrix,
            camera_extrinsics=camera_pose,
            camera_source=source,
            capture_time_s=float(camera_status.time),
        )
