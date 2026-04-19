from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

from ask2act_grasp.types import HeadAlignmentConfig, HeadObservation
from ask2act_grasp.utils.head_camera_orientation import rotate_head_observation_clockwise
from ask2act_grasp.utils.tf_utils import make_transform


HEAD_CAMERA_BODY_NAME = "realsense"
HEAD_CAMERA_NAME = StretchCameras.cam_d435i_rgb.camera_name_in_mjcf
HEAD_CAMERA_LOCAL_TRANSFORM = make_transform((0.0, 0.015, 0.0), (1.57, -1.57, 0.0))
GL_TO_CV_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)


class HeadAligner:
    """Move the head to a fixed observation pose and capture RGB-D."""

    HEAD_TOLERANCE_RAD = 0.12
    HEAD_WAIT_TOLERANCE_RAD = 0.12
    MAX_CAPTURE_RETRIES = 12
    MAX_ALIGNMENT_ATTEMPTS = 5

    def __init__(self, head_config: HeadAlignmentConfig, scene_xml_path: Path | None = None) -> None:
        self.head_config = head_config
        self.scene_xml_path = Path(scene_xml_path) if scene_xml_path is not None else None

    def align_and_capture(self, sim, target_center_xy: np.ndarray | None = None) -> HeadObservation:
        """Command the head, wait for it to settle, then capture valid RGB-D."""
        camera_status = None
        last_settle_error: RuntimeError | None = None
        last_capture_error: RuntimeError | None = None
        commanded_pan, commanded_tilt = self._resolve_head_targets(sim, target_center_xy)
        for attempt in range(1, self.MAX_ALIGNMENT_ATTEMPTS + 1):
            self._command_head_pose(sim, head_pan_rad=commanded_pan, head_tilt_rad=commanded_tilt)
            settled_ok = False
            try:
                self._validate_head_settle(sim, head_pan_rad=commanded_pan, head_tilt_rad=commanded_tilt)
                last_settle_error = None
                settled_ok = True
            except RuntimeError as exc:
                last_settle_error = exc
                print(
                    f"Head alignment retry {attempt}/{self.MAX_ALIGNMENT_ATTEMPTS}: {exc}",
                    flush=True,
                )
            time.sleep(float(self.head_config.settle_seconds))

            if not settled_ok:
                continue

            try:
                camera_status = self._poll_valid_head_camera_data(sim)
                last_capture_error = None
                break
            except RuntimeError as exc:
                last_capture_error = exc
                print(
                    f"Head camera capture retry {attempt}/{self.MAX_ALIGNMENT_ATTEMPTS}: {exc}",
                    flush=True,
                )
                camera_status = None

        if camera_status is None:
            if last_settle_error is not None:
                raise last_settle_error
            if last_capture_error is not None:
                raise last_capture_error
            camera_status = self._poll_valid_head_camera_data(sim)

        rgb = np.asarray(camera_status.cam_d435i_rgb, dtype=np.uint8)
        depth = np.asarray(camera_status.cam_d435i_depth, dtype=np.float32)
        camera_intrinsics, intrinsics_source = self._resolve_head_camera_intrinsics(
            depth_shape=depth.shape,
            fallback_k=getattr(camera_status, "cam_d435i_K", None),
        )
        camera_extrinsics, extrinsics_source = self._resolve_head_camera_pose(sim)
        rgb, depth, camera_intrinsics, camera_extrinsics = rotate_head_observation_clockwise(
            rgb,
            depth,
            camera_intrinsics,
            camera_extrinsics,
        )
        return HeadObservation(
            rgb_image=rgb,
            depth_image=depth,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            camera_source=f"{intrinsics_source}|{extrinsics_source}|rotated_cw90",
            capture_time_s=float(camera_status.time),
        )

    def _command_head_pose(self, sim, *, head_pan_rad: float, head_tilt_rad: float) -> None:
        sim.move_to(Actuators.head_pan, float(head_pan_rad))
        if not sim.wait_until_at_setpoint(
            Actuators.head_pan,
            timeout=30.0,
            position_tolerance=self.HEAD_WAIT_TOLERANCE_RAD,
        ):
            actual_pan = float(sim.pull_status().head_pan.pos)
            print(
                f"Head pan tolerated for perception: command={float(head_pan_rad):.3f}, actual={actual_pan:.3f}",
                flush=True,
            )

        sim.move_to(Actuators.head_tilt, float(head_tilt_rad))
        if not sim.wait_until_at_setpoint(
            Actuators.head_tilt,
            timeout=30.0,
            position_tolerance=self.HEAD_WAIT_TOLERANCE_RAD,
        ):
            actual_tilt = float(sim.pull_status().head_tilt.pos)
            print(
                f"Head tilt tolerated for perception: command={float(head_tilt_rad):.3f}, actual={actual_tilt:.3f}",
                flush=True,
            )

    def _validate_head_settle(self, sim, *, head_pan_rad: float, head_tilt_rad: float) -> None:
        """Verify the simulated head actually reached the requested pose."""
        status = sim.pull_status()
        actual_pan = float(status.head_pan.pos)
        actual_tilt = float(status.head_tilt.pos)
        if abs(actual_pan - float(head_pan_rad)) > self.HEAD_TOLERANCE_RAD:
            raise RuntimeError(
                f"head_pan did not settle near target: command={head_pan_rad:.3f}, actual={actual_pan:.3f}"
            )
        if abs(actual_tilt - float(head_tilt_rad)) > self.HEAD_TOLERANCE_RAD:
            raise RuntimeError(
                f"head_tilt did not settle near target: command={head_tilt_rad:.3f}, actual={actual_tilt:.3f}"
            )

    def _resolve_head_targets(self, sim, target_center_xy: np.ndarray | None) -> tuple[float, float]:
        del sim, target_center_xy
        commanded_pan = float(self.head_config.head_pan_rad)
        commanded_tilt = float(self.head_config.head_tilt_rad)
        return commanded_pan, commanded_tilt

    def _poll_valid_head_camera_data(self, sim):
        """Poll until the head RGB-D stream contains a usable depth image."""
        for _ in range(self.MAX_CAPTURE_RETRIES):
            camera_status = sim.pull_camera_data()
            depth = getattr(camera_status, "cam_d435i_depth", None)
            rgb = getattr(camera_status, "cam_d435i_rgb", None)
            if depth is None or rgb is None:
                time.sleep(0.2)
                continue
            depth_array = np.asarray(depth, dtype=float)
            valid_count = int(np.count_nonzero(depth_array > 0.1))
            if valid_count > 500:
                return camera_status
            time.sleep(0.2)
        raise RuntimeError("head camera did not produce valid depth after alignment")

    def _resolve_head_camera_intrinsics(
        self,
        *,
        depth_shape: tuple[int, int],
        fallback_k: np.ndarray | None,
    ) -> tuple[np.ndarray, str]:
        """Return a head-camera K matrix, preferring MuJoCo camera parameters."""
        mujoco_intrinsics = self._compute_intrinsics_from_mujoco(depth_shape)
        if mujoco_intrinsics is not None:
            return mujoco_intrinsics, "mujoco_cam_fovy"

        if fallback_k is not None:
            return np.asarray(fallback_k, dtype=float).reshape(3, 3), "camera_status_K"

        height, width = depth_shape
        fovy_rad = math.radians(
            StretchCameras.cam_d435i_rgb.initial_camera_settings.field_of_view_vertical_in_degrees
        )
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx, cy = width / 2.0, height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float), "settings_fovy"

    def _compute_intrinsics_from_mujoco(self, depth_shape: tuple[int, int]) -> np.ndarray | None:
        """Compute K from MuJoCo camera FOV if the local runtime scene can be loaded."""
        if self.scene_xml_path is None or not self.scene_xml_path.exists():
            return None
        try:
            import mujoco
        except Exception:
            return None

        model = mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME)
        if cam_id < 0:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                for index in range(int(model.ncam))
            ]
            raise RuntimeError(f"Failed to find head camera {HEAD_CAMERA_NAME}. Available cameras: {available}")

        height, width = depth_shape
        fovy_rad = float(model.cam_fovy[cam_id]) * math.pi / 180.0
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx, cy = width / 2.0, height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

    def _resolve_head_camera_pose(self, sim) -> tuple[np.ndarray, str]:
        """Return a camera-to-world transform for the head RGB camera."""
        mujoco_pose = self._compute_camera_pose_from_mujoco(sim)
        if mujoco_pose is not None:
            return mujoco_pose, "mujoco_cam_xpos_xmat"

        try:
            link_pose = np.asarray(sim.get_link_pose(HEAD_CAMERA_BODY_NAME), dtype=float)
            return link_pose @ HEAD_CAMERA_LOCAL_TRANSFORM, "link_pose_realsense"
        except Exception:
            return np.eye(4, dtype=float), "identity_fallback"

    def _compute_camera_pose_from_mujoco(self, sim) -> np.ndarray | None:
        """Load the runtime scene locally and derive the head camera pose from MuJoCo kinematics."""
        if self.scene_xml_path is None or not self.scene_xml_path.exists():
            return None
        try:
            import mujoco
        except Exception:
            return None

        model = mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        data = mujoco.MjData(model)
        qpos = np.array(model.qpos0, copy=True)

        base_x, base_y, base_theta = map(float, sim.get_base_pose())
        qpos[0] = base_x
        qpos[1] = base_y
        qpos[3] = math.cos(base_theta / 2.0)
        qpos[4] = 0.0
        qpos[5] = 0.0
        qpos[6] = math.sin(base_theta / 2.0)

        status = sim.pull_status()
        self._set_joint_qpos(model, qpos, "joint_head_pan", float(status.head_pan.pos), mujoco)
        self._set_joint_qpos(model, qpos, "joint_head_tilt", float(status.head_tilt.pos), mujoco)

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME)
        if cam_id < 0:
            return None

        cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=float)
        cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = cam_mat @ GL_TO_CV_ROTATION
        transform[:3, 3] = cam_pos
        return transform

    @staticmethod
    def _set_joint_qpos(model, qpos: np.ndarray, joint_name: str, value: float, mujoco_module) -> None:
        """Set a scalar MuJoCo joint value inside a qpos vector."""
        joint_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return
        qpos_address = int(model.jnt_qposadr[joint_id])
        qpos[qpos_address] = value
