from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from validation_common import ensure_dir


class PipelineVideoRecorder:
    def __init__(self, output_path: str | Path, *, fps: float = 2.0, frame_size: tuple[int, int] = (1280, 720)):
        self.output_path = Path(output_path)
        ensure_dir(self.output_path.parent)
        self.fps = float(fps)
        self.frame_size = frame_size
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.frame_size,
        )

    @staticmethod
    def _to_bgr(image_rgb: np.ndarray | None) -> np.ndarray:
        if image_rgb is None:
            return np.zeros((360, 640, 3), dtype=np.uint8)
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def add_frame(
        self,
        *,
        stage_name: str,
        head_rgb: np.ndarray | None,
        wrist_rgb: np.ndarray | None,
        lines: list[str] | None = None,
        repeat: int = 1,
    ) -> None:
        left = self._to_bgr(head_rgb)
        right = self._to_bgr(wrist_rgb)
        target_half_width = self.frame_size[0] // 2
        target_height = self.frame_size[1]
        left = cv2.resize(left, (target_half_width, target_height))
        right = cv2.resize(right, (target_half_width, target_height))
        canvas = np.concatenate([left, right], axis=1)

        title = f"Stage: {stage_name}"
        cv2.putText(canvas, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Head D435i", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Wrist D405",
            (target_half_width + 20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (120, 255, 120),
            2,
            cv2.LINE_AA,
        )
        if lines:
            y = 110
            for line in lines:
                cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
                y += 28

        for _ in range(max(1, int(repeat))):
            self.writer.write(canvas)

    def close(self) -> None:
        self.writer.release()
