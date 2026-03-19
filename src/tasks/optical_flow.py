"""
Optical Flow task using the Farneback dense optical flow algorithm.

Computes motion between consecutive frames and visualizes it either
as a colour-coded HSV map or as a vector arrow field.

Note: this task requires a sequence of frames to work well.
      On a single image it will show no motion.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from config import FARNEBACK_PARAMS
from src.tasks.base_task import BaseTask

_VIS_MODES = ["HSV colour map", "Arrow field"]


class OpticalFlowTask(BaseTask):
    """Farneback dense optical flow task."""

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None
        self._vis_mode: str = _VIS_MODES[0]
        self._scale: float = 1.0

    def get_name(self) -> str:
        return "Optical Flow"

    def get_icon(self) -> str:
        return "🌊"

    def get_description(self) -> str:
        return "Visualise dense motion between frames (Farneback algorithm)."

    def get_settings(self) -> None:
        self._vis_mode = st.sidebar.selectbox(
            "Visualisation mode", _VIS_MODES, key="of_vis"
        )
        self._scale = st.sidebar.slider(
            "Frame scale", 0.25, 1.0, 1.0, 0.25, key="of_scale"
        )

    def reset(self) -> None:
        """Reset the previous frame buffer (call when switching sources)."""
        self._prev_gray = None

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"flow": "computed"}

        try:
            # Optionally downscale for speed
            if self._scale < 1.0:
                h, w = annotated.shape[:2]
                small = cv2.resize(
                    annotated,
                    (int(w * self._scale), int(h * self._scale)),
                )
            else:
                small = annotated.copy()

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if self._prev_gray is None or self._prev_gray.shape != gray.shape:
                self._prev_gray = gray
                meta["flow"] = "initialising"
                return annotated, meta

            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, **FARNEBACK_PARAMS
            )
            self._prev_gray = gray

            if self._vis_mode == "HSV colour map":
                output = self._flow_to_hsv(flow)
            else:
                output = self._flow_to_arrows(small.copy(), flow)

            # Scale back up if needed
            if self._scale < 1.0:
                h, w = annotated.shape[:2]
                output = cv2.resize(output, (w, h))

            annotated = output

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flow_to_hsv(flow: np.ndarray) -> np.ndarray:
        """Convert flow field to an HSV colour-coded BGR image."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2   # hue = direction
        hsv[..., 1] = 255                       # full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _flow_to_arrows(frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
        """Draw motion arrows on *frame* sampled every *step* pixels."""
        h, w = frame.shape[:2]
        y_coords, x_coords = np.mgrid[step // 2:h:step, step // 2:w:step]
        fx = flow[y_coords, x_coords, 0]
        fy = flow[y_coords, x_coords, 1]

        for (x, y, dx, dy) in zip(
            x_coords.flat, y_coords.flat, fx.flat, fy.flat
        ):
            end = (int(x + dx), int(y + dy))
            cv2.arrowedLine(
                frame, (int(x), int(y)), end, (0, 255, 0), 1,
                tipLength=0.4, line_type=cv2.LINE_AA,
            )
        return frame
