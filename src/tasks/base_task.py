"""
Base abstract class for all Computer Vision tasks.

Every concrete task must extend BaseTask and implement the abstract methods
so that the task manager can treat all tasks uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class BaseTask(ABC):
    """Abstract base class for a computer vision task."""

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for the task."""

    @abstractmethod
    def get_icon(self) -> str:
        """Return an emoji / icon string for the task."""

    @abstractmethod
    def get_description(self) -> str:
        """Return a short one-line description of the task."""

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR image (H × W × 3).

        Returns
        -------
        annotated_frame : np.ndarray
            Frame with visual overlays drawn on it.
        results : dict
            Task-specific metadata / statistics (e.g. number of detections).
        """

    # ------------------------------------------------------------------
    # Streamlit settings widget
    # ------------------------------------------------------------------

    def get_settings(self) -> None:
        """
        Render task-specific Streamlit widgets in the sidebar.

        The default implementation renders nothing; subclasses should
        override this method and use ``st.sidebar.*`` widgets to expose
        their tunable parameters.
        """

    # ------------------------------------------------------------------
    # Convenience helpers shared by many tasks
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        """Return a 3-channel BGR copy of *frame*, converting if needed."""
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if frame.ndim == 2:
            return np.stack([frame] * 3, axis=-1)
        return frame.copy()

    @staticmethod
    def _overlay_text(
        frame: np.ndarray,
        lines: List[str],
        origin: Tuple[int, int] = (10, 30),
        font_scale: float = 0.6,
        thickness: int = 2,
        color: Tuple[int, int, int] = (0, 255, 0),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Draw a list of text lines on *frame* with a shadow background."""
        import cv2

        x, y = origin
        line_height = int(30 * font_scale)
        for i, line in enumerate(lines):
            cy = y + i * line_height
            cv2.putText(frame, line, (x + 1, cy + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, bg_color, thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, line, (x, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)
        return frame
