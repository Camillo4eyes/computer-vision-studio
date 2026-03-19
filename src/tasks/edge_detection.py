"""
Edge Detection task using classical OpenCV algorithms.

Supports Canny, Sobel and Laplacian edge detection methods
with optional overlay on the original frame.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from src.tasks.base_task import BaseTask

_ALGORITHMS = ["Canny", "Sobel", "Laplacian"]


class EdgeDetectionTask(BaseTask):
    """Classical OpenCV edge detection task."""

    def __init__(self) -> None:
        self._algorithm: str = "Canny"
        self._canny_low: int = 50
        self._canny_high: int = 150
        self._kernel_size: int = 3
        self._overlay: bool = False

    def get_name(self) -> str:
        return "Edge Detection"

    def get_icon(self) -> str:
        return "📐"

    def get_description(self) -> str:
        return "Detect edges using Canny, Sobel, or Laplacian algorithms."

    def get_settings(self) -> None:
        self._algorithm = st.sidebar.selectbox(
            "Algorithm", _ALGORITHMS, key="ed_algo"
        )
        if self._algorithm == "Canny":
            self._canny_low = st.sidebar.slider(
                "Canny low threshold", 0, 255, 50, 5, key="ed_clow"
            )
            self._canny_high = st.sidebar.slider(
                "Canny high threshold", 0, 255, 150, 5, key="ed_chigh"
            )
        else:
            self._kernel_size = st.sidebar.select_slider(
                "Kernel size", [1, 3, 5, 7], value=3, key="ed_ksize"
            )
        self._overlay = st.sidebar.checkbox(
            "Overlay on original", False, key="ed_overlay"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"algorithm": self._algorithm}

        try:
            gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)

            if self._algorithm == "Canny":
                edges = cv2.Canny(gray, self._canny_low, self._canny_high)
            elif self._algorithm == "Sobel":
                ksize = self._kernel_size if self._kernel_size % 2 == 1 else 3
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                edges = cv2.magnitude(sobel_x, sobel_y)
                edges = np.clip(edges, 0, 255).astype(np.uint8)
            else:  # Laplacian
                ksize = self._kernel_size if self._kernel_size % 2 == 1 else 3
                lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                edges = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            if self._overlay:
                annotated = cv2.addWeighted(annotated, 0.7, edges_bgr, 0.3, 0)
            else:
                annotated = edges_bgr

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
