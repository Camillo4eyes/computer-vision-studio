"""
Pose Estimation task using YOLOv8-pose.

Detects human body keypoints and draws the skeleton on each detected person.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from config import DEFAULT_CONFIDENCE, YOLO_POSE_MODEL
from src.tasks.base_task import BaseTask

# COCO 17 skeleton connectivity pairs (0-indexed)
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]


class PoseEstimationTask(BaseTask):
    """YOLOv8-pose skeleton detection task."""

    def __init__(self) -> None:
        self._model = None
        self._confidence: float = DEFAULT_CONFIDENCE
        self._show_skeleton: bool = True
        self._show_keypoints: bool = True
        self._skeleton_color: Tuple[int, int, int] = (0, 255, 0)

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(YOLO_POSE_MODEL)
            except Exception as exc:
                st.error(f"Could not load YOLO pose model: {exc}")
        return self._model

    def get_name(self) -> str:
        return "Pose Estimation"

    def get_icon(self) -> str:
        return "🏃"

    def get_description(self) -> str:
        return "Detect human body keypoints and skeleton using YOLOv8-pose."

    def get_settings(self) -> None:
        self._confidence = st.sidebar.slider(
            "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05,
            key="pose_conf",
        )
        self._show_skeleton = st.sidebar.checkbox(
            "Show skeleton lines", True, key="pose_skeleton"
        )
        self._show_keypoints = st.sidebar.checkbox(
            "Show keypoint dots", True, key="pose_kp"
        )
        color_choice = st.sidebar.selectbox(
            "Skeleton color",
            ["Green", "Cyan", "Yellow", "Red", "White"],
            key="pose_color",
        )
        color_map = {
            "Green": (0, 255, 0),
            "Cyan": (255, 255, 0),
            "Yellow": (0, 255, 255),
            "Red": (0, 0, 255),
            "White": (255, 255, 255),
        }
        self._skeleton_color = color_map.get(color_choice, (0, 255, 0))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"persons": 0}

        if model is None:
            return annotated, meta

        try:
            results = model(annotated, conf=self._confidence, verbose=False)
            res = results[0]

            if res.keypoints is not None:
                kpts_all = res.keypoints.xy.cpu().numpy()  # (N, 17, 2)
                kpts_conf = res.keypoints.conf
                if kpts_conf is not None:
                    kpts_conf = kpts_conf.cpu().numpy()

                meta["persons"] = len(kpts_all)

                for person_kpts in kpts_all:
                    # Draw skeleton lines
                    if self._show_skeleton:
                        for (a, b) in _SKELETON:
                            if a < len(person_kpts) and b < len(person_kpts):
                                xa, ya = int(person_kpts[a][0]), int(person_kpts[a][1])
                                xb, yb = int(person_kpts[b][0]), int(person_kpts[b][1])
                                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                                    cv2.line(annotated, (xa, ya), (xb, yb),
                                             self._skeleton_color, 2, cv2.LINE_AA)

                    # Draw keypoint dots
                    if self._show_keypoints:
                        for kp in person_kpts:
                            x, y = int(kp[0]), int(kp[1])
                            if x > 0 and y > 0:
                                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
                                cv2.circle(annotated, (x, y), 4, (255, 255, 255), 1)

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
