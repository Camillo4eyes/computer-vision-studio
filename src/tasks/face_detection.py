"""
Face Detection task using MediaPipe Face Detection.

Detects faces in a frame and optionally draws bounding boxes
and landmark points.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from src.tasks.base_task import BaseTask


class FaceDetectionTask(BaseTask):
    """MediaPipe-based face detection task."""

    def __init__(self) -> None:
        self._detector = None
        self._min_confidence: float = 0.5
        self._show_box: bool = True
        self._show_landmarks: bool = True

    def _load_detector(self):
        if self._detector is None:
            try:
                import mediapipe as mp
                self._mp_fd = mp.solutions.face_detection
                self._detector = self._mp_fd.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=self._min_confidence,
                )
            except Exception as exc:
                st.error(f"Could not load MediaPipe face detection: {exc}")
        return self._detector

    def get_name(self) -> str:
        return "Face Detection"

    def get_icon(self) -> str:
        return "👤"

    def get_description(self) -> str:
        return "Detect faces and facial landmarks using MediaPipe."

    def get_settings(self) -> None:
        new_conf = st.sidebar.slider(
            "Min detection confidence", 0.0, 1.0, 0.5, 0.05, key="fd_conf"
        )
        # Reinitialize detector if confidence changed
        if new_conf != self._min_confidence:
            self._min_confidence = new_conf
            self._detector = None

        self._show_box = st.sidebar.checkbox("Show bounding box", True, key="fd_box")
        self._show_landmarks = st.sidebar.checkbox(
            "Show landmarks", True, key="fd_landmarks"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        detector = self._load_detector()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"faces": 0}

        if detector is None:
            return annotated, meta

        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                meta["faces"] = len(results.detections)
                h, w = annotated.shape[:2]

                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    x2 = min(w - 1, x1 + bw)
                    y2 = min(h - 1, y1 + bh)

                    if self._show_box:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        score = det.score[0] if det.score else 0.0
                        cv2.putText(
                            annotated, f"Face {score:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA,
                        )

                    if self._show_landmarks:
                        for kp in det.location_data.relative_keypoints:
                            px = int(kp.x * w)
                            py = int(kp.y * h)
                            cv2.circle(annotated, (px, py), 4, (255, 0, 0), -1)

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
