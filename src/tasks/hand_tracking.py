"""
Hand Tracking task using MediaPipe Hands.

Detects and tracks hand landmarks and draws connections between them.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from src.tasks.base_task import BaseTask


class HandTrackingTask(BaseTask):
    """MediaPipe Hands tracking task."""

    def __init__(self) -> None:
        self._hands = None
        self._max_hands: int = 2
        self._min_detection_confidence: float = 0.5
        self._show_landmarks: bool = True
        self._show_connections: bool = True

    def _load_model(self):
        if self._hands is None:
            try:
                import mediapipe as mp
                self._mp_hands = mp.solutions.hands
                self._hands = self._mp_hands.Hands(
                    max_num_hands=self._max_hands,
                    min_detection_confidence=self._min_detection_confidence,
                    min_tracking_confidence=0.5,
                )
            except Exception as exc:
                st.error(f"Could not load MediaPipe Hands: {exc}")
        return self._hands

    def get_name(self) -> str:
        return "Hand Tracking"

    def get_icon(self) -> str:
        return "✋"

    def get_description(self) -> str:
        return "Track hand landmarks and connections using MediaPipe Hands."

    def get_settings(self) -> None:
        new_max = st.sidebar.slider(
            "Max number of hands", 1, 4, 2, 1, key="ht_max"
        )
        new_conf = st.sidebar.slider(
            "Min detection confidence", 0.0, 1.0, 0.5, 0.05, key="ht_conf"
        )
        # Reinitialize if params changed
        if new_max != self._max_hands or new_conf != self._min_detection_confidence:
            self._max_hands = new_max
            self._min_detection_confidence = new_conf
            self._hands = None

        self._show_landmarks = st.sidebar.checkbox(
            "Show landmark dots", True, key="ht_lm"
        )
        self._show_connections = st.sidebar.checkbox(
            "Show connections", True, key="ht_conn"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"hands": 0, "handedness": []}

        if model is None:
            return annotated, meta

        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_hands = mp.solutions.hands

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            results = model.process(rgb)

            if results.multi_hand_landmarks:
                meta["hands"] = len(results.multi_hand_landmarks)

                if results.multi_handedness:
                    meta["handedness"] = [
                        h.classification[0].label
                        for h in results.multi_handedness
                    ]

                for hand_lm in results.multi_hand_landmarks:
                    if self._show_connections:
                        mp_drawing.draw_landmarks(
                            annotated,
                            hand_lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                    elif self._show_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated,
                            hand_lm,
                            None,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            None,
                        )

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
