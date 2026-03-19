"""
Face Mesh task using MediaPipe Face Mesh.

Draws 468 facial landmarks and optional mesh tessellation on detected faces.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from src.tasks.base_task import BaseTask


class FaceMeshTask(BaseTask):
    """MediaPipe Face Mesh with 468 landmarks."""

    def __init__(self) -> None:
        self._face_mesh = None
        self._draw_mesh: bool = True
        self._draw_contours: bool = True
        self._mesh_color: Tuple[int, int, int] = (0, 200, 120)

    def _load_model(self):
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._mp_fm = mp.solutions.face_mesh
                self._face_mesh = self._mp_fm.FaceMesh(
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as exc:
                st.error(f"Could not load MediaPipe face mesh: {exc}")
        return self._face_mesh

    def get_name(self) -> str:
        return "Face Mesh"

    def get_icon(self) -> str:
        return "🎭"

    def get_description(self) -> str:
        return "Draw 468 facial landmark points and mesh tessellation with MediaPipe."

    def get_settings(self) -> None:
        self._draw_mesh = st.sidebar.checkbox(
            "Draw mesh tessellation", True, key="fm_mesh"
        )
        self._draw_contours = st.sidebar.checkbox(
            "Draw contours", True, key="fm_contours"
        )
        color_choice = st.sidebar.selectbox(
            "Mesh color",
            ["Green", "Cyan", "Yellow", "White", "Blue"],
            key="fm_color",
        )
        color_map = {
            "Green": (0, 200, 120),
            "Cyan": (255, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Blue": (255, 100, 0),
        }
        self._mesh_color = color_map.get(color_choice, (0, 200, 120))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"faces": 0}

        if model is None:
            return annotated, meta

        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_fm = mp.solutions.face_mesh

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            results = model.process(rgb)

            if results.multi_face_landmarks:
                meta["faces"] = len(results.multi_face_landmarks)
                draw_spec = mp_drawing.DrawingSpec(
                    color=self._mesh_color, thickness=1, circle_radius=1
                )

                for face_lm in results.multi_face_landmarks:
                    if self._draw_mesh:
                        mp_drawing.draw_landmarks(
                            image=annotated,
                            landmark_list=face_lm,
                            connections=mp_fm.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=self._mesh_color, thickness=1
                            ),
                        )
                    if self._draw_contours:
                        mp_drawing.draw_landmarks(
                            image=annotated,
                            landmark_list=face_lm,
                            connections=mp_fm.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style(),
                        )

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
