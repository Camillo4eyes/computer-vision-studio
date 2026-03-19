"""
Semantic Segmentation task using YOLOv8-seg.

Applies a per-class colormap to produce a semantic segmentation overlay
where all instances of the same class share the same color.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from config import DEFAULT_CONFIDENCE, SEGMENTATION_PALETTE, YOLO_SEGMENTATION_MODEL
from src.tasks.base_task import BaseTask

_COLORMAPS = {
    "Viridis (default)": cv2.COLORMAP_VIRIDIS,
    "Jet": cv2.COLORMAP_JET,
    "HSV": cv2.COLORMAP_HSV,
    "Hot": cv2.COLORMAP_HOT,
    "Cool": cv2.COLORMAP_COOL,
    "Rainbow": cv2.COLORMAP_RAINBOW,
    "Ocean": cv2.COLORMAP_OCEAN,
}


class SemanticSegmentationTask(BaseTask):
    """YOLOv8-seg semantic segmentation (per-class colormap) task."""

    def __init__(self) -> None:
        self._model = None
        self._confidence: float = DEFAULT_CONFIDENCE
        self._opacity: float = 0.5
        self._colormap_name: str = "Viridis (default)"

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(YOLO_SEGMENTATION_MODEL)
            except Exception as exc:
                st.error(f"Could not load segmentation model: {exc}")
        return self._model

    def get_name(self) -> str:
        return "Semantic Segmentation"

    def get_icon(self) -> str:
        return "🧬"

    def get_description(self) -> str:
        return "Color-code regions by class using a configurable colormap."

    def get_settings(self) -> None:
        self._confidence = st.sidebar.slider(
            "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05,
            key="sem_conf",
        )
        self._opacity = st.sidebar.slider(
            "Overlay opacity", 0.0, 1.0, 0.5, 0.05, key="sem_opacity"
        )
        self._colormap_name = st.sidebar.selectbox(
            "Colormap", list(_COLORMAPS.keys()), key="sem_colormap"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"classes_found": []}

        if model is None:
            return annotated, meta

        try:
            results = model(annotated, conf=self._confidence, verbose=False)
            res = results[0]
            names = res.names
            h, w = annotated.shape[:2]

            # Build a per-pixel class label map
            class_map = np.zeros((h, w), dtype=np.uint8)
            classes_found = set()

            if res.masks is not None:
                masks = res.masks.data.cpu().numpy()
                for i, mask in enumerate(masks):
                    cls_id = int(res.boxes.cls[i]) if res.boxes is not None else 0
                    mask_resized = cv2.resize(mask, (w, h))
                    class_map[mask_resized > 0.5] = cls_id % 255
                    classes_found.add(names.get(cls_id, str(cls_id)))

            # Apply selected colormap
            colormap = _COLORMAPS.get(self._colormap_name, cv2.COLORMAP_VIRIDIS)
            colored = cv2.applyColorMap(class_map, colormap)

            annotated = cv2.addWeighted(colored, self._opacity, annotated, 1 - self._opacity, 0)
            meta["classes_found"] = list(classes_found)

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
