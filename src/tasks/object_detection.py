"""
Object Detection task using YOLOv8.

Detects objects in a frame and draws bounding boxes with class labels
and confidence scores.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from config import (
    COCO_CLASSES,
    DEFAULT_CONFIDENCE,
    SEGMENTATION_PALETTE,
    YOLO_DETECTION_MODEL,
)
from src.tasks.base_task import BaseTask


class ObjectDetectionTask(BaseTask):
    """YOLOv8-based object detection task."""

    def __init__(self) -> None:
        self._model = None
        self._confidence: float = DEFAULT_CONFIDENCE
        self._selected_classes: Optional[List[int]] = None
        self._show_labels: bool = True
        self._show_boxes: bool = True

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(YOLO_DETECTION_MODEL)
            except Exception as exc:
                st.error(f"Could not load YOLO detection model: {exc}")
        return self._model

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "Object Detection"

    def get_icon(self) -> str:
        return "🔍"

    def get_description(self) -> str:
        return "Detect and locate objects using YOLOv8 with bounding boxes."

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_settings(self) -> None:
        self._confidence = st.sidebar.slider(
            "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05,
            key="od_conf",
        )
        self._show_labels = st.sidebar.checkbox("Show labels", True, key="od_labels")
        self._show_boxes = st.sidebar.checkbox("Show bounding boxes", True, key="od_boxes")

        class_names = COCO_CLASSES
        selected = st.sidebar.multiselect(
            "Filter classes (empty = all)",
            options=class_names,
            default=[],
            key="od_classes",
        )
        self._selected_classes = (
            [class_names.index(c) for c in selected if c in class_names]
            if selected
            else None
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        results_meta: Dict[str, Any] = {"detections": 0, "classes": []}

        if model is None:
            return annotated, results_meta

        try:
            results = model(
                annotated,
                conf=self._confidence,
                verbose=False,
                classes=self._selected_classes,
            )
            boxes = results[0].boxes
            names = results[0].names

            detected_classes = []
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names.get(cls_id, str(cls_id))
                    detected_classes.append(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = SEGMENTATION_PALETTE[cls_id % len(SEGMENTATION_PALETTE)]

                    if self._show_boxes:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    if self._show_labels:
                        text = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                        )
                        cv2.rectangle(
                            annotated, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1
                        )
                        cv2.putText(
                            annotated, text, (x1 + 1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                        )

            results_meta["detections"] = len(detected_classes)
            results_meta["classes"] = list(set(detected_classes))

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, results_meta
