"""
Instance Segmentation task using YOLOv8-seg.

Detects objects and draws per-instance color masks with optional contours.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from config import DEFAULT_CONFIDENCE, SEGMENTATION_PALETTE, YOLO_SEGMENTATION_MODEL
from src.tasks.base_task import BaseTask


class InstanceSegmentationTask(BaseTask):
    """YOLOv8-seg instance segmentation task."""

    def __init__(self) -> None:
        self._model = None
        self._confidence: float = DEFAULT_CONFIDENCE
        self._mask_opacity: float = 0.45
        self._show_contours: bool = True

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(YOLO_SEGMENTATION_MODEL)
            except Exception as exc:
                st.error(f"Could not load YOLO segmentation model: {exc}")
        return self._model

    def get_name(self) -> str:
        return "Instance Segmentation"

    def get_icon(self) -> str:
        return "🎭"

    def get_description(self) -> str:
        return "Segment each object instance with a unique color mask (YOLOv8-seg)."

    def get_settings(self) -> None:
        self._confidence = st.sidebar.slider(
            "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05,
            key="seg_conf",
        )
        self._mask_opacity = st.sidebar.slider(
            "Mask opacity", 0.0, 1.0, 0.45, 0.05, key="seg_opacity"
        )
        self._show_contours = st.sidebar.checkbox(
            "Show contours", True, key="seg_contours"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"instances": 0, "classes": []}

        if model is None:
            return annotated, meta

        try:
            results = model(annotated, conf=self._confidence, verbose=False)
            res = results[0]
            names = res.names

            overlay = annotated.copy()
            detected_classes = []

            if res.masks is not None:
                masks = res.masks.data.cpu().numpy()  # (N, H, W)
                boxes = res.boxes

                for i, mask in enumerate(masks):
                    cls_id = int(boxes.cls[i]) if boxes is not None else 0
                    color = SEGMENTATION_PALETTE[cls_id % len(SEGMENTATION_PALETTE)]
                    label = names.get(cls_id, str(cls_id))
                    detected_classes.append(label)

                    # Resize mask to frame size
                    h, w = annotated.shape[:2]
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_bool = mask_resized > 0.5

                    overlay[mask_bool] = color

                    if self._show_contours:
                        contour_mask = (mask_resized * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(
                            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(annotated, contours, -1, color, 2)

            # Blend overlay
            annotated = cv2.addWeighted(
                overlay, self._mask_opacity, annotated, 1 - self._mask_opacity, 0
            )

            meta["instances"] = len(detected_classes)
            meta["classes"] = list(set(detected_classes))

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
