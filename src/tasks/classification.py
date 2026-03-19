"""
Image Classification task using YOLOv8-cls.

Runs image classification and displays the top-5 predictions
with a visual probability bar overlay.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from config import YOLO_CLASSIFICATION_MODEL
from src.tasks.base_task import BaseTask


class ClassificationTask(BaseTask):
    """YOLOv8-cls image classification task."""

    def __init__(self) -> None:
        self._model = None
        self._top_k: int = 5

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(YOLO_CLASSIFICATION_MODEL)
            except Exception as exc:
                st.error(f"Could not load YOLO classification model: {exc}")
        return self._model

    def get_name(self) -> str:
        return "Image Classification"

    def get_icon(self) -> str:
        return "🏷️"

    def get_description(self) -> str:
        return "Classify the entire frame into ImageNet categories (YOLOv8-cls)."

    def get_settings(self) -> None:
        self._top_k = st.sidebar.slider(
            "Top-K predictions", 1, 10, 5, 1, key="cls_topk"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self._load_model()
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"top_class": None, "predictions": []}

        if model is None:
            return annotated, meta

        try:
            results = model(annotated, verbose=False)
            res = results[0]

            if res.probs is not None:
                probs = res.probs.data.cpu().numpy()
                names = res.names
                top_indices = probs.argsort()[::-1][: self._top_k]

                preds = [
                    (names.get(int(i), str(i)), float(probs[i]))
                    for i in top_indices
                ]
                meta["top_class"] = preds[0][0] if preds else None
                meta["predictions"] = preds

                # Draw probability bars on frame
                h, w = annotated.shape[:2]
                bar_w = min(300, w // 2)
                x_start = 10
                y_start = 20
                bar_h = 22
                gap = 6

                for rank, (label, prob) in enumerate(preds):
                    y = y_start + rank * (bar_h + gap)
                    filled = int(prob * bar_w)

                    # Background bar
                    cv2.rectangle(
                        annotated,
                        (x_start, y),
                        (x_start + bar_w, y + bar_h),
                        (50, 50, 50),
                        -1,
                    )
                    # Filled portion
                    cv2.rectangle(
                        annotated,
                        (x_start, y),
                        (x_start + filled, y + bar_h),
                        (0, 200, 100),
                        -1,
                    )
                    # Label
                    text = f"{label[:25]}: {prob:.1%}"
                    cv2.putText(
                        annotated, text, (x_start + 4, y + bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                    )

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
