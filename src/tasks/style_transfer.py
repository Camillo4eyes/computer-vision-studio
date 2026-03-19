"""
Neural Style Transfer task using OpenCV DNN.

Downloads pre-trained Torch7 style models (.t7) and applies
artistic style to each frame using the fast feed-forward network
from Johnson et al. (2016).

Models are cached locally in a 'models/' directory to avoid
re-downloading on every run.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from config import STYLE_MODELS
from src.tasks.base_task import BaseTask

_MODELS_DIR = Path("models")


class StyleTransferTask(BaseTask):
    """Neural Style Transfer using pre-trained OpenCV DNN models."""

    def __init__(self) -> None:
        self._net = None
        self._current_style: Optional[str] = None
        self._selected_style: str = list(STYLE_MODELS.keys())[0]

    def _load_model(self, style_name: str):
        """Download (if needed) and load the selected style model."""
        if self._net is not None and self._current_style == style_name:
            return self._net

        url = STYLE_MODELS.get(style_name)
        if not url:
            return None

        _MODELS_DIR.mkdir(exist_ok=True)
        model_path = _MODELS_DIR / f"{style_name.lower().replace(' ', '_')}.t7"

        if not model_path.exists():
            try:
                import requests
                st.info(f"Downloading style model '{style_name}'…")
                # Try HTTPS first, then fall back to HTTP if the server rejects the secure request
                urls_to_try = [url]
                if url.startswith("https://"):
                    urls_to_try.append("http://" + url[len("https://"):])
                last_exc: Exception = RuntimeError("No URLs to try.")
                for attempt_url in urls_to_try:
                    try:
                        response = requests.get(attempt_url, timeout=60, stream=True)
                        response.raise_for_status()
                        model_path.write_bytes(response.content)
                        last_exc = None
                        break
                    except requests.exceptions.HTTPError as http_err:
                        last_exc = http_err
                    except requests.exceptions.ConnectionError as conn_err:
                        last_exc = conn_err
                if last_exc is not None:
                    raise last_exc
            except Exception as exc:
                st.error(
                    f"Could not download style model '{style_name}': {exc}\n\n"
                    "The model host may be temporarily unavailable. "
                    "Please try again later or select a different style."
                )
                return None

        try:
            self._net = cv2.dnn.readNetFromTorch(str(model_path))
            self._current_style = style_name
        except Exception as exc:
            st.error(f"Could not load style model: {exc}")
            self._net = None

        return self._net

    def get_name(self) -> str:
        return "Style Transfer"

    def get_icon(self) -> str:
        return "🎨"

    def get_description(self) -> str:
        return "Apply artistic neural style transfer (fast feed-forward, OpenCV DNN)."

    def get_settings(self) -> None:
        self._selected_style = st.sidebar.selectbox(
            "Style", list(STYLE_MODELS.keys()), key="st_style"
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        net = self._load_model(self._selected_style)
        annotated = self._ensure_bgr(frame)
        meta: Dict[str, Any] = {"style": self._selected_style}

        if net is None:
            self._overlay_text(
                annotated,
                ["Style model not available.", "Check your internet connection."],
            )
            return annotated, meta

        try:
            h, w = annotated.shape[:2]
            # Resize for faster inference (style models work at any size)
            target_w = min(w, 512)
            target_h = int(h * target_w / w)

            blob = cv2.dnn.blobFromImage(
                annotated,
                scalefactor=1.0,
                size=(target_w, target_h),
                mean=(103.939, 116.779, 123.680),
                swapRB=False,
                crop=False,
            )
            net.setInput(blob)
            output = net.forward()

            # output shape: (1, 3, H, W)
            output = output.squeeze().transpose(1, 2, 0)
            output += np.array([103.939, 116.779, 123.680])
            output = np.clip(output, 0, 255).astype(np.uint8)

            # Resize back to original
            annotated = cv2.resize(output, (w, h))

        except Exception as exc:
            cv2.putText(
                annotated, f"Error: {exc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        return annotated, meta
