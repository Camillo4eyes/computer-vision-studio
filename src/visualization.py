"""
Visualization helpers for Computer Vision Studio.

Provides utility functions for displaying frames, metadata overlays,
and result grids in the Streamlit UI.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR NumPy array to RGB for Streamlit / PIL display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def display_frame(
    frame: np.ndarray,
    caption: str = "",
    use_column_width: bool = True,
) -> None:
    """
    Display a BGR frame in Streamlit.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to display.
    caption : str
        Optional caption shown below the image.
    use_column_width : bool
        Stretch image to fill the column width.
    """
    rgb = bgr_to_rgb(frame)
    st.image(rgb, caption=caption, use_container_width=use_column_width)


def display_combined_grid(
    results: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    num_cols: int = 2,
) -> None:
    """
    Display multiple task results in a responsive grid.

    Parameters
    ----------
    results : list of (label, frame, metadata)
        Output from ``task_manager.run_combined_tasks``.
    num_cols : int
        Number of columns in the grid (default 2).
    """
    cols = st.columns(num_cols)
    for i, (label, frame, meta) in enumerate(results):
        col = cols[i % num_cols]
        with col:
            st.markdown(f"**{label}**")
            display_frame(frame)
            _render_meta_pills(meta)


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw a semi-transparent FPS counter in the top-right corner."""
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    h, w = frame.shape[:2]
    x = w - tw - 12
    y = 22
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA,
    )
    return frame


def render_result_info(meta: Dict[str, Any]) -> None:
    """
    Render a compact result info bar below the main frame.

    Displays key metrics from the task metadata dict.
    """
    if not meta:
        return

    _render_meta_pills(meta)


def _render_meta_pills(meta: Dict[str, Any]) -> None:
    """Render metadata as small Streamlit metric / info badges."""
    skip_keys = set()
    items = []

    for key, value in meta.items():
        if key in skip_keys:
            continue
        if isinstance(value, (int, float)):
            items.append(f"**{key.replace('_', ' ').title()}:** {value}")
        elif isinstance(value, list) and value:
            items.append(
                f"**{key.replace('_', ' ').title()}:** {', '.join(str(v) for v in value[:5])}"
            )
        elif isinstance(value, str) and value:
            items.append(f"**{key.replace('_', ' ').title()}:** {value}")

    if items:
        st.caption(" · ".join(items))


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Down-scale *frame* to fit within *max_width* while preserving aspect."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
