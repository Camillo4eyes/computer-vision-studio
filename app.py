"""
Computer Vision Studio - Main Streamlit Application

Entry point for the multi-task, multi-source computer vision app.
Run with:  streamlit run app.py
"""

import time
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

from config import APP_ICON, APP_NAME, APP_VERSION
from src.input_handler import get_live_webcam_frame, get_uploaded_image, get_video_frame, get_webcam_frame
from src.task_manager import (
    get_task_list,
    get_task_by_name,
    run_combined_tasks,
    run_single_task,
)
from src.visualization import (
    display_combined_grid,
    display_frame,
    draw_fps,
    render_result_info,
    resize_frame,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main application entry point."""

    # ── Sidebar header ───────────────────────────────────────────────────────
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_NAME}")
        st.caption(f"v{APP_VERSION}")
        st.divider()

        # ── Input source ─────────────────────────────────────────────────────
        st.subheader("📥 Input Source")
        source = st.radio(
            "Choose source",
            ["🎥 Live Webcam", "📹 Webcam", "🎬 Video Upload", "🖼️ Image Upload"],
            key="input_source",
            label_visibility="collapsed",
        )
        st.divider()

        # ── Task selection ───────────────────────────────────────────────────
        tasks_all = get_task_list()
        task_names = [f"{t.get_icon()} {t.get_name()}" for t in tasks_all]
        task_name_to_obj = {f"{t.get_icon()} {t.get_name()}": t for t in tasks_all}

        st.subheader("🎛️ Task Selection")
        combined_mode = st.toggle("🔀 Combined Mode", value=False, key="combined_mode")

        if combined_mode:
            st.markdown("Select 2–4 tasks to run simultaneously:")
            selected_labels = [
                name for name in task_names
                if st.checkbox(name, key=f"cb_{name}", value=False)
            ]
            active_tasks = [task_name_to_obj[n] for n in selected_labels]

            if len(active_tasks) > 4:
                st.warning("Please select at most 4 tasks.")
                active_tasks = active_tasks[:4]

            overlay_mode = st.checkbox(
                "🖼️ Overlay all on single frame",
                value=False,
                key="overlay_mode",
            )
        else:
            selected_label = st.radio(
                "Active task",
                task_names,
                key="active_task",
                label_visibility="collapsed",
            )
            active_tasks = [task_name_to_obj[selected_label]]
            overlay_mode = False

        # ── Task settings ────────────────────────────────────────────────────
        st.divider()
        st.subheader("⚙️ Task Settings")
        for task in active_tasks:
            with st.expander(f"{task.get_icon()} {task.get_name()}", expanded=True):
                task.get_settings()

    # ── Main area ────────────────────────────────────────────────────────────
    if combined_mode:
        header_label = "🔀 Combined Mode"
        desc = f"Running: {', '.join(t.get_name() for t in active_tasks)}"
    else:
        task = active_tasks[0]
        header_label = f"{task.get_icon()} {task.get_name()}"
        desc = task.get_description()

    st.header(header_label)
    st.caption(desc)

    # Placeholder for the main frame / grid
    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    video_msg_placeholder = st.empty()

    # ── FPS tracking ─────────────────────────────────────────────────────────
    FPS_KEY = "_cv_fps_ts"
    if FPS_KEY not in st.session_state:
        st.session_state[FPS_KEY] = time.time()

    def _get_fps() -> float:
        now = time.time()
        elapsed = now - st.session_state[FPS_KEY]
        st.session_state[FPS_KEY] = now
        return 1.0 / elapsed if elapsed > 0 else 0.0

    # ── Acquire frame from the selected source ────────────────────────────────
    frame: Optional[np.ndarray] = None

    if source == "🎥 Live Webcam":
        frame, _ = get_live_webcam_frame(video_msg_placeholder)

    elif source == "📹 Webcam":
        frame = get_webcam_frame()

    elif source == "🎬 Video Upload":
        frame, has_more = get_video_frame(video_msg_placeholder)

    else:  # Image upload
        frame = get_uploaded_image()

    # ── Nothing to process ───────────────────────────────────────────────────
    if frame is None:
        with frame_placeholder.container():
            if source == "🎥 Live Webcam":
                st.info(
                    "📷 Click **▶ Start** in the sidebar to begin live webcam streaming.\n\n"
                    "Make sure your webcam is connected and not in use by another application."
                )
            elif source == "📹 Webcam":
                st.info(
                    "📷 Click the camera button above to capture a frame from your webcam."
                )
            elif source == "🎬 Video Upload":
                st.info("🎬 Upload a video file and press **▶ Play** to start.")
            else:
                st.info("🖼️ Upload an image from the sidebar to get started.")
        return

    # ── Resize for manageable processing ─────────────────────────────────────
    frame = resize_frame(frame, max_width=1280)

    # ── Process ───────────────────────────────────────────────────────────────
    if combined_mode and len(active_tasks) >= 2:
        results = run_combined_tasks(active_tasks, frame, overlay_mode=overlay_mode)
        fps = _get_fps()

        with frame_placeholder.container():
            if overlay_mode:
                # Show the last (fully overlaid) frame
                _, last_frame, last_meta = results[-1]
                last_frame = draw_fps(last_frame, fps)
                display_frame(last_frame)
                with info_placeholder.container():
                    render_result_info(last_meta)
            else:
                display_combined_grid(results)
    else:
        task = active_tasks[0]
        annotated, meta = run_single_task(task, frame)
        fps = _get_fps()
        annotated = draw_fps(annotated, fps)

        with frame_placeholder.container():
            display_frame(annotated)

        with info_placeholder.container():
            render_result_info(meta)

    # ── Auto-rerun for live sources ───────────────────────────────────────────
    if source in ("🎥 Live Webcam", "📹 Webcam", "🎬 Video Upload"):
        time.sleep(0.03)  # ~30 FPS cap
        st.rerun()


if __name__ == "__main__":
    main()
