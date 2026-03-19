"""
Input Handler module.

Manages the four input sources for Computer Vision Studio:
  - Live Webcam (real-time OpenCV stream via cv2.VideoCapture)
  - Webcam snapshot (single frame via st.camera_input)
  - Video upload (frame-by-frame playback)
  - Image upload (single frame processing)
"""

import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    WEBCAM_INDEX,
)


def get_live_webcam_frame(
    msg_placeholder,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Stream live frames from the system webcam using OpenCV VideoCapture.

    Opens the default webcam device directly and delivers frames
    continuously.  Best suited for local deployments where the Streamlit
    server and the webcam share the same machine.

    Parameters
    ----------
    msg_placeholder : streamlit.delta_generator.DeltaGenerator
        A Streamlit placeholder used for status / error messages.

    Returns
    -------
    (frame, is_active) : (np.ndarray or None, bool)
        ``frame`` is the next BGR frame; ``is_active`` is True while the
        webcam capture is open and streaming.
    """
    CAP_KEY = "_cv_studio_webcam_cap"
    PLAY_KEY = "_cv_studio_webcam_playing"

    # ------------------------------------------------------------------ #
    # Start / Stop controls
    # ------------------------------------------------------------------ #
    st.sidebar.markdown("**🎬 Webcam Controls**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶ Start", key="webcam_start"):
            cap = cv2.VideoCapture(WEBCAM_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if cap.isOpened():
                st.session_state[CAP_KEY] = cap
                st.session_state[PLAY_KEY] = True
            else:
                cap.release()
                st.session_state[CAP_KEY] = None
                st.session_state[PLAY_KEY] = False
                st.sidebar.error("❌ Could not open webcam.")
    with col2:
        if st.button("⏹ Stop", key="webcam_stop"):
            try:
                if st.session_state.get(CAP_KEY) is not None:
                    st.session_state[CAP_KEY].release()
            finally:
                st.session_state[CAP_KEY] = None
                st.session_state[PLAY_KEY] = False

    if PLAY_KEY not in st.session_state:
        st.session_state[PLAY_KEY] = False

    if not st.session_state[PLAY_KEY]:
        return None, False

    cap = st.session_state.get(CAP_KEY)
    if cap is None or not cap.isOpened():
        msg_placeholder.error(
            "❌ Webcam not available. "
            "Check that your camera is connected and not in use by another app."
        )
        st.session_state[PLAY_KEY] = False
        return None, False

    ret, frame = cap.read()
    if not ret:
        msg_placeholder.warning("⚠️ Failed to read a frame from the webcam.")
        return None, True

    return frame, True


def get_webcam_frame() -> Optional[np.ndarray]:
    """
    Capture a single frame from the webcam using st.camera_input.

    Returns
    -------
    frame : np.ndarray or None
        BGR frame captured from the webcam, or None if unavailable.
    """
    img_file = st.camera_input("📷 Capture from webcam", label_visibility="collapsed")
    if img_file is None:
        return None
    pil_img = Image.open(img_file).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def get_uploaded_image() -> Optional[np.ndarray]:
    """
    Let the user upload an image file and return it as a BGR frame.

    Returns
    -------
    frame : np.ndarray or None
    """
    uploaded = st.sidebar.file_uploader(
        "Upload an image",
        type=SUPPORTED_IMAGE_TYPES,
        key="img_uploader",
    )
    if uploaded is None:
        return None
    pil_img = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def get_video_frame(
    video_placeholder,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Stream frames from an uploaded video file using session state.

    Uses Streamlit's session state to keep track of the current OpenCV
    VideoCapture object and frame position across reruns.

    Parameters
    ----------
    video_placeholder : streamlit.delta_generator.DeltaGenerator
        A Streamlit placeholder that will hold the "processing" message
        while a video is being processed.

    Returns
    -------
    (frame, has_more) : (np.ndarray or None, bool)
        ``frame`` is the next BGR frame; ``has_more`` indicates whether
        the video still has frames left to read.
    """
    # ------------------------------------------------------------------ #
    # Uploader widget
    # ------------------------------------------------------------------ #
    uploaded = st.sidebar.file_uploader(
        "Upload a video",
        type=SUPPORTED_VIDEO_TYPES,
        key="vid_uploader",
    )

    # ------------------------------------------------------------------ #
    # Session-state keys
    # ------------------------------------------------------------------ #
    CAP_KEY = "_cv_studio_cap"
    FILE_KEY = "_cv_studio_vid_name"
    PLAY_KEY = "_cv_studio_playing"

    if uploaded is None:
        # Clear any existing capture
        if CAP_KEY in st.session_state:
            cap = st.session_state[CAP_KEY]
            if cap is not None:
                cap.release()
            del st.session_state[CAP_KEY]
            del st.session_state[FILE_KEY]
        return None, False

    # ------------------------------------------------------------------ #
    # Play / Pause / Restart controls
    # ------------------------------------------------------------------ #
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("▶ Play", key="vid_play"):
            st.session_state[PLAY_KEY] = True
    with col2:
        if st.button("⏸ Pause", key="vid_pause"):
            st.session_state[PLAY_KEY] = False
    with col3:
        if st.button("⏹ Restart", key="vid_restart"):
            if CAP_KEY in st.session_state:
                cap = st.session_state[CAP_KEY]
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            st.session_state[PLAY_KEY] = True

    if PLAY_KEY not in st.session_state:
        st.session_state[PLAY_KEY] = False

    # ------------------------------------------------------------------ #
    # Re-open capture only when the file changes
    # ------------------------------------------------------------------ #
    if (
        CAP_KEY not in st.session_state
        or st.session_state.get(FILE_KEY) != uploaded.name
    ):
        # Write to temp file because OpenCV needs a path.
        # Use only the file extension (not the full name) as the suffix so
        # that the temp path stays short and free of spaces or special
        # characters that would cause OpenCV's MSMF backend to reject it on
        # Windows (error status -1072873821 / MF_E_INVALID_FORMAT).
        _ext = os.path.splitext(uploaded.name)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=_ext
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        st.session_state[CAP_KEY] = cap
        st.session_state[FILE_KEY] = uploaded.name
        st.session_state[PLAY_KEY] = False

    cap = st.session_state[CAP_KEY]

    if not st.session_state[PLAY_KEY]:
        video_placeholder.info("⏸ Video paused. Press **▶ Play** to start.")
        return None, True

    ret, frame = cap.read()
    if not ret:
        video_placeholder.success("✅ Video playback finished.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state[PLAY_KEY] = False
        return None, False

    return frame, True
