"""
Task Manager module.

Instantiates and manages all available CV tasks.
Supports both single-task and combined (multi-task) modes.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from src.tasks.base_task import BaseTask
from src.tasks.object_detection import ObjectDetectionTask
from src.tasks.instance_segmentation import InstanceSegmentationTask
from src.tasks.semantic_segmentation import SemanticSegmentationTask
from src.tasks.pose_estimation import PoseEstimationTask
from src.tasks.classification import ClassificationTask
from src.tasks.face_detection import FaceDetectionTask
from src.tasks.face_mesh import FaceMeshTask
from src.tasks.hand_tracking import HandTrackingTask
from src.tasks.edge_detection import EdgeDetectionTask
from src.tasks.optical_flow import OpticalFlowTask
from src.tasks.style_transfer import StyleTransferTask


def _build_task_registry() -> Dict[str, BaseTask]:
    """Instantiate all tasks and return a name→task mapping."""
    tasks: List[BaseTask] = [
        ObjectDetectionTask(),
        InstanceSegmentationTask(),
        SemanticSegmentationTask(),
        PoseEstimationTask(),
        ClassificationTask(),
        FaceDetectionTask(),
        FaceMeshTask(),
        HandTrackingTask(),
        EdgeDetectionTask(),
        OpticalFlowTask(),
        StyleTransferTask(),
    ]
    return {t.get_name(): t for t in tasks}


# Singleton registry stored in session state so tasks are not re-created
# on every Streamlit rerun.
_REGISTRY_KEY = "_cv_studio_tasks"


def get_task_registry() -> Dict[str, BaseTask]:
    """Return (or create) the singleton task registry."""
    if _REGISTRY_KEY not in st.session_state:
        st.session_state[_REGISTRY_KEY] = _build_task_registry()
    return st.session_state[_REGISTRY_KEY]


def get_task_list() -> List[BaseTask]:
    """Return all tasks as an ordered list."""
    return list(get_task_registry().values())


def get_task_by_name(name: str) -> Optional[BaseTask]:
    """Return a task by its display name, or None."""
    return get_task_registry().get(name)


def run_single_task(
    task: BaseTask, frame: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """Process *frame* with a single task and return annotated frame + metadata."""
    return task.process(frame)


def run_combined_tasks(
    tasks: List[BaseTask],
    frame: np.ndarray,
    overlay_mode: bool = False,
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Process *frame* with multiple tasks.

    Parameters
    ----------
    tasks : list of BaseTask
        Tasks to apply.
    frame : np.ndarray
        Input BGR frame.
    overlay_mode : bool
        If True, apply all tasks sequentially on the same frame.
        If False, apply each task independently on a copy of *frame*.

    Returns
    -------
    results : list of (task_label, annotated_frame, metadata)
    """
    results = []

    if overlay_mode:
        combined = frame.copy()
        for task in tasks:
            combined, meta = task.process(combined)
            results.append(
                (f"{task.get_icon()} {task.get_name()}", combined.copy(), meta)
            )
    else:
        for task in tasks:
            annotated, meta = task.process(frame.copy())
            results.append(
                (f"{task.get_icon()} {task.get_name()}", annotated, meta)
            )

    return results
