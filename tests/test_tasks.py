"""
Basic tests for the Computer Vision Studio task system.

These tests verify that:
1. Each task class can be instantiated successfully.
2. Each task implements the required BaseTask interface.
3. The `process` method accepts a synthetic frame without raising exceptions.
"""

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a blank (black) BGR frame of the given dimensions."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# List of all task classes
# ---------------------------------------------------------------------------

ALL_TASK_CLASSES = [
    ObjectDetectionTask,
    InstanceSegmentationTask,
    SemanticSegmentationTask,
    PoseEstimationTask,
    ClassificationTask,
    FaceDetectionTask,
    FaceMeshTask,
    HandTrackingTask,
    EdgeDetectionTask,
    OpticalFlowTask,
    StyleTransferTask,
]


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_instantiation(TaskClass):
    """Each task class must be instantiable without errors."""
    task = TaskClass()
    assert task is not None


@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_is_base_task_subclass(TaskClass):
    """Each task must be a subclass of BaseTask."""
    assert issubclass(TaskClass, BaseTask)


@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_get_name_returns_non_empty_string(TaskClass):
    """get_name() must return a non-empty string."""
    task = TaskClass()
    name = task.get_name()
    assert isinstance(name, str)
    assert len(name) > 0


@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_get_icon_returns_non_empty_string(TaskClass):
    """get_icon() must return a non-empty string."""
    task = TaskClass()
    icon = task.get_icon()
    assert isinstance(icon, str)
    assert len(icon) > 0


@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_get_description_returns_non_empty_string(TaskClass):
    """get_description() must return a non-empty string."""
    task = TaskClass()
    desc = task.get_description()
    assert isinstance(desc, str)
    assert len(desc) > 0


@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_process_returns_tuple(TaskClass):
    """
    process() must return a tuple of (np.ndarray, dict) even when no model
    is available (the task should handle errors gracefully).
    """
    task = TaskClass()
    frame = _blank_frame()
    result = task.process(frame)

    assert isinstance(result, tuple), "process() must return a tuple"
    assert len(result) == 2, "process() must return exactly 2 elements"

    annotated_frame, metadata = result
    assert isinstance(annotated_frame, np.ndarray), "First element must be np.ndarray"
    assert isinstance(metadata, dict), "Second element must be dict"


@pytest.mark.parametrize("TaskClass", ALL_TASK_CLASSES)
def test_process_output_shape(TaskClass):
    """
    The annotated frame returned by process() must have 3 channels (BGR).
    """
    task = TaskClass()
    frame = _blank_frame(240, 320)
    annotated, _ = task.process(frame)

    assert annotated.ndim == 3, "Output frame must be 3-dimensional"
    assert annotated.shape[2] == 3, "Output frame must have 3 channels (BGR)"


def test_optical_flow_reset():
    """OpticalFlowTask.reset() must clear the previous frame buffer."""
    task = OpticalFlowTask()
    frame = _blank_frame()
    task.process(frame)  # initialise buffer
    task.reset()
    assert task._prev_gray is None


def test_edge_detection_canny_vs_sobel():
    """
    EdgeDetectionTask must produce different outputs for Canny vs Sobel
    on a non-trivial frame.
    """
    import cv2

    task_canny = EdgeDetectionTask()
    task_canny._algorithm = "Canny"

    task_sobel = EdgeDetectionTask()
    task_sobel._algorithm = "Sobel"

    # A gradient frame (non-trivial edges)
    frame = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
    frame = np.stack([frame] * 3, axis=-1)

    out_canny, _ = task_canny.process(frame)
    out_sobel, _ = task_sobel.process(frame)

    # Outputs must differ
    assert not np.array_equal(out_canny, out_sobel)
