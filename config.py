"""
Global configuration constants for Computer Vision Studio.

Contains default parameters, color maps, model configurations,
and COCO class mappings used throughout the application.
"""

import os
import tempfile

# ---------------------------------------------------------------------------
# YOLO / Ultralytics environment setup
# ---------------------------------------------------------------------------
# On Windows, the default ultralytics config directory is resolved from the
# user home path (~/.config/ultralytics).  When the home directory contains
# non-ASCII characters (e.g. accented letters in the username) or resides on
# a network drive, model loading raises "[Errno 22] Invalid argument".
# Pointing YOLO_CONFIG_DIR at the system temp directory guarantees a safe,
# writable, ASCII-only path on all platforms.
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics")

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME = "Computer Vision Studio"
APP_VERSION = "1.0.0"
APP_ICON = "🎥"

# ---------------------------------------------------------------------------
# Default UI parameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_MAX_DETECTIONS = 300

# ---------------------------------------------------------------------------
# Video / webcam settings
# ---------------------------------------------------------------------------
WEBCAM_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# YOLO model names
# ---------------------------------------------------------------------------
YOLO_DETECTION_MODEL = "yolov8n.pt"
YOLO_SEGMENTATION_MODEL = "yolov8n-seg.pt"
YOLO_POSE_MODEL = "yolov8n-pose.pt"
YOLO_CLASSIFICATION_MODEL = "yolov8n-cls.pt"

# ---------------------------------------------------------------------------
# Default drawing colors (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE = (0, 165, 255)

# Default skeleton / pose color
POSE_COLOR = COLOR_GREEN
POSE_LINE_COLOR = COLOR_CYAN

# Face mesh color
FACE_MESH_COLOR = (0, 200, 120)

# Hand tracking color
HAND_LANDMARK_COLOR = COLOR_RED
HAND_CONNECTION_COLOR = COLOR_GREEN

# Edge detection overlay color
EDGE_COLOR = COLOR_WHITE

# ---------------------------------------------------------------------------
# COCO class names (80 classes)
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ---------------------------------------------------------------------------
# Color palette for segmentation (BGR)
# ---------------------------------------------------------------------------
SEGMENTATION_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# ---------------------------------------------------------------------------
# Style Transfer model URLs (OpenCV DNN)
# The models are ONNX models from the fast neural style transfer project.
# ---------------------------------------------------------------------------
STYLE_MODELS = {
    "Mosaic": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7",
    "Candy": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7",
    "Udnie": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7",
    "Rain Princess": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/rain-princess.t7",
    "La Muse": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/la_muse.t7",
    "The Scream": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7",
    "Feathers": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7",
}

# ---------------------------------------------------------------------------
# Optical flow parameters (Farneback)
# ---------------------------------------------------------------------------
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
