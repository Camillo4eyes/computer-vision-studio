# 🎥 Computer Vision Studio

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Camillo4eyes/computer-vision-studio/blob/main/notebooks/Computer_Vision_Studio_Demo.ipynb)

**A real-time, multi-task Computer Vision application built with Streamlit, OpenCV, YOLOv8 and MediaPipe.**

*Apply 11 different CV algorithms on webcam streams, uploaded videos or images — switch tasks on the fly or combine multiple tasks simultaneously.*

</div>

---

## ✨ Features

| Task | Technology | Description |
|------|-----------|-------------|
| 🔍 **Object Detection** | YOLOv8n | Detect and locate 80 COCO objects with bounding boxes |
| 🎭 **Instance Segmentation** | YOLOv8n-seg | Unique colour mask per object instance |
| 🧬 **Semantic Segmentation** | YOLOv8n-seg + colormap | Per-class colour overlay with configurable colormaps |
| 🏃 **Pose Estimation** | YOLOv8n-pose | 17-keypoint skeleton detection for humans |
| 🏷️ **Image Classification** | YOLOv8n-cls | Top-K ImageNet category predictions with probability bars |
| 👤 **Face Detection** | MediaPipe | Bounding boxes and 6 facial landmarks |
| 🎭 **Face Mesh** | MediaPipe | 468 facial landmarks + mesh tessellation |
| ✋ **Hand Tracking** | MediaPipe | 21 hand landmarks and connection drawing |
| 📐 **Edge Detection** | OpenCV | Canny, Sobel and Laplacian edge detectors |
| 🌊 **Optical Flow** | OpenCV Farneback | Dense motion estimation with HSV or arrow visualisation |
| 🎨 **Style Transfer** | OpenCV DNN | Fast neural style transfer (Mosaic, Candy, Udnie, …) |

### Additional capabilities
- **3 input sources:** 📹 Live Webcam · 🎬 Video Upload · 🖼️ Image Upload
- **Real-time task switching** via sidebar buttons — no restart required
- **Combined Mode** — run 2–4 tasks simultaneously in a responsive 2-column grid
- **Overlay mode** — compose all selected tasks onto a single frame
- **Per-task settings** — confidence thresholds, colours, algorithm parameters
- **FPS counter** — live performance monitoring

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Camillo4eyes/computer-vision-studio.git
cd computer-vision-studio
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🎮 Usage

### Input Sources

| Source | How to use |
|--------|-----------|
| 📹 **Webcam** | Select _Webcam_ in the sidebar and click the camera button |
| 🎬 **Video Upload** | Select _Video Upload_, upload a `.mp4`/`.avi`/`.mov`/`.mkv` file, press ▶ Play |
| 🖼️ **Image Upload** | Select _Image Upload_ and upload a `.jpg`/`.png`/`.webp` file |

### Single Task Mode

1. Open the sidebar
2. Select the desired task from the list (Object Detection, Pose Estimation, etc.)
3. Expand **Task Settings** to fine-tune parameters
4. The processed frame is displayed in the main area with result metadata

### Combined Mode

1. Toggle **🔀 Combined Mode** in the sidebar
2. Check 2–4 tasks you want to run simultaneously
3. Optionally enable **Overlay** to compose all results on a single frame
4. Results are shown in a 2-column grid, one cell per task

---

## 🏗️ Architecture

```
computer-vision-studio/
├── app.py                   # Streamlit entry point
├── config.py                # Global constants (colours, model names, COCO classes)
├── requirements.txt
├── setup.py
├── src/
│   ├── input_handler.py     # Webcam / video / image input management
│   ├── task_manager.py      # Task registry, single & combined processing
│   ├── visualization.py     # Frame display, result grids, FPS overlay
│   └── tasks/
│       ├── base_task.py             # Abstract BaseTask interface
│       ├── object_detection.py      # YOLOv8 detection
│       ├── instance_segmentation.py # YOLOv8-seg instance masks
│       ├── semantic_segmentation.py # Per-class colormap overlay
│       ├── pose_estimation.py       # YOLOv8-pose skeleton
│       ├── classification.py        # YOLOv8-cls top-K predictions
│       ├── face_detection.py        # MediaPipe face detection
│       ├── face_mesh.py             # MediaPipe 468-landmark mesh
│       ├── hand_tracking.py         # MediaPipe hand landmarks
│       ├── edge_detection.py        # Canny / Sobel / Laplacian
│       ├── optical_flow.py          # Farneback dense optical flow
│       └── style_transfer.py        # OpenCV DNN style transfer
├── notebooks/
│   └── Computer_Vision_Studio_Demo.ipynb   # Google Colab demo
├── assets/
│   └── README.md            # Instructions for sample assets
└── tests/
    └── test_tasks.py        # Unit tests for all tasks
```

### Extending with new tasks

1. Create a new file in `src/tasks/` extending `BaseTask`.
2. Implement `get_name()`, `get_icon()`, `get_description()`, `process()`.
3. Optionally implement `get_settings()` with Streamlit sidebar widgets.
4. Import and instantiate the class in `src/task_manager.py`.

---

## 📓 Google Colab Demo

Click the badge below to open the interactive demo notebook in Google Colab.
No local installation required!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Camillo4eyes/computer-vision-studio/blob/main/notebooks/Computer_Vision_Studio_Demo.ipynb)

The notebook demonstrates every task on a sample street-scene image, including:
- Individual task demos with inline visualisations
- Combined Mode grid (4 tasks side by side)
- Video processing and GIF export
- Google Colab webcam capture

---

## 🧪 Tests

```bash
pytest tests/ -v
```

The test suite verifies that every task:
- Can be instantiated
- Is a proper `BaseTask` subclass
- Implements all required interface methods
- Returns the expected `(np.ndarray, dict)` tuple from `process()`
- Produces correct output shapes

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web UI |
| `opencv-python` | Frame capture and classical CV algorithms |
| `ultralytics` | YOLOv8 models (detection, segmentation, pose, classification) |
| `mediapipe` | Face detection/mesh, hand tracking |
| `Pillow` | Image loading and format conversion |
| `numpy` | Array operations |

---

## 🤝 Contributing

Contributions are welcome! To add a new CV task or improve the UI:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-task`
3. Implement your changes following the architecture above
4. Add tests in `tests/test_tasks.py`
5. Open a Pull Request

Please keep code clean, add docstrings, and use type hints where appropriate.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ using Streamlit, OpenCV, YOLOv8 and MediaPipe
</div>