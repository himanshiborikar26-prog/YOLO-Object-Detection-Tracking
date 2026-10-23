# YOLO Object Detection and Tracking

## About the Project
This project detects and tracks objects in real time using YOLO, OpenCV, and Python. It displays detected objects through a live camera dashboard.

## Features
- Real-time object detection
- Object tracking with IDs
- Confidence score display
- Live FPS and object count dashboard
- Object movement trails
- Line-crossing counter
- Save detection images
- Export detection reports as CSV

## Technologies Used
- Python
- YOLO ONNX Model
- OpenCV
- ONNX Runtime
- NumPy
- Pillow
- Tkinter

## Project Files
- `main.py` — Main application code
- `yolov8n (1).onnx` — Object detection model
- `bytetrack.yaml` — Tracking configuration
- `requirements.txt` — Required Python libraries
- `.gitignore` — Files excluded from Git tracking

## Installation

1. Install Python.
2. Download or clone this repository.
3. Open the project folder in VS Code.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

If the camera does not start, check camera permissions and ensure no other application is using it.

## Output
The application displays live object detection, tracking IDs, confidence scores, FPS, object counts, and line-crossing statistics.

## Author
GitHub: [himanshiborikar26-prog](https://github.com/himanshiborikar26-prog)
