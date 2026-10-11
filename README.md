# CodeMind Vision Studio 2.1

A modern dark-themed desktop image editor built with Python, CustomTkinter, Pillow, NumPy, and OpenCV.

## Features

- Modern dark graphical interface
- Brightness, contrast, and saturation controls
- Grayscale mode and multiple image filters
- Image cropping with mouse selection
- Rotate, flip, and resize tools
- Undo and redo history
- Zoom and fit-to-window controls
- RGB histogram and color analysis
- Image dimensions and file information
- Export to PNG, JPEG, WebP, BMP, and TIFF
- Optional YuNet face detection with bounding boxes
- Keyboard shortcuts and error handling

## Requirements

- Python 3.10 or newer
- CustomTkinter
- OpenCV
- NumPy
- Pillow

## Installation on Windows

Clone the repository, then open its folder:

```powershell
git clone https://github.com/mohamadtaha-dev/codemind-vision-studio.git
cd codemind-vision-studio
```

Create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the application:

```powershell
.\.venv\Scripts\python.exe codemind.py
```

## YuNet Face Detection

The application supports optional face detection using the YuNet ONNX model. The model may need to be downloaded when face detection is enabled. Internet access may be required.

Official model repository:

https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

If the required YuNet model or OpenCV API is unavailable, face detection may not work.

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+O | Open image |
| Ctrl+S | Export image |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+0 | Fit to window |
| Esc | Cancel crop selection |

## Project Information

- **Brand:** CodeMind
- **Application:** CodeMind Vision Studio
- **Version:** 2.1
- **Language:** Python

A computer vision and image editing portfolio project by CodeMind.