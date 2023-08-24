# onnx-yolo-mwe
Minimum working examples of inference using an ONNX Ultralytics YOLO model in javascript and python.

---
## Requirements
The Python code requires the packages
- `ultralytics`
- `onnxruntime`

The javascript implementation requires `onnxruntime` and `jimp`, which are accessed using a CDN.

---
## Usage

- Download inference image and YOLOv8 (and convert to .onnx) with `python init.py`.
- Run inference on image in python with `python main.py`.
- Run inference on image in javascript by hosting `index.html` on local server and pressing 'Click to detect.'.

Non maximum suppression is currently only used in the javascript implementation.