import urllib.request
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx')

urllib.request.urlretrieve('https://projectblue.blob.core.windows.net/media/Default/Beef%20&%20Lamb/beef%20%20sheep-1.jpg', 'cows-and-sheep.jpeg')