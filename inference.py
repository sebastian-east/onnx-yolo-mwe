import onnxruntime as ort
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def drawbox(box):
    x, y, w, h = box[:4]
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'r')

model = ort.InferenceSession("./yolov8n.onnx")

img = Image.open('./cows-and-sheep.jpeg')
img = img.resize((640, 640)) # Resize image to 640 x 640
input = (np.array(img) / 255) # Scale data to [0, 1]
input = input.astype(np.float32) # Convert to float
input = input.transpose(2, 0, 1) # Change order to (channels, height, width)
input = np.expand_dims(input, 0) # Add dimension for batch: (batch, channels, height, width)
outputs = model.run(["output0"], {"images": input})
output = outputs[0]
threshold = 0.5
index = np.max(output[0, 4:, :], axis=0) >= threshold
boxes = output[0, :, index]
plt.imshow(img)
for box in boxes:
    drawbox(box)
plt.show()