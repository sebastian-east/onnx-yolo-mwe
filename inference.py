import onnxruntime as ort
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

NMS_THRESHOLD = 0.9

def intersection(box1, box2):
    xc1, yc1, w1, h1, _ = box1
    xc2, yc2, w2, h2, _ = box2
    xa1 = xc1 - w1/2
    xa2 = xc2 - w2/2
    xb1 = xc1 + w1/2
    xb2 = xc2 + w2/2
    ya1 = yc1 - h1/2
    ya2 = yc2 - h2/2
    yb1 = yc1 + h1/2
    yb2 = yc2 + h2/2
    xa = max(xa1, xa2)
    xb = min(xb1, xb2)
    ya = max(ya1, ya2)
    yb = min(yb1, yb2)
    w = xb - xa
    h = yb - ya
    if (w > 0) and (h > 0):
        return w * h
    else:
        return 0

def boxSize(box):
    return box[2] * box[3]

def iou(box1, box2):
    i = intersection(box1, box2)
    u = boxSize(box1) + boxSize(box2) - i
    return i / u

def nonMaximumSuppression(boxes):
    if len(boxes) > 0:
        boxes = [b[:4] + [max(b[4:])] for b in boxes]
        boxes.sort(key=lambda x : x[4])
        newBoxes = [boxes.pop()]
        while len(boxes) > 0:
            box = boxes.pop()
            should_include = True
            for newBox in newBoxes:
                if iou(box, newBox) > NMS_THRESHOLD:
                    should_include = False
                    break
            if should_include:
                newBoxes.append(box)

        return newBoxes
    else:
        return boxes


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
boxes = output[0, :, index].tolist()
boxes = nonMaximumSuppression(boxes)
plt.imshow(img)
for box in boxes:
    drawbox(box)
plt.show()