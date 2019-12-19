import numpy as np
import os
import imutils
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "imgs")
model_dir = os.path.join(BASE_DIR, 'model')

caffeModel = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
prototextPath = os.path.join(model_dir, 'deploy.prototxt.txt')

net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)
cap = cv2.VideoCapture(0)

curr_id = 0
label_ids = {}
train_x = []
train_y = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if label_ids.get(label) is None:
                label_ids[label] = curr_id
                curr_id += 1
            image = cv2.imread(path)
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            confidence = detections[0, 0, 0, 2]
            if confidence >= 0.5:
                box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                # print(startX, startY, endX, endY)
                # text = "{:.2f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                pil_image = Image.open(path).convert("L")
                image_array = np.array(pil_image, "uint8")
                roi = image_array[startY:endY, startX:endX]
                train_x.append(roi)
                train_y.append(label_ids[label])
                print(file)
            # cv2.imshow("Output", image)
            # cv2.waitKey(0)

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    cv2.imshow("Face Identification", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
