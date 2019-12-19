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
labelToId = {}
idToLabel = {}
train_x = []
train_y = []
recognizer = cv2.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()
            idToLabel[curr_id] = label
            if labelToId.get(label) is None:
                labelToId[label] = curr_id
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
                pil_image = Image.open(path).convert("L")
                # final_image = pil_image.resize((150, 150), Image.LANCZOS)
                image_array = np.array(pil_image, "uint8")
                roi = image_array[startY:endY, startX:endX]
                train_x.append(roi)
                train_y.append(labelToId[label])

recognizer.train(train_x, np.array(train_y))

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=700)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            roi_gray = gray[startY:endY, startX:endX]
            pred = recognizer.predict(roi_gray)
            pred_label = idToLabel[pred[0]]
            pred_confidence = pred[1]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
            if pred_confidence >= 0.5:
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, pred_label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Face Identification", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
