import numpy as np
import os
import imutils
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, 'model')

caffeModel = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
prototextPath = os.path.join(model_dir, 'deploy.prototxt.txt')

net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)
cap = cv2.VideoCapture(0)

idToLabel = {}
with open("labels.pickle", "rb") as f:
    labelToId = pickle.load(f)
    idToLabel = {v: k for k, v in labelToId.items()}

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

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
