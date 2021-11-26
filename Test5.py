import numpy as np
import cv2
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0) # '0' for webcam

while video.isOpened():
    _, frame = video.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Frame", frame)
video.release()
cv2.destroyAllWindows()