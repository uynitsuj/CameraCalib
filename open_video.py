#!/usr/bin/env python

import cv2
import numpy as np
import glob

# Start video capture from the camera
cap = cv2.VideoCapture(0)
w = 640
h = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

mtx = np.array([[562.4043074,    0.,         291.73285431],
 [  0.,         561.35411997, 250.70510786],
 [  0.,           0.,           1.        ]])
dist = np.array([[-0.40886105,  0.23773848, -0.00053296,  0.00061475, -0.0904396 ]])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))




if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    img = dst[y:y+h, x:x+w]
    cv2.imshow('img',img)
    print(img.shape)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
