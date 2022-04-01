import keyboard
import cv2
import numpy as np
import pyautogui
import time
import d3dshot


d = d3dshot.create(capture_output="numpy")
d.display = d.displays[0]


while True:

    frame = d.screenshot(region=(585, 532, 1323, 939))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensitivity = 50

    lower_white = np.array([0, 0, 255 - sensitivity], dtype=np.uint8)
    upper_white = np.array([255, sensitivity, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('CV View', res)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
