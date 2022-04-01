import cv2
import numpy as np
import d3dshot

d = d3dshot.create(capture_output="numpy")
d.display = d.displays[0]

while True:

    # capturing the frame
    frame = d.screenshot(region=(585, 532, 1323, 939))

    # masking the color white
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensitivity = 50
    lower_white = np.array([0, 0, 255 - sensitivity], dtype=np.uint8)
    upper_white = np.array([255, sensitivity, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # finding the lines and drawing them on the frame
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 5
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(gray2, 0.8, line_image, 1, 0)

    # showing the result
    cv2.imshow('CV View', res)
    cv2.imshow('CV View2', lines_edges)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
