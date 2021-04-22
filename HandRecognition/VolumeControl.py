import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math


w, h = 640, 480                                                                 # camera heights, width
cap = cv2.VideoCapture(0)                                                       # capture the webcam
cap.set(3, w)
cap.set(4, h)
p_time = 0                                                                      # previous time (fps calculus)
c_time = 0                                                                      # current time (fps calculus)
bar = 400                                                                       # volume bar initial position
per = 0                                                                         # initial percentage value
detector = htm.HandDetector(detectionCon=0.7)                                   # HandDetector object

while True:
    success, img = cap.read()                                                   # get the image
    img = detector.find_hands(img)                                              # get hand
    land_mark_list = detector.find_position(img)                                # get finger positions
    if len(land_mark_list) != 0:

        x1, y1 = land_mark_list[4][1], land_mark_list[4][2]                     # get finger 4 (thumb)
        x2, y2 = land_mark_list[8][1], land_mark_list[8][2]                     # get finger 8 (index)
        length = math.hypot(x2-x1, y2-y1)                                       # get length from finger 4 to finger 8

        # Hand range defined from 50 to 300
        bar = np.interp(length, [50, 300], [400, 150])                          # conversion (50,300) to (400,150)
        per = np.interp(length, [50, 300], [0, 100])                            # conversion (50,300) to (0,100)

    cv2.rectangle(img, (50, 150), (100, 400), (0, 0, 255), 2)                    # draw volume bar
    cv2.rectangle(img, (50, int(bar)), (100, 400), (0, 0, 255), cv2.FILLED)      # fill the bar
    cv2.putText(img, f' {int(per)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN,       # draw the bar percentage
                1.5, (0, 0, 255), 2)

    # draw fds value
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN,
                1.5, (0, 255, 255), 2)

    cv2.imshow("Image", img)                                                    # show image
    cv2.waitKey(1)                                                              # 1 ms delay
