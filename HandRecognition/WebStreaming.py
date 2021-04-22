import HandTrackingModule as htm
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import math
import numpy as np

w, h = 640, 480  # camera heights, width
cap = cv2.VideoCapture(0)  # capture the webcam
cap.set(3, w)
cap.set(4, h)
p_time = 0  # previous time (fps calculus)
c_time = 0  # current time (fps calculus)
bar = 400  # volume bar initial position
per = 0  # initial percentage value
detector = htm.HandDetector(detectionCon=0.7)  # HandDetector object

app = Flask(__name__)  # initialize a flask object


@app.route("/")
def index():
    return render_template("index.html")  # return the rendered template


def get_frame():
    global per, bar
    success, img = cap.read()  # get the image
    img = detector.find_hands(img)  # get hand
    land_mark_list = detector.find_position(img)  # get finger positions
    if len(land_mark_list) != 0:
        x1, y1 = land_mark_list[4][1], land_mark_list[4][2]  # get finger 4 (thumb)
        x2, y2 = land_mark_list[8][1], land_mark_list[8][2]  # get finger 8 (index)
        length = math.hypot(x2 - x1, y2 - y1)  # get length from finger 4 to finger 8

        # Hand range defined from 50 to 300
        bar = np.interp(length, [50, 300], [400, 150])  # conversion (50,300) to (400,150)
        per = np.interp(length, [50, 300], [0, 100])  # conversion (50,300) to (0,100)

    cv2.rectangle(img, (50, 150), (100, 400), (0, 0, 255), 2)  # draw volume bar
    cv2.rectangle(img, (50, int(bar)), (100, 400), (0, 0, 255), cv2.FILLED)  # fill the bar
    cv2.putText(img, f' {int(per)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN,  # draw the bar percentage
                1.5, (0, 0, 255), 2)
    return img


def generate():
    while True:
        data = get_frame()
        flag, encoded_image = cv2.imencode(".jpg", data)  # encode the frame in JPEG format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=generate, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
