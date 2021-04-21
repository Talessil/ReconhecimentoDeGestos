#---------------------------------------------------------
# Hand Tracking Module
# Author: Tales Lopes
# email: talessil.sil@gmail.com
#---------------------------------------------------------

import cv2
import mediapipe as mp
import time

# define hand detector class
class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # get hand
    def find_hands(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        return img

    # get finger positions
    def find_position(self, img, handNo=0, draw=True):
        land_mark_list = []
        if self.results.multi_hand_landmarks:                               # if detect hand
            my_hand = self.results.multi_hand_landmarks[handNo]
            for lm_id, lm in enumerate(my_hand.landmark):                   # lm get landmark, lm_id get landmark id
                h, w, c = img.shape                                         # heights, width, channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                land_mark_list.append([lm_id, cx, cy])
        return land_mark_list
