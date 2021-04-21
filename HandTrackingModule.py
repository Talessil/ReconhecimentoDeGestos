import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:  # if detect hand
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMarks,
                                               self.mpHands.HAND_CONNECTIONS)  # draw points and connections
        return img

    def find_position(self, img, handNo=0, draw=True):

        land_mark_list = []
        if self.results.multi_hand_landmarks:  # if detect hand
            my_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):  # lm get landmark, id get landmark id
                # print(id,lm)
                h, w, c = img.shape  # heights, width, channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                land_mark_list.append([id, cx, cy])
                #if draw:
                #    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return land_mark_list

def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)  # capture webcam
    detector = HandDetector()
    while True:
        success, img = cap.read()  # get the image
        img = detector.find_hands(img)
        land_mark_list = detector.find_position(img)
        if len(land_mark_list) != 0:
            print(land_mark_list[5])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
