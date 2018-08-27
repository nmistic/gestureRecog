import cv2
from keras.models import load_model
import numpy as np
import os

model = load_model('gesturerecog.h5')


def main():
    emojis = get_emojis()
    cap = cv2.VideoCapture(0)

    x, y, w, h = 300, 50, 350, 350

    while cap.isOpened():

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))

        res = cv2.bitwise_and(img, img, mask=mask2)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)

        dilation = cv2.dilate(median, kernel_square, iterations=2)

        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)

        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]

        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        

def get_emojis():
    emojis_folder = 'hand_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    return emojis