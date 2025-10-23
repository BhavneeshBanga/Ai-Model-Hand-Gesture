import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math
import os

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
counter = 0

# Folder for saving images
folder = '/Users/bhavi/OneDrive/Desktop/Sign_Language_Detection/Data/'
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping remains within bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        # Aspect ratio calculations
        aspectratio = h / w

        if aspectratio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            wGap = math.ceil((imgsize - wCal) / 2)
            imgwhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            hGap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hGap:hGap + imgResize.shape[0], :] = imgResize

        # Show images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgwhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('d'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(f"Saved Image {counter}")

    elif key == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()



