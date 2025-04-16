import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

def augment_data():
    print("In augment_data()")

    input_csv = "data/driving_log.csv"
    df = pd.read_csv(input_csv, header=None)

    images = []
    steerings = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        center_path, _, _, steering, *_ = row
        steering = float(steering)

        center_img = cv2.imread(center_path.strip())
        if center_img is None:
            continue

        # Preprocess image
        img = center_img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.resize(img, (200, 66))
        img = img / 255.0

        images.append(img)
        steerings.append(steering)

        # Flip
        if random.random() < 0.5:
            flipped = cv2.flip(img, 1)
            images.append(flipped)
            steerings.append(-steering)

        # Brightness adjustment
        if random.random() < 0.3:
            bright = center_img.copy()
            hsv = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.5, 1.5)
            bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bright = bright[60:135, :, :]
            bright = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)
            bright = cv2.resize(bright, (200, 66))
            bright = bright / 255.0
            images.append(bright)
            steerings.append(steering)

    return np.array(images), np.array(steerings)
