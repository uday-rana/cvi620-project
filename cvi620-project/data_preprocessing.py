import os
import cv2 as cv
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split


def preprocess_data(dataset_path):
    print("Preprocessing the data...")

    # Load driving log and select needed columns
    df = read_csv(
        os.path.join(dataset_path, "driving_log.csv"),
        names=["center", "left", "right", "steering", "throttle", "brake", "speed"],
        usecols=["center", "steering"],
    )

    images, steerings = [], []

    for row in df.itertuples():
        if abs(row.steering) < 0.25 and np.random.rand() < 0.5:
            continue
        if abs(row.steering) < 0.1 and np.random.rand() < 0.3:
            continue

        steering = row.steering
        repeats = 1

        if abs(steering) > 0.4:
            repeats = 100
        elif abs(steering) > 0.3:
            repeats = 16
        elif abs(steering) > 0.15:
            repeats = 8

        for _ in range(repeats):
            # Get path to image
            image_path = os.path.normpath(
                os.path.join(dataset_path, "IMG", os.path.basename(row.center))
            )

            print(f"Reading {image_path}")
            img = cv.imread(image_path)

            img = img[60:135, :]
            img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
            img = cv.resize(img, (200, 66))
            img = img / 255

            images.append(img)
            steerings.append(row.steering)

    images = np.array(images)
    steerings = np.array(steerings)

    return train_test_split(images, steerings, test_size=0.2, shuffle=True)
