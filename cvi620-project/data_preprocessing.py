import os
import cv2 as cv
import numpy as np
from pandas import read_csv


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
        # Keep ~50% of 0 steering values
        if abs(row.steering) < 0.01 and np.random.rand() < 0.5:
            continue

        # Get relative path to image
        image_path = os.path.join(
            dataset_path, *os.path.normpath(row.center).split(os.sep)[-2:]
        )

        # Replace backslashes with forward slashes
        image_path = image_path.replace("\\", "/")

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

    return images, steerings