import os
import cv2
import csv
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

def augment_data():
    print("In augment_data()")

    # Input and output paths
    input_csv = "../data/forward/driving_log.csv"
    output_dir = "../data/augmented/IMG"
    output_csv = "../data/augmented/driving_log_augmented.csv"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the driving log CSV (no header in this file)
    df = pd.read_csv(input_csv, header=None)

    augmented_rows = []

    # Iterate through each row of the dataset
    for index, row in tqdm(df.iterrows(), total=len(df)):
        center_path, left_path, right_path, steering, throttle, brake, speed = row

        # Read the center camera image
        center_img = cv2.imread(center_path.strip())
        if center_img is None:
            continue  # Skip if the image can't be read

        # Save the original image to output directory
        filename = os.path.basename(center_path)
        new_path = os.path.join(output_dir, filename)
        cv2.imwrite(new_path, center_img)
        augmented_rows.append([new_path, steering, throttle, brake, speed])

        steering = float(steering)

        # Randomly apply horizontal flipping
        if random.random() < 0.5:
            flipped_img = cv2.flip(center_img, 1)
            flipped_filename = "flipped_" + filename
            flipped_path = os.path.join(output_dir, flipped_filename)
            cv2.imwrite(flipped_path, flipped_img)
            augmented_rows.append([flipped_path, -steering, throttle, brake, speed])

        # Randomly apply brightness adjustment
        if random.random() < 0.3:
            hsv = cv2.cvtColor(center_img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.5, 1.5)  # Change brightness
            bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bright_filename = "bright_" + filename
            bright_path = os.path.join(output_dir, bright_filename)
            cv2.imwrite(bright_path, bright_img)
            augmented_rows.append([bright_path, steering, throttle, brake, speed])

    # Save the new augmented dataset
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(augmented_rows)

    print(f"Saved augmented data to {output_csv}")