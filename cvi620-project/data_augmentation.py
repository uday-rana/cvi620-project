import os
import pandas as pd
import cv2
import numpy as np
import random
from tqdm import tqdm

def augment_data():
    print("In augment_data()")

    input_csv = '../data/forward/driving_log.csv'
    output_dir = '../data/augmented/IMG'
    output_csv = '../data/augmented/driving_log_augmented.csv'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_csv, header=None)
    augmented_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row[0]
        steering = float(row[3])
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = os.path.basename(img_path)
        new_img_path = os.path.join(output_dir, base_name)
        cv2.imwrite(new_img_path, img)
        augmented_rows.append([new_img_path, steering])

        # Data Augmentation - Flip
        if random.random() < 0.5:
            flipped = cv2.flip(img, 1)
            flipped_name = f'flip_{base_name}'
            flipped_path = os.path.join(output_dir, flipped_name)
            cv2.imwrite(flipped_path, flipped)
            augmented_rows.append([flipped_path, -steering])

        # Brightness
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.5, 1.2)
            bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bright_name = f'bright_{base_name}'
            bright_path = os.path.join(output_dir, bright_name)
            cv2.imwrite(bright_path, bright)
            augmented_rows.append([bright_path, steering])

        # Rotation
        if random.random() < 0.3:
            h, w = img.shape[:2]
            angle = random.uniform(-10, 10)
            rot_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            rotated = cv2.warpAffine(img, rot_matrix, (w, h))
            rot_name = f'rot_{base_name}'
            rot_path = os.path.join(output_dir, rot_name)
            cv2.imwrite(rot_path, rotated)
            augmented_rows.append([rot_path, steering])

    # Save new CSV
    pd.DataFrame(augmented_rows).to_csv(output_csv, index=False, header=False)
    print(f"Saved augmented data to {output_csv}")
