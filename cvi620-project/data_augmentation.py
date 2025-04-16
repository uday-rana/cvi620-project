import numpy as np
import cv2
import random

def augment_data(images, steerings):
    print("Augmenting the data...")

    aug_images = []
    aug_steerings = []

    for img, steering in zip(images, steerings):
        aug_images.append(img)
        aug_steerings.append(steering)

        # Flip horizontally
        if random.random() < 0.5:
            flipped = cv2.flip(img, 1)
            aug_images.append(flipped)
            aug_steerings.append(-steering)

        # Brightness adjustment (in HSV)
        if random.random() < 0.3:
            bright = (img * 255).astype(np.uint8)
            bright = cv2.cvtColor(bright, cv2.COLOR_YUV2BGR)
            hsv = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.5, 1.5)
            hsv = np.clip(hsv, 0, 255)
            bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bright = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)
            bright = bright.astype(np.float32) / 255.0
            aug_images.append(bright)
            aug_steerings.append(steering)

    return np.array(aug_images), np.array(aug_steerings)
