import numpy as np
import cv2
import random


def augment_data(images, steerings):
    print("Augmenting the training data...")

    aug_images = []
    aug_steerings = []

    for img, steering in zip(images, steerings):
        # Keep original images
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

        # Zoom
        if random.random() < 0.3:
            zoom_factor = random.uniform(1.0, 1.2)  # only zoom in
            h, w = img.shape[:2]
            zh, zw = int(h * zoom_factor), int(w * zoom_factor)
            zoomed = cv2.resize(img, (zw, zh))

            start_y = (zh - h) // 2
            start_x = (zw - w) // 2
            zoomed = zoomed[start_y : start_y + h, start_x : start_x + w]

            aug_images.append(zoomed)
            aug_steerings.append(steering)

        # Shift
        if random.random() < 0.3:
            dx = random.randint(-20, 20)
            dy = random.randint(-10, 10)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            aug_images.append(shifted)
            aug_steerings.append(steering + dx * 0.002)

        # Rotate
        if random.random() < 0.2:
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D(
                (img.shape[1] // 2, img.shape[0] // 2), angle, 1
            )
            rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            aug_images.append(rotated)
            aug_steerings.append(steering + angle * 0.01)

    return np.array(aug_images), np.array(aug_steerings)
