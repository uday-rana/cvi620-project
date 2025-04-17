import numpy as np
import cv2


def augment_data(images, steerings):
    print("Augmenting the training data...")

    aug_images = []
    aug_steerings = []

    for img, steering in zip(images, steerings):
        # Keep original images
        aug_images.append(img)
        aug_steerings.append(steering)

        # Augment 20% of the dataset
        if np.random.randint(1, 5) == 1:
            h, w = img.shape[:2]

            # Flip horizontally
            aug = cv2.flip(img, 1)
            aug_steering = -steering

            # Brightness adjustment (in HSV)
            bright = (aug * 255).astype(np.uint8)
            bright = cv2.cvtColor(bright, cv2.COLOR_YUV2BGR)
            hsv = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] *= np.random.uniform(0.5, 1.5)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            aug = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)
            aug = aug.astype(np.float32) / 255.0

            # Zoom
            zoom_factor = np.random.uniform(1.0, 1.2)  # only zoom in
            zh, zw = int(h * zoom_factor), int(w * zoom_factor)
            aug = cv2.resize(aug, (zw, zh))
            start_y = (zh - h) // 2
            start_x = (zw - w) // 2
            aug = aug[start_y : start_y + h, start_x : start_x + w]

            # Shift
            dx = np.random.randint(-30, 30)
            dy = np.random.randint(-10, 10)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aug = cv2.warpAffine(aug, M, (w, h))
            aug_steering += dx * 0.003

            # Rotate
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            aug = cv2.warpAffine(aug, M, (w, h))
            aug_steering += angle * 0.01

            aug_images.append(aug)
            aug_steerings.append(aug_steering)

    return np.array(aug_images), np.array(aug_steerings)
