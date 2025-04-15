from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
import numpy as np
import os

def augment_with_generator(input_csv, output_csv, image_dir):
    df = pd.read_csv(input_csv, header=None)
    augmented_rows = []

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    os.makedirs(image_dir, exist_ok=True)

    for index, row in df.iterrows():
        image_path = row[0].strip()
        steering, throttle, brake, speed = row[1:]

        image = cv2.imread(image_path)
        if image is None:
            continue

        image = np.expand_dims(image, axis=0)  # Convert to 4D array (1, H, W, C)
        aug_iter = datagen.flow(image, batch_size=1)

        # Save original image to the output directory
        filename = os.path.basename(image_path)
        new_path = os.path.join(image_dir, filename)
        cv2.imwrite(new_path, image[0])
        augmented_rows.append([new_path, steering, throttle, brake, speed])

        # Save augmented images to the output directory
        for i in range(2):  # Generate 2 augmented versions per image
            aug_image = next(aug_iter)[0].astype(np.uint8)
            aug_filename = f"aug_{i}_{filename}"
            aug_path = os.path.join(image_dir, aug_filename)
            cv2.imwrite(aug_path, aug_image)
            augmented_rows.append([aug_path, steering, throttle, brake, speed])

    pd.DataFrame(augmented_rows).to_csv(output_csv, index=False, header=False)
    print(f"Saved augmented images and CSV to: {output_csv}")
