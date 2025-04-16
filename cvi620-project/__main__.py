from data_augmentation import augment_data
from data_preprocessing import preprocess_data
from dataset_batching import batch_dataset
from model_training import train_model
from training_visualization import visualize_training

def main():
    DATASET_PATH = "./data"

    print("Starting...")

    images, steerings = preprocess_data(DATASET_PATH)
    aug_images, aug_steerings = augment_data(images, steerings)

    batch_dataset(aug_images, aug_steerings)
    train_model()
    visualize_training()


if __name__ == "__main__":
    main()
