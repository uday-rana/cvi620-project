from data_augmentation import augment_data
from data_preprocessing import preprocess_data
from dataset_batching import batch_dataset
from model_training import train_model
from training_visualization import visualize_training


def main():
    DATASET_PATH = "./data"

    print("Starting...")

    augment_data()
    preprocess_data(DATASET_PATH)
    batch_dataset()
    train_model()
    visualize_training()


if __name__ == "__main__":
    main()
