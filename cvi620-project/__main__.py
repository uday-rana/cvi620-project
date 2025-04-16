from data_augmentation import augment_data
from data_preprocessing import preprocess_data
from model_training import train_model
from training_visualization import visualize_training


def main():
    DATASET_PATH = "./data"

    print("Starting...")

    x_train, x_test, y_train, y_test = preprocess_data(DATASET_PATH)
    x_train, y_train = augment_data(x_train, y_train)
    H = train_model(x_train, x_test, y_train, y_test)
    visualize_training(H)


if __name__ == "__main__":
    main()
