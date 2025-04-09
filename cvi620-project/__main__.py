from data_augmentation import augment_data
from data_preprocessing import preprocess_data
from training import train
from inference import test

def main():
    print("In main()")
    augment_data()
    preprocess_data()
    train()
    test()

if __name__ == "__main__":
    main()
