import matplotlib.pyplot as plt


def visualize_training(h):
    print("Visualizing the training...")

    plt.figure()
    plt.plot(h.history["loss"], label="Training loss")
    plt.plot(h.history["val_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Evaluation")
    plt.legend()
    plt.show()
