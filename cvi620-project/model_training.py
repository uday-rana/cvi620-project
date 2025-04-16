from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam


def train_model(x_train, x_test, y_train, y_test):
    MODEL_FILENAME = "model.h5"

    print("Training the neural network...")

    net = Sequential(
        [
            Conv2D(24, (5, 5), activation="relu", input_shape=(66, 200, 3)),
            MaxPool2D((2, 2)),
            Conv2D(36, (5, 5), activation="relu"),
            MaxPool2D((2, 2)),
            Conv2D(48, (5, 5), activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(1164, activation="relu"),
            Dense(100, activation="relu"),
            Dense(50, activation="relu"),
            Dense(10, activation="relu"),
            Dense(1),
        ]
    )

    opt = Adam(learning_rate=0.0001)

    net.compile(optimizer=opt, loss="mse")

    net.summary()

    H = net.fit(
        x_train,
        y_train,
        batch_size=16,
        validation_data=(x_test, y_test),
        epochs=10,
    )

    net.save(MODEL_FILENAME)
    print(f"Saved model to {MODEL_FILENAME}")

    return H
