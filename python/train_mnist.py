import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

print("Downloading MNIST dataset (Keras)...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Starting MNIST training (GPU)...")
start_time = time.time()

model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=2)

elapsed = time.time() - start_time
print(f"Training time MNIST: {elapsed:.2f} seconds")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
