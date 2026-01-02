import keras
from keras import layers
from keras.datasets import mnist

# Importing MNIST training and test data
(train_images, train_labels), (test_images, test_label) = mnist.load_data()

# Network architecture
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
