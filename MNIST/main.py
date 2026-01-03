import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.datasets import mnist

# Importing MNIST training and test data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
#
# print(train_labels[4])

# Network architecture
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Preparing the image data
train_images = train_images.reshape((60_000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10_000, 28 * 28))
test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Making predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print("Predictions:\n", predictions[0])
print(predictions[0].argmax())
print(predictions[0][7])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accurary:", test_acc)
