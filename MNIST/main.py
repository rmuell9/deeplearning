import keras
import math
import matplotlib.pyplot as plt
from keras import layers
from keras.datasets import mnist
from keras import ops
from keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def show_image(i, train_images, train_labels):
    digit = train_images[i]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()

    print("Corresponding Train Label:", train_labels[i])


def one_training_step(model, images_batch, labels_batch):
    predictions = model(images_batch)  # Runs the forard pass

    loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss = ops.mean(loss)

    gradients = get_gradients_of_loss_wrt_weights(loss, model.weights)
    update_weights(gradients, model.weights)

    return loss

# Move the weights such that the loss is reduced
def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size),
            initializer="uniform"
        )
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")

    def __call__(self, inputs):
        # Applies the forward pass
        x = ops.matmul(inputs, self.W)
        x = x + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weighs
        return weights


# Iterate over MNIST data in mini-batches
class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def keras_imported():
    # Importing MNIST training and test data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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


def main():
    model = NaiveSequential(
        [
            NaiveDense(
                input_size=28*28,
                output_size=512,
                activation=ops.relu
            ),
            NaiveDense(
                input_size=512,
                output_size=10,
                activation=ops.softmax
            ),
        ]
    )


if __name__ == "__main__":
    main()
