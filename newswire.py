from keras.datasets import reuters
import keras
from keras import layers
from keras import Sequential
import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000
)


len(train_data)

len(test_data)

train_data[10]

# Decoding back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[10]]
)

train_labels[10]

# Encoding input data
x_train = imdb.multi_hot_encode(train_data, num_classes=10000)
x_test = imdb.multi_hot_encode(test_data, num_classes=10000)

x_train


# Can also use `from keras.utils import to_categorical`
def one_hot_encode(labels, num_classes=46):
    results = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results


y_train = one_hot_encode(train_labels)
y_test = one_hot_encode(test_labels)


model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),  # outputs a prob distribution
    ]
)

top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", top_3_accuracy],
)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)


loss = history.history["loss"]
loss

