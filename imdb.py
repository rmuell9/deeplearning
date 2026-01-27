import numpy as np
from keras.datasets import imdb

import numpy as np

def multi_hot_encode(sequences, num_classes):
    # Create a zero matrix with shape (number of sequences, number of classes)
    # Each row will represent one sequence's multi-hot encoding
    results = np.zeros((len(sequences), num_classes))
    
    # Iterate through each sequence with its index
    for i, sequence in enumerate(sequences):
        # Set positions corresponding to indices in the sequence to 1.0
        # This creates a multi-hot vector where multiple positions can be "hot"
        # Example: if sequence=[1,3,5] and num_classes=7, row becomes
        # [0, 1, 0, 1, 0, 1, 0]
        results[i][sequence] = 1.0

    return results

if __name__ == "__main__":

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000
    )

    train_data[0]

    max([max(sequence) for sequence in train_data])

    word_index = imdb.get_word_index()

    reverse_word_index = dict([(value, key) for (key, value) in 
        word_index.items()])
    decoded_review = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
    )


    decoded_review[:100]


    x_train = multi_hot_encode(train_data, num_classes=10000)
    x_test = multi_hot_encode(test_data, num_classes=10000)


    x_train[0]

    y_train = train_labels.astype("float32")
    y_test = test_labels.astype("float32")


    import keras
    from keras import layers

    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]


# Training the model
    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )



# Viewing what happened during training
    history_dict = history.history
    history_dict.keys()

# Plotting Training and Val. Loss
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "r--", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("[IMDB] Training and validation loss")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Plotting the training and val. accuracy
    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "r--", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("[IMDB] Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)


    results


    model.predict(x_test)
