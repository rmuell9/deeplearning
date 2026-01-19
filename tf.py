import tensorflow as tf
import numpy as np

a = tf.ones((2, 2))
print(a)

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)

# Immutable
x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)  # Random tensor
x

# Immutable
x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
x

# State of the tensor becomes mutable
v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
v



input_dim = 2 
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


def model(inputs, W, b):
    return tf.matmul(inputs, W) + b


def mean_squared_error(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

learning_rate = 0.1


@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(predictions, targets)
        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
        W.assign_sub(grad_loss_wrt_W * learning_rate)
        b.assign_sub(grad_loss_wrt_b * learning_rate)
        return loss
