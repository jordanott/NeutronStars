import tensorflow as tf


AVAILABLE_ACTIVATIONS = {
    'elu': tf.nn.elu,
    'relu': tf.nn.relu,
    'leaky_relu': tf.nn.leaky_relu,
}
