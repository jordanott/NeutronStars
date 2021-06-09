import tensorflow as tf


def eos_m1_metric(y_true, y_pred):
    y_true = tf.gather(tf.transpose(y_true), [0])
    y_pred = tf.gather(tf.transpose(y_pred), [0])

    percent_error = (y_true - y_pred) / y_true
    return tf.math.reduce_mean(tf.math.abs(percent_error))


def eos_m2_metric(y_true, y_pred):
    y_true = tf.gather(tf.transpose(y_true), [1])
    y_pred = tf.gather(tf.transpose(y_pred), [1])

    percent_error = (y_true - y_pred) / y_true
    return tf.math.reduce_mean(tf.math.abs(percent_error))


if __name__ == '__main__':
    import numpy as np
    y_true = tf.convert_to_tensor(np.random.rand(100, 2))
    y_pred = tf.convert_to_tensor(np.random.rand(100, 2))

    print(eos_m1_metric(y_true, y_pred))
    print(eos_m2_metric(y_true, y_pred))
