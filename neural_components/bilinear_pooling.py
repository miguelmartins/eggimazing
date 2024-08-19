import tensorflow as tf


def signed_sqrt(x):
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + tf.keras.backend.epsilon())


def L2_norm(x, axis=-1):
    return tf.keras.backend.l2_normalize(x, axis=axis)


def bilinear_pooling(a, b):
    blp = tf.matmul(a, b, transpose_a=True)
    blp = tf.keras.layers.Flatten()(blp)
    blp = tf.keras.layers.Lambda(signed_sqrt)(blp)
    blp = tf.keras.layers.Lambda(L2_norm)(blp)
    return blp
