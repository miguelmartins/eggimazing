import tensorflow as tf


def basic_augmentation(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def basic_plus_color_augmentation(x):
    x = basic_augmentation(x)
    x = tf.image.random_contrast(x, 0.5, 1.0)  # the closer it is to 1.0, the less aggressive it is
    return x
