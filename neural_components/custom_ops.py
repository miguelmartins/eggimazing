import tensorflow as tf


def standardize_values(x, target_min=0., target_max=1.):
    # Good explanation in:
    # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    x_min = tf.reduce_min(x, axis=-1)[..., tf.newaxis]
    x_max = tf.reduce_max(x, axis=-1)[..., tf.newaxis]
    r_min = tf.ones_like(x_min) * target_min
    r_max = tf.ones_like(x_max) * target_max
    scaled_x = (x - x_min) / (x_max - x_min)
    return (scaled_x * (r_max - r_min)) + r_min


def min_pool(x, *, size, stride, padding):
    n_channel = x.shape[-1]
    patches = tf.image.extract_patches(images=x,
                                       sizes=[1, size, size, 1],
                                       strides=[1, stride, stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding=padding)
    channel_pool = [tf.reduce_min(patches[:, :, :, c::n_channel], keepdims=True, axis=-1) for c in range(n_channel)]
    return tf.concat(channel_pool, axis=-1)


def Conv2DBatchNorm(x, activation_fn=tf.nn.relu, *args, **kwargs):
    # Note that this function must be called using activation_fn as a positional argument, else
    # python will duplicate it in **kwargs
    x = tf.keras.layers.Conv2D(activation=None, *args, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return activation_fn(x)


class SharedConv2D(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SharedConv2D, self).__init__()
        self.shared_conv = tf.keras.layers.Conv2D(*args, **kwargs)
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)

    def build(self, input_shape):
        _input_shape = list(input_shape[:-1]) + [1]
        self.shared_conv.build(_input_shape)
        super(SharedConv2D, self).build(input_shape)

    def call(self, x):
        num_channels = x.shape[-1]
        return self.concat_layer([self.shared_conv(x[..., i][..., tf.newaxis]) for i in range(num_channels)])


def MaxOut(x, y):
    x_ = x[..., tf.newaxis, :]
    y_ = y[..., tf.newaxis, :]
    return tf.reduce_max(tf.keras.layers.Concatenate(axis=-2)([x_, y_]), axis=-2)


def MinPooling2DWrapper(x, *args, **kwargs):
    return -tf.keras.layers.MaxPooling2D(*args, **kwargs)(-x)


def MinPooling3DWrapper(x, *args, **kwargs):
    return -tf.keras.layers.MaxPooling3D(*args, **kwargs)(-x)


def sample_min_max_scaling(x, minimum_value=1, maximum_value=256):
    min_ = tf.reduce_min(x, axis=[1, 2])[:, tf.newaxis, tf.newaxis, :]
    max_ = tf.reduce_max(x, axis=[1, 2])[:, tf.newaxis, tf.newaxis, :]
    denominator_ = tf.clip_by_value(max_ - min_,
                                    clip_value_min=10e-8,
                                    clip_value_max=10e8)  # Following original implementation
    return minimum_value + (((x - min_) / denominator_) * (maximum_value - minimum_value))
