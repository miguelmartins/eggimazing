import tensorflow as tf
from tqdm import tqdm

from fractal_geometry.differential_blocks import LeastSquaresFittingLayer
from neural_components.custom_ops import MinPooling3DWrapper


def soft_box_counting_3d(x, max_scale):
    layer_scale = x.shape[3]
    n_boxes = []
    for scale in range(2, max_scale):
        # SAME padding to ensure no elements are dropped and that grid "divides" H, W, and T
        g_max = tf.keras.layers.MaxPooling3D(pool_size=(scale, scale, layer_scale),
                                             strides=(scale, scale, layer_scale),
                                             padding='SAME')(x) / scale
        g_min = MinPooling3DWrapper(x, pool_size=(scale, scale, layer_scale),
                                    strides=(scale, scale, layer_scale),
                                    padding='SAME') / scale
        n_boxes.append(tf.reduce_sum(g_max - g_min + 1, axis=[1, 2, 3]))
    return tf.stack(n_boxes, axis=-1)


def sliding_box_counting_dimension(x, box_x, box_y, box_z, max_scale):
    batch, h, w, t, c = tf.shape(x)  # this might not work with tf.function, maybe try tf.unstack
    # TODO: inquire why not use local patch statistics instead of global.
    spatial_dim_min = tf.cast(tf.reduce_min([h, w]), dtype=tf.float32)
    # Max for each image in batch, dimensions adjusted for broadcast
    i_max = tf.reduce_max(
        tf.reduce_max(
            tf.reduce_max(x, axis=1),
            axis=1),
        axis=1)[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    hs = i_max / spatial_dim_min
    x = x / hs  # we will later divide by s in each box count so that hs <- shs

    volumes = tf.extract_volume_patches(x,
                                        ksizes=[1, box_x, box_y, box_z, 1],
                                        strides=[1, 1, 1, box_z, 1],
                                        padding='SAME')  # note that same padding forces one to use box_z in stride
    # We collapse spatial axis to the batch dimension to take advantage of native tensorflow vectorization operations
    volumes = tf.reshape(volumes, [batch * h * w, box_x, box_y, box_z, c])
    measures = soft_box_counting_3d(volumes, max_scale)
    measures = tf.reshape(measures, [batch, h, w, c, max_scale - 2])
    lfd = -LeastSquaresFittingLayer(max_scale - 2)(measures)  # N_boxes \propto L ** -D
    return lfd


def layer_normalization(in_layer, bottom_layers, channel_norm):
    channel_normalizing_filters = [tf.keras.layers.Conv2D(channel_norm, kernel_size=(1, 1))
                                   for _ in range(len(bottom_layers) + 1)]
    spatial_normalizing_filters = [tf.keras.layers.UpSampling2D(size=(2, 2))
                                   for _ in range(len(bottom_layers))]

    in_layer = channel_normalizing_filters[0](in_layer)
    bottom_layers = [channel_normalizing_filters[i + 1](bottom_layer)
                     for (i, bottom_layer) in enumerate(bottom_layers)]

    bottom_layer = [spatial_normalizing_filters[i](bottom_layers[i])
                    for i in range(len(bottom_layers))]
    return in_layer, bottom_layer


def layer_normalization_once(layers, channel_norm):
    # Assume layers is sorted biggest to smallest so that [x1, x2, ..., xn], xi > xi+1 for all i
    channel_normalizing_filters = [tf.keras.layers.Conv2D(channel_norm, kernel_size=(1, 1))
                                   for _ in range(len(layers))]
    spatial_normalizing_filters = [tf.keras.layers.UpSampling2D(size=(2 ** i, 2 ** i))
                                   for i in range(1, len(layers))]
    color_norm_layers = [channel_normalizing_filters[i](layer)
                         for (i, layer) in enumerate(layers)]
    spatial_norm_layers = [spatial_normalizing_filters[i](color_norm_layers[i + 1])
                           for i in range(len(spatial_normalizing_filters))]
    return tf.stack([color_norm_layers[0]] + spatial_norm_layers, axis=-2)

