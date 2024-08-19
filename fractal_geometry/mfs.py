import tensorflow as tf

from fractal_geometry.differential_blocks import HolderExponentsLayer, LeastSquaresFittingLayer, SoftGroupAssignment, \
    soft_gliding_boxes, SoftGlidingBoxesLayer
from neural_components.custom_ops import sample_min_max_scaling


def fractal_encoding_preprocessing(x, n_out_channnels):
    x = tf.keras.layers.Conv2D(filters=n_out_channnels, kernel_size=(1, 1), strides=(1, 1), activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x


def fe_upscale(x, n_filters):
    return tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           groups=n_filters)(x)


def get_mfs(x, borel_scales=6, global_scales=16):
    x = HolderExponentsLayer(borel_scales)(x)
    x = LeastSquaresFittingLayer(borel_scales)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = SoftGroupAssignment(global_scales)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = soft_gliding_boxes(x, borel_scales)
    x = LeastSquaresFittingLayer(borel_scales)(x)
    return x


class MultiFractalSpectrumLayer(tf.keras.layers.Layer):
    def __init__(self, local_scale, global_scale, trainable=True, **kwargs):
        self.local_scale = local_scale
        self.global_scale = global_scale
        self.holder = HolderExponentsLayer(self.local_scale, trainable)
        self.lsf = LeastSquaresFittingLayer(self.local_scale)
        self.holder_bn = tf.keras.layers.BatchNormalization()
        self.sga = SoftGroupAssignment(self.global_scale)
        self.sga_bn = tf.keras.layers.BatchNormalization()
        self.soft_boxes = SoftGlidingBoxesLayer(self.local_scale)
        super(MultiFractalSpectrumLayer, self).__init__(**kwargs)

    def call(self, x):
        x = sample_min_max_scaling(x)
        x = self.holder(x)
        x = self.lsf(x)
        x = self.holder_bn(x)
        x = self.sga(x)
        x = self.sga_bn(x)
        x = self.soft_boxes(x)
        x = -self.lsf(x)
        return x


class PointSetLayer(tf.keras.layers.Layer):
    def __init__(self, local_scale, global_scale, trainable=True, **kwargs):
        self.local_scale = local_scale
        self.global_scale = global_scale
        self.holder = HolderExponentsLayer(self.local_scale, trainable)
        self.lsf = LeastSquaresFittingLayer(self.local_scale)
        self.holder_bn = tf.keras.layers.BatchNormalization()
        self.sga = SoftGroupAssignment(self.global_scale)
        self.sga_bn = tf.keras.layers.BatchNormalization()
        super(PointSetLayer, self).__init__(**kwargs)

    def call(self, x):
        x = sample_min_max_scaling(x)
        x = self.holder(x)
        x = self.lsf(x)
        x = self.holder_bn(x)
        x = self.sga(x)
        x = self.sga_bn(x)
        return x


class HolderRegularityLayer(tf.keras.layers.Layer):
    def __init__(self, local_scale, trainable=True, **kwargs):
        self.local_scale = local_scale
        self.holder = HolderExponentsLayer(self.local_scale, trainable)
        self.lsf = LeastSquaresFittingLayer(self.local_scale)
        self.holder_bn = tf.keras.layers.BatchNormalization()
        super(HolderRegularityLayer, self).__init__(**kwargs)

    def call(self, x):
        x = sample_min_max_scaling(x)
        x = self.holder(x)
        x = self.lsf(x)
        x = self.holder_bn(x)
        return x
