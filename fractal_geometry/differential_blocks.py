import tensorflow as tf
from tensorflow.keras import backend as K

from neural_components.custom_ops import min_pool, MinPooling2DWrapper, standardize_values, SharedConv2D

EPSILON = 1e-12


def soft_gliding_boxes(x, max_scale):
    # Calculate FD given a soft membership tensor E[C,K] in [B, H, W, C, K]
    membership_map = []
    for scale in range(1, max_scale + 1):
        n_boxes = []
        for level_set in range(max_scale):
            boxes = tf.reduce_sum(tf.keras.layers.MaxPooling2D(pool_size=(scale, scale),
                                                               strides=(1, 1),
                                                               padding='valid')(x[..., level_set]), axis=[2, 1])
            n_boxes.append(boxes)
        membership_map.append(tf.stack(n_boxes, axis=-1))
    return tf.nn.relu(tf.stack(membership_map, axis=-1)) + 1


class SoftGlidingBoxesLayerDepracated(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        self.max_scale = max_scale
        self.pool_list = [tf.keras.layers.MaxPooling2D(pool_size=(scale, scale),
                                                       strides=(1, 1),
                                                       padding='valid') for scale in range(1, max_scale + 1)]
        super(SoftGlidingBoxesLayerDepracated, self).__init__(**kwargs)

    def call(self, x):
        membership_map = []
        for scale in range(1, self.max_scale + 1):
            n_boxes = []
            for level_set in range(self.max_scale):
                boxes = tf.reduce_sum(self.pool_list[scale - 1](x[..., level_set]), axis=[2, 1])
                n_boxes.append(boxes)
            membership_map.append(tf.stack(n_boxes, axis=-1))
        return tf.nn.relu(tf.stack(membership_map, axis=-1)) + 1


class SoftGlidingBoxesLayer(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        self.max_scale = max_scale
        self.pool_list = [tf.keras.layers.MaxPooling3D(pool_size=(scale, scale, 1),
                                                       strides=(1, 1, 1)) for scale in range(1, max_scale + 1)]
        super(SoftGlidingBoxesLayer, self).__init__(**kwargs)

    def call(self, x):
        boxes_at_scale = []
        for scale in range(1, self.max_scale + 1):
            n_boxes = tf.reduce_sum(self.pool_list[scale - 1](x), axis=[2, 1])
            boxes_at_scale.append(n_boxes)
        return tf.nn.relu(tf.stack(boxes_at_scale, axis=-1)) + 1


class DepthwiseMeasurementsAtScale(tf.keras.layers.Layer):
    """
    Each feature map is convolved with an independent filter for each scale.
    This means that different feature maps will be convolved with different Gr at each r-measurements
    """

    def __init__(self, max_scale, kernel_init, dilation_rate=(1, 1), **kwargs):
        super(DepthwiseMeasurementsAtScale, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.kernel_init = kernel_init
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.op_list = []
        for scale in range(1, self.max_scale + 1):
            self.op_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(scale, scale),
                                                                strides=(1, 1),
                                                                padding="same",
                                                                # same padding ensures that all points are a center of a disk
                                                                activation='relu',
                                                                use_bias=False,
                                                                depth_multiplier=1,
                                                                dilation_rate=self.dilation_rate,
                                                                # TODO: maybe use the bias when "counting" normalized data
                                                                depthwise_initializer=self.kernel_init,
                                                                name=f'depth_conv_{scale}',
                                                                # TODO: the paper
                                                                # actually uses a Gaussian Kernel with bandwidth 1
                                                                ))

    def get_config(self):
        config = super(DepthwiseMeasurementsAtScale, self).get_config()
        config.update({
            "max_scale": self.max_scale,
            "kernel_init": self.kernel_init,
            "dilation_rate": self.dilation_rate,
        })
        return config

    def call(self, x):
        conv_list = []
        for i in range(self.max_scale):
            out = self.op_list[i](x)
            conv_list.append(out)
        return tf.stack(conv_list, axis=-1)


class FSpecialGaussianInitializer(tf.keras.initializers.Initializer):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, shape, dtype=None):
        assert shape[0] == shape[1]
        length = tf.cast(shape[0], dtype=tf.int32)
        start_ = tf.cast(-(length - 1) / 2, dtype=tf.int32)
        end_ = tf.cast((length - 1) / 2, dtype=tf.int32)
        axis = tf.linspace(start=start_, stop=end_, num=length)
        gauss = tf.exp(-0.5 * (axis ** 2) / (self.sigma ** 2))
        kernel = tf.tensordot(gauss, gauss, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)
        return tf.cast(kernel[:, :, tf.newaxis, tf.newaxis], dtype=dtype)

    def get_config(self):  # To support serialization
        return {'sigma': self.sigma}


class FSpecialGaussianInitializer1D(tf.keras.initializers.Initializer):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, shape, dtype=None):
        assert shape[0] > 0

        length = tf.cast(shape[0], dtype=tf.int32)
        start_ = tf.cast(-(length - 1) / 2, dtype=tf.int32)
        end_ = tf.cast((length - 1) / 2, dtype=tf.int32)
        axis = tf.linspace(start=start_, stop=end_, num=length)
        one_d = tf.exp(-0.5 * (axis ** 2) / (self.sigma ** 2))
        return tf.cast(one_d / tf.reduce_sum(one_d), dtype=dtype)

    def get_config(self):  # To support serialization
        return {'sigma': self.sigma}


class Rank1Conv2D(tf.keras.layers.Layer):
    """
    A rank-1 learnable kernel intialized as a guassian low pass filter of length sigma
    """

    def __init__(self, kernel_size, sigma, padding, **kwargs):
        super(Rank1Conv2D, self).__init__(**kwargs)
        self.sigma = sigma
        self.padding = padding
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.base_kernel = tf.Variable(
            FSpecialGaussianInitializer1D(self.sigma)([self.kernel_size], dtype=tf.float32),
            dtype=tf.float32, trainable=True)
        super(Rank1Conv2D, self).build(input_shape)

    def call(self, x):
        first_kernel = tf.expand_dims(self.base_kernel, 1)[:, :, tf.newaxis, tf.newaxis]
        second_kernel = tf.expand_dims(self.base_kernel, 0)[:, :, tf.newaxis, tf.newaxis]

        conv1 = tf.nn.conv2d(x, first_kernel, strides=[1, 1, 1, 1], padding=self.padding)
        return tf.nn.conv2d(conv1, second_kernel, strides=[1, 1, 1, 1], padding=self.padding)

    def get_config(self):
        config = super(Rank1Conv2D, self).get_config()
        config.update({
            "sigma": self.sigma,
            "padding": self.padding,
            "kernel_size": self.kernel_size
        })


class GaussianConv2DAtScale(tf.keras.layers.Layer):
    """
    A learnable gaussian kernel convolution at scale where the output of the convolution is stacked
    instead of being summed in the channel dimension.
    """

    def __init__(self, length, sigma, padding, **kwargs):
        super(GaussianConv2DAtScale, self).__init__(**kwargs)
        self.length = length
        self.sigma = sigma
        self.padding = padding

    def build(self, input_shape):
        self.op_list = []
        for i in range(input_shape[-1]):
            kernel_init = FSpecialGaussianInitializer(self.sigma)
            self.op_list.append(tf.keras.layers.Conv2D(1,
                                                       kernel_size=(self.length, self.length),
                                                       kernel_initializer=kernel_init,
                                                       padding=self.padding,
                                                       activation='relu'))

    def call(self, x):
        output_stack = []
        for i in range(len(self.op_list)):
            conv = self.op_list[i](x[..., i][..., tf.newaxis])
            conv = conv * (self.sigma ** 2)
            output_stack.append(conv)
        return tf.squeeze(tf.stack(output_stack, axis=-1), axis=-2)

    def get_config(self):
        config = super(GaussianConv2DAtScale, self).get_config()
        config.update({
            "length": self.length,
            "sigma": self.sigma,
            "padding": self.padding
        })
        return config


class LeastSquaresFittingLayer(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(LeastSquaresFittingLayer, self).__init__(**kwargs)
        self.max_scale = max_scale

    def call(self, x):
        scales = tf.range(1, self.max_scale + 1,
                          dtype=tf.float32)  # Inquiry if adding EPSILON here makes sense
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = (log_measures - mean_log_measures) * (log_scales - mean_log_scales)
        denominator = (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)

    def get_config(self):
        config = super(LeastSquaresFittingLayer, self).get_config()
        config.update({
            "max_scale": self.max_scale
        })
        return config


class HolderExponentsLayer(tf.keras.layers.Layer):
    def __init__(self, max_scale, trainable=True, **kwargs):
        super(HolderExponentsLayer, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.trainable = trainable

    def build(self, input_shape):
        self.op_list = []
        for i in range(1, self.max_scale + 1):
            self.op_list.append(SharedConv2D(1, kernel_size=(i, i),
                                             strides=(1, 1),
                                             padding='SAME',
                                             kernel_initializer=FSpecialGaussianInitializer(sigma=i / 2),
                                             activation=None,
                                             use_bias=False,
                                             trainable=self.trainable))

        super(HolderExponentsLayer, self).build(input_shape)

    def call(self, x):
        measurements = []
        for i in range(self.max_scale):
            # out = self.op_list[i](x) * ((i + 1) ** 2) + 1 add add relu in the conv definition
            out = self.op_list[i](x) * ((i + 1) ** 2)
            out = tf.nn.relu(out) + 1
            measurements.append(out)

        return tf.stack(measurements, axis=-1)

    def get_config(self):
        config = super(HolderExponentsLayer, self).get_config()
        config.update({
            "max_scale": self.max_scale
        })
        return config


def level_set_gate(x, max_scales=6, k=16):
    # TODO: This is an initial implementation and probably requires correction
    x_shape = x.get_shape()
    spatial_squeeze = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)
    mfs = HolderExponentsLayer(max_scales)(spatial_squeeze)
    mfs = LeastSquaresFittingLayer(max_scales)(mfs)
    mfs = tf.squeeze(SoftGroupAssignment(k)(mfs), axis=-2)
    mfs_shape = mfs.get_shape()

    mfs = tf.keras.layers.Reshape([mfs_shape[1] * mfs_shape[2], mfs_shape[-1]])(mfs)
    x_ = tf.keras.layers.Reshape([x_shape[1] * x_shape[2], x_shape[-1]])(x)
    blp = tf.matmul(mfs, x_, transpose_a=True)
    blp = tf.transpose(blp, [0, 2, 1])
    gate_kernel = tf.squeeze(tf.keras.layers.Conv1D(1, kernel_size=1, activation='relu', use_bias=True)(blp), axis=-1)
    gate_kernel = gate_kernel[:, tf.newaxis, tf.newaxis, :]
    gate = tf.nn.sigmoid(gate_kernel)
    return x * gate


def holder_blp_gate(x, max_scales=6):
    # TODO: This is an initial implementation and probably requires correctio
    x_shape = x.get_shape()
    spatial_squeeze = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)
    mfs = HolderExponentsLayer(max_scales)(spatial_squeeze)
    mfs = LeastSquaresFittingLayer(max_scales)(mfs)
    mfs_shape = mfs.get_shape()

    mfs = tf.keras.layers.Reshape([mfs_shape[1] * mfs_shape[2], mfs_shape[-1]])(mfs)
    x_ = tf.keras.layers.Reshape([x_shape[1] * x_shape[2], x_shape[-1]])(x)
    blp = tf.matmul(mfs, x_, transpose_a=True)
    blp = tf.transpose(blp, [0, 2, 1])
    gate_kernel = tf.squeeze(tf.keras.layers.Conv1D(1, kernel_size=1, activation='relu', use_bias=True)(blp), axis=-1)
    gate_kernel = gate_kernel[:, tf.newaxis, tf.newaxis, :]
    gate = tf.nn.sigmoid(gate_kernel)
    return x * gate


class SoftGroupAssignment(tf.keras.layers.Layer):
    """
    A TensorFlow Layer that determines a (soft) membership function for a set of (feature) point set categorizations
    in a data driven fashion.
    Implementation of Point Grouping Block (equations 9 through 12) of:
    Encoding Spatial Distribution of Convolutional Features for Texture Representation by Yong Xu et al.
    """

    def __init__(self, num_anchors, **kwargs):
        self.num_anchors = num_anchors
        super(SoftGroupAssignment, self).__init__(**kwargs)

    def build(self, input_shape):
        # TODO: investigate why these shapes behave similarly
        self.anchor_tensor = self.add_weight(name="anchor_tensor", shape=(input_shape[-1], self.num_anchors),
                                             dtype=tf.float32,
                                             initializer="uniform",
                                             trainable=True)  # TODO: this initialization is different in original implementation

        self.membership_weights = self.add_weight(name="membership_weights", shape=(input_shape[-1], self.num_anchors),
                                                  dtype=tf.float32, initializer="uniform",
                                                  trainable=True
                                                  )

        super(SoftGroupAssignment, self).build(input_shape)

    def call(self, x):
        anchor_feature_tensor = (x[..., tf.newaxis] - self.anchor_tensor) ** 2
        membership_matrix = -self.membership_weights * anchor_feature_tensor
        return K.softmax(membership_matrix, axis=-1)

    def get_config(self):
        config = super(SoftGroupAssignment, self).get_config()
        config.update({
            "num_anchors": self.num_anchors
        })
        return config


class SharedSoftGroupAssignment(tf.keras.layers.Layer):
    """
    A TensorFlow Layer that determines a (soft) membership function for a set of (feature) point set categorizations
    in a data driven fashion.
    Implementation of Point Grouping Block (equations 9 through 12) of:
    Encoding Spatial Distribution of Convolutional Features for Texture Representation by Yong Xu et al.
    """

    def __init__(self, num_anchors, **kwargs):
        self.num_anchors = num_anchors
        super(SharedSoftGroupAssignment, self).__init__(**kwargs)

    def build(self, input_shape):
        # TODO: investigate why these shapes behave similarly
        self.anchor_tensor = self.add_weight(name="anchor_tensor", shape=(1, self.num_anchors),
                                             dtype=tf.float32,
                                             initializer="uniform",
                                             trainable=True)  # TODO: this initialization is different in original implementation

        self.membership_weights = self.add_weight(name="membership_weights", shape=(1, self.num_anchors),
                                                  dtype=tf.float32, initializer="uniform",
                                                  trainable=True
                                                  )

        super(SharedSoftGroupAssignment, self).build(input_shape)

    def call(self, x):
        anchor_feature_tensor = (x[..., tf.newaxis] - self.anchor_tensor) ** 2
        membership_matrix = -self.membership_weights * anchor_feature_tensor
        return K.softmax(membership_matrix, axis=-1)


class HolderGate(tf.keras.layers.Layer):
    def __init__(self, max_scales, **kwargs):
        super(HolderGate, self).__init__(**kwargs)
        self.max_scales = max_scales

    def build(self, input_shape):
        self.downscale = tf.keras.layers.Conv2D(self.max_scales, kernel_size=(1, 1), use_bias=True,
                                                padding='valid', activation=None)
        self.down_bn = tf.keras.layers.BatchNormalization()
        self.density = HolderExponentsLayer(self.max_scales)
        self.holder_exponents = LeastSquaresFittingLayer(self.max_scales)
        self.holder_pool = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), use_bias=True,
                                                  padding='valid', activation=None)
        self.holder_bn = tf.keras.layers.BatchNormalization()
        super(HolderGate, self).build(input_shape)

    def call(self, x, training=False, **kwargs):
        # Characterize each feature slice according to a scaling density function
        down_x = self.downscale(x)
        down_x = self.down_bn(down_x, training=training)
        down_x = tf.nn.relu(down_x)
        density_x = self.density(down_x)
        # Determine the holder regularity at each point
        holder_exponents = self.holder_exponents(density_x)
        # Pool the holder singularities across channels
        holder_pooled = self.holder_pool(holder_exponents)
        holder_pooled = self.holder_bn(holder_pooled, training=training)
        # Learn a gating function
        holder_gate = tf.nn.sigmoid(holder_pooled)
        # Filter input x according to holder gate
        return x * holder_gate

    def get_config(self):
        config = super(HolderGate, self).get_config()
        config.update({
            "max_scales": self.max_scales
        })
        return config


class LevelHolderGate(tf.keras.layers.Layer):
    def __init__(self, down_channels: int = 3,
                 max_scales: int = 6,
                 n_sets: int = 16,
                 gate_activation_fn=tf.nn.sigmoid,
                 **kwargs):
        super(LevelHolderGate, self).__init__(**kwargs)
        self.down_channels = down_channels
        self.max_scales = max_scales
        self.n_sets = n_sets
        self.gate_activation_fn = gate_activation_fn

    def build(self, input_shape):
        self.downscale = tf.keras.layers.Conv2D(self.down_channels, kernel_size=(1, 1), use_bias=True,
                                                padding='valid', activation=None)
        self.down_bn = tf.keras.layers.BatchNormalization()
        self.density = HolderExponentsLayer(self.max_scales)
        self.holder_exponents = LeastSquaresFittingLayer(self.max_scales)
        self.soft_level_set = SoftGroupAssignment(self.n_sets)
        self.level_pool = tf.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), use_bias=True,
                                                 padding='valid', activation=None)
        self.level_bn = tf.keras.layers.BatchNormalization()
        super(LevelHolderGate, self).build(input_shape)

    def call(self, x, training=False, **kwargs):
        # Characterize each feature slice according to a scaling density function
        down_x = self.downscale(x)
        down_x = self.down_bn(down_x, training=training)
        down_x = tf.nn.relu(down_x)
        density_x = self.density(down_x)
        # Determine the holder regularity at each point
        holder_exponents = self.holder_exponents(density_x)
        soft_sets = self.soft_level_set(holder_exponents)
        # Pool the holder singularities across channels
        pooled_sets = tf.squeeze(self.level_pool(soft_sets), axis=-1)
        pooled_sets = self.level_bn(pooled_sets, training=training)
        # Learn a gating function
        level_holder_gate = self.gate_activation_fn(pooled_sets)
        # Filter input x according to holder gate
        return x * level_holder_gate

    def get_config(self):
        config = super(LevelHolderGate, self).get_config()
        config.update({
            "down_channels": self.down_channels,
            "max_scales": self.max_scales,
            "n_sets": self.n_sets,
            "gate_activation_fn": self.gate_activation_fn
        })
        return config


def differential_box_counting_2d(x, max_scale):
    # this will not work for surfaces that are centered around 0 with negative values
    max_intensity = tf.reduce_max(tf.reduce_max(x, axis=1), axis=1)[:, tf.newaxis, tf.newaxis, :]
    n_boxes = []
    for scale in range(2, max_scale):  # x.shape[1] // 2
        h_s = tf.cast((scale * max_intensity) / x.shape[1], dtype=tf.float32)
        # TODO: remember that floor is not diff. so we need to remove it in the future
        g_max = tf.keras.layers.MaxPooling2D(pool_size=(scale, scale),
                                             strides=(scale, scale),
                                             padding='SAME')(x)
        g_max = tf.math.floor(g_max / h_s)

        g_min = MinPooling2DWrapper(x, pool_size=(scale, scale),
                                    strides=(scale, scale),
                                    padding='SAME')
        g_min = tf.math.floor(g_min / h_s)
        n_box = tf.reduce_sum(g_max - g_min + 1, axis=[1, 2])
        n_boxes.append(n_box)

    return tf.stack(n_boxes, axis=-1)


def faster_soft_box_count(x, max_scale):
    n_boxes = []
    for scale in range(1, max_scale + 1):
        boxes = tf.reduce_sum(tf.keras.layers.MaxPool3D(pool_size=(scale, scale, 1),
                                                        strides=(1, 1, 1),
                                                        padding="same")(x), axis=[1, 2])
        n_boxes.append(boxes)
    return tf.stack(n_boxes, axis=-1)


def soft_differential_box_counting_2d(x, max_scale):
    """
    Returns TOTAL number of boxes for the entire input for 1,...,max_scale
    :param x:
    :param max_scale:
    :return:
    """
    # this will not work for surfaces that are centered around 0 with negative values
    max_intensity = tf.reduce_max(tf.reduce_max(x, axis=1), axis=1)[:, tf.newaxis, tf.newaxis,
                    :]  # see max for each of the samples in the batch and broadcast
    n_boxes = []

    for scale in range(2, max_scale):  # x.shape[1] // 2
        h_s = tf.cast((scale * max_intensity) / x.shape[1], dtype=tf.float32)
        # TODO: remember that floor is not diff. so we need to remove it in the future
        g_max = tf.keras.layers.MaxPooling2D(pool_size=(scale, scale),
                                             strides=(scale, scale),
                                             padding='SAME')(x) / h_s

        g_min = MinPooling2DWrapper(x, pool_size=(scale, scale),
                                    strides=(scale, scale),
                                    padding='SAME') / h_s
        n_box = tf.reduce_sum(g_max - g_min + 1, axis=[1, 2])
        n_boxes.append(n_box)

    return tf.stack(n_boxes, axis=-1)


def soft_box_counting_2d(x, max_scale):
    "Returns the number of boxes at i,j at scale r"
    # this will not work for surfaces that are centered around 0 with negative values
    max_intensity = tf.reduce_max(tf.reduce_max(x, axis=1), axis=1)[:, tf.newaxis, tf.newaxis,
                    :]  # see max for each of the samples in the batch and broadcast
    n_boxes = []
    for scale in range(2, max_scale):  # x.shape[1] // 2
        h_s = tf.cast((scale * max_intensity) / x.shape[1], dtype=tf.float32)
        # TODO: remember that floor is not diff. so we need to remove it in the future
        g_max = tf.keras.layers.MaxPooling2D(pool_size=(scale, scale),
                                             strides=(scale, scale),
                                             padding='SAME')(x) / h_s

        g_min = MinPooling2DWrapper(x, pool_size=(scale, scale),
                                    strides=(scale, scale),
                                    padding='SAME') / h_s
        n_boxes.append(g_max - g_min + 1)

    return tf.stack(n_boxes, axis=-1)


def max_mfs(x, holder_scales=6, num_anchors=13):
    x = standardize_values(x)
    x = HolderExponentsLayer(max_scale=holder_scales)(x)
    x = LeastSquaresFittingLayer(max_scale=holder_scales)(x)
    x = SoftGroupAssignment(num_anchors=num_anchors)(x)
    return tf.reduce_max(x, axis=-1)
