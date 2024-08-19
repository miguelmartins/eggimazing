import tensorflow as tf

from fractal_geometry.mfs import get_mfs, MultiFractalSpectrumLayer


class SpatialAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        super(SpatialAttentionBlock, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.bilinear_upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.project_encoder = tf.keras.layers.Conv2D(self.attention_dim,
                                                      kernel_size=(1, 1),
                                                      use_bias=False,
                                                      padding='valid',
                                                      activation=None)
        self.pe_bn = tf.keras.layers.BatchNormalization()

        self.project_decoder = tf.keras.layers.Conv2D(self.attention_dim,
                                                      kernel_size=(1, 1),
                                                      use_bias=True,
                                                      padding='valid',
                                                      activation=None)
        self.upconv_g = tf.keras.layers.Conv2DTranspose(self.attention_dim,
                                                        kernel_size=3,
                                                        strides=(2, 2),
                                                        activation=None,
                                                        padding="same",
                                                        kernel_initializer='HeNormal', )
        self.upconv_g_bn = tf.keras.layers.BatchNormalization()

        self.pd_bn = tf.keras.layers.BatchNormalization()

        self.psi_map = tf.keras.layers.Conv2D(1,
                                              kernel_size=(1, 1),
                                              use_bias=True,
                                              padding='valid',
                                              activation=None)
        self.psi_bn = tf.keras.layers.BatchNormalization()
        super(SpatialAttentionBlock, self).build(input_shape)

    def call(self, inputs, training=False, use_upconv=False, **kwargs):
        x, g = inputs
        proj_x = self.project_encoder(x)
        proj_x = self.pe_bn(proj_x, training=training)

        proj_g = self.project_decoder(g)
        proj_g = self.pd_bn(proj_g, training=training)
        if use_upconv:
            proj_g = self.upconv_g(proj_g)
            proj_g = self.upconv_g_bn(proj_g, training=training)
            proj_g = tf.nn.relu(proj_g)
        else:
            proj_g = self.bilinear_upsample(proj_g)

        psi = tf.nn.relu(proj_x + proj_g)
        psi_map = self.psi_map(psi)
        psi_map = self.psi_bn(psi_map, training=training)
        return x * tf.nn.sigmoid(psi_map)

    def get_config(self):
        config = super(SpatialAttentionBlock, self).get_config()
        config.update({
            "attention_dim": self.attention_dim
        })
        return config


class DualAttentionModule(tf.keras.layers.Layer):
    # PAM module in CVPR 2019 paper:
    # Dual Attention Network for Scene Segmentation by Jun Fu et al.
    def __init__(self, *,
                 channel_dim,
                 kernel_dim,
                 activation_conv,
                 activation_attention,
                 use_conv=True,
                 attention_type='channel',  # Flag that makes module behave as CAM or PAM
                 attention_weight_init=tf.keras.initializers.zeros):

        super(DualAttentionModule, self).__init__()
        self.channel_dim = channel_dim
        self.use_conv = use_conv
        self.attention_type = attention_type
        self.query_conv = tf.keras.layers.Conv2D(channel_dim,
                                                 (kernel_dim, kernel_dim),
                                                 padding='same',
                                                 activation=None)
        self.key_conv = tf.keras.layers.Conv2D(channel_dim,
                                               (kernel_dim, kernel_dim),
                                               padding='same',
                                               activation=None)
        self.value_conv = tf.keras.layers.Conv2D(channel_dim,
                                                 (kernel_dim, kernel_dim),
                                                 padding='same',
                                                 activation=None)

        self.query_bn = tf.keras.layers.BatchNormalization()
        self.value_bn = tf.keras.layers.BatchNormalization()
        self.key_bn = tf.keras.layers.BatchNormalization()

        self.activation_conv = activation_conv
        self.activation_attention = activation_attention

        self.attention_weight = None
        self.attention_weight_init = attention_weight_init

    def build(self, input_shape):
        self.attention_weight = self.add_weight(shape=(1,),
                                                initializer=self.attention_weight_init,
                                                name='attention_weight',
                                                trainable=True)

    def call(self, x, use_conv=False, training=None):
        h = tf.shape(x)[1]  # we have to use this because tf.autograph sucks pp
        w = tf.shape(x)[2]
        if self.use_conv:
            query = self.query_conv(x)
            query = self.query_bn(query)
            query = self.activation_conv(query)

            key = self.key_conv(x)
            key = self.key_bn(key)
            key = self.activation_conv(key)

            value = self.value_conv(x)
            value = self.value_bn(value)
            value = self.activation_conv(value)
        else:
            query = tf.identity(x)
            key = tf.identity(x)
            value = tf.identity(x)

        value = tf.reshape(value, [-1, h * w, self.channel_dim])
        query = tf.reshape(query, [-1, h * w, self.channel_dim])
        key = tf.reshape(key, [-1, h * w, self.channel_dim])

        if self.attention_type == 'channel':
            energy = tf.matmul(query, key, transpose_a=True)
        else:
            energy = tf.matmul(query, key, transpose_b=True)
        attention = self.activation_attention(energy)

        if self.attention_type == 'channel':
            out = tf.matmul(attention, value, transpose_b=True)
        else:
            out = tf.matmul(attention, value)
        out = tf.reshape(out, [-1, h, w, self.channel_dim])
        return (self.attention_weight * out) + x


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, r, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SqueezeExcite, self).build(input_shape)

    def call(self, x):
        squeeze = self.gap(x)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super(SqueezeExcite, self).get_config()
        config.update({
            "r": self.r
        })
        return config


class MFSSqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, r, proj_dim, local_scale, global_scale, **kwargs):
        self.r = r
        self.proj_dim = proj_dim
        self.local_scale = local_scale
        self.global_scale = global_scale
        self.flatten = tf.keras.layers.Flatten()
        self.cnn = tf.keras.layers.Conv2D(self.proj_dim,
                                          kernel_size=(1, 1),
                                          strides=(1, 1),
                                          activation=None)
        self.mfs = MultiFractalSpectrumLayer(local_scale=self.local_scale,
                                             global_scale=self.global_scale)
        self.bn = tf.keras.layers.BatchNormalization()
        self.w1 = None
        self.w2 = None
        super(MFSSqueezeExcite, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.proj_dim is not None:
            self.w1 = tf.keras.layers.Dense(self.proj_dim // self.r, activation='relu')
        else:
            self.w1 = tf.keras.layers.Dense(input_shape[-1] // self.r, activation='relu')
        self.w2 = tf.keras.layers.Dense(input_shape[-1], activation='sigmoid')

    def call(self, x):
        if self.proj_dim is not None:
            squeeze = self.cnn(x)
            squeeze = self.bn(squeeze)
            squeeze = tf.nn.relu(squeeze)
        else:
            squeeze = tf.identity(x)
        squeeze = self.flatten(self.mfs(squeeze))
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super(MFSSqueezeExcite, self).get_config()
        config.update({
            "r": self.r,
            "proj_dim": self.proj_dim,
            "local_scale": self.local_scale,
            "global_scale": self.global_scale
        })
        return config
