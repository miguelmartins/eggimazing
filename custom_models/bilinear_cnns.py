import tensorflow as tf
from keras import Input
from tensorflow.keras.applications.resnet50 import ResNet50

from custom_models.cnns import simple_cnn_bn
from fractal_geometry.differential_blocks import HolderExponentsLayer, LeastSquaresFittingLayer, \
    SharedSoftGroupAssignment, soft_gliding_boxes


def signed_sqrt(x):
    """
    Calculate signed square root

    @param
    x -> a tensor

    """
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-9)


def L2_norm(x, axis=-1):
    """
    Calculate L2-norm

    @param
    x -> a tensor

    """
    return tf.keras.backend.l2_normalize(x, axis=axis)


def bilinear_pool(x, y):
    blp = tf.linalg.matmul(x, y, transpose_a=True)
    blp = tf.keras.layers.Flatten()(blp)
    blp = tf.keras.layers.Lambda(signed_sqrt)(blp)
    blp = tf.keras.layers.Lambda(L2_norm)(blp)
    return blp


def bilinear_cnn(input_shape=(224, 224, 3), n_classes=3):
    # Input layer
    inputs = Input(shape=input_shape)
    cnn1 = simple_cnn_bn(input_shape=input_shape, n_classes=n_classes, return_embeddings=True)
    cnn2 = simple_cnn_bn(input_shape=input_shape, n_classes=n_classes, return_embeddings=True)

    x1 = cnn1(inputs)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)

    # x2 = cnn2(inputs)
    # x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)[:, tf.newaxis, :]

    x_blp = bilinear_pool(x1[:, tf.newaxis, :], x1[:, tf.newaxis, :])
    # x_blp = tf.keras.layers.Dropout(0.5)(x_blp)
    x_blp = tf.keras.layers.Dense(128)(x_blp)
    x_blp = tf.keras.layers.BatchNormalization()(x_blp)
    x_blp = tf.keras.layers.Activation('relu')(x_blp)

    x = tf.keras.layers.Concatenate()([x1, x_blp])
    # x = x1 + x_blp
    if n_classes > 2:
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def fractal_encoding_preprocessing(x, n_out_channnels):
    x = tf.keras.layers.Conv2D(filters=n_out_channnels, kernel_size=(1, 1), strides=(1, 1), activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x


def get_upscale(x, n_filters):
    return tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           groups=n_filters
                                           )(x)


def fractal_average_pooling_block(x, borell_scales=6, global_scales=16):
    x = HolderExponentsLayer(borell_scales, trainable=False)(x)
    x = LeastSquaresFittingLayer(borell_scales)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = SharedSoftGroupAssignment(global_scales)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = soft_gliding_boxes(x, borell_scales)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = -LeastSquaresFittingLayer(borell_scales)(x)
    return x


def fe_resnet(input_shape=[224, 224, 3], n_classes=3):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    gap_x = tf.keras.layers.GlobalAveragePooling2D()(x)
    gap_x = tf.keras.layers.Dense(128,
                                  activation='relu',
                                  use_bias=True,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0))(gap_x)
    gap_x = tf.keras.layers.Dense(48,
                                  activation='relu',
                                  use_bias=True,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0))(gap_x)
    gap_x = tf.keras.layers.Reshape([1, gap_x.get_shape()[-1]])(gap_x)

    up_scale_x = get_upscale(x, 2048)

    fractal_x = fractal_encoding_preprocessing(up_scale_x, 3)
    fractal_x = fractal_average_pooling_block(fractal_x, 6, 16)
    fractal_shape = fractal_x.get_shape()
    fractal_x = tf.keras.layers.Reshape([1, fractal_shape[-1] * fractal_shape[-2]])(fractal_x)
    #blp = fractal_x = tf.keras.layers.Reshape([fractal_shape[-1] * fractal_shape[-2]])(fractal_x)
    blp = bilinear_pool(fractal_x, gap_x)
    # blp = tf.squeeze(fractal_x, axis=1)
    if n_classes > 2:
        output = tf.keras.layers.Dense(n_classes,
                                       activation='softmax',
                                       use_bias=True,
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.0))(blp)
    else:
        output = tf.keras.layers.Dense(1,
                                       activation='sigmoid',
                                       use_bias=True,
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.0))(blp)
    return tf.keras.models.Model(inputs=base_model.input, outputs=[output])
