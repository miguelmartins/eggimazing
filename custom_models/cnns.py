import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, models, Input


def simple_cnn(input_shape=(224, 224, 3), n_classes=3):
    model = models.Sequential()

    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flattening the output from the convolutional blocks
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(128, activation='relu'))

    # Output layer with softmax activation
    model.add(layers.Dense(n_classes, activation='softmax'))
    return model


def simple_cnn_bn(input_shape=(224, 224, 3), n_classes=3, return_embeddings=False, gap=False):
    # Input layer
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third convolutional block
    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    if return_embeddings:
        return models.Model(inputs=inputs, outputs=x)
    if gap:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Flattening the output from the convolutional blocks
    # Fully connected layer
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Output layer with softmax activation
    if n_classes > 2:
        outputs = layers.Dense(n_classes, activation='softmax')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def base_resnet50(input_shape=(224, 224, 3), n_classes=3):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model
