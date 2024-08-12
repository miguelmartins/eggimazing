import tensorflow as tf
from keras.metrics import Precision, Recall, AUC

from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset
from tensorflow.keras import layers, models


def simple_cnn(input_shape=(224, 224, 3), n_classes=10):
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


def main():
    target_dir = '../test_files/TOGETHER'
    dp = DatasetProcessor(target_dir)
    df = dp.process()
    tf_df = get_tf_eggim_patch_dataset(df, num_classes=3)
    tf_df = tf_df.batch(1)
    # Example usage
    n_classes = 3  # Replace with the number of classes you have
    model = simple_cnn(input_shape=(224, 224, 3), n_classes=n_classes)
    # Compile the model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])

    model.fit(tf_df, epochs=10)


if __name__ == '__main__':
    main()
