import tensorflow as tf
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy

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
    target_dir = '../test_files/EGGIMazing/Dataset'
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4

    dp = DatasetProcessor(target_dir)
    df = dp.process()
    df = df[~df.isna().any(axis=1)].reset_index(drop=True)
    X, y = df['image_directory'], df['eggim_square']

    split = dp.stratified_k_splits(X, y, k=1, train_size=0.8, val_size=0.1, test_size=0.1)
    train_idx, val_idx, test_idx = next(split)
    # df_train = df.loc[train_idx]
    tf_train_df = get_tf_eggim_patch_dataset(df.loc[train_idx], num_classes=3)
    tf_val_df = get_tf_eggim_patch_dataset(df.loc[val_idx], num_classes=3)
    tf_test_df = get_tf_eggim_patch_dataset(df.loc[test_idx], num_classes=3)

    tf_train_df = tf_train_df.batch(batch_size)
    tf_val_df = tf_val_df.batch(batch_size)
    tf_test_df = tf_test_df.batch(batch_size)

    n_classes = 3  # Replace with the number of classes you have
    model = simple_cnn(input_shape=(224, 224, 3), n_classes=n_classes)
    # Compile the model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[CategoricalAccuracy(name='cat_accuracy'), Precision(name='precision'), Recall(name='recall'),
                           AUC(name='auc')])

    model.fit(tf_train_df, validation_data=tf_val_df, epochs=num_epochs)
    model.evaluate(tf_test_df)


if __name__ == '__main__':
    main()
