import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy

from custom_models.optimization_utilities import get_standard_callbacks
from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset


def main():
    target_dir = '../test_files/EGGIMazing/Dataset'
    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-4

    dp = DatasetProcessor(target_dir)
    df = dp.process()
    df_togas = df[[x.startswith('2024') for x in df['patient_id'].values]].reset_index(drop=True)
    df_ipo = df[[not x.startswith('2024') for x in df['patient_id'].values]].reset_index(drop=True)

    split = dp.group_k_splits(df_togas, k=1, train_size=0.7, val_size=0.1, test_size=0.2, random_state=42)
    train_idx, val_idx, test_idx = next(split)
    df_togas_train = df_togas.loc[train_idx]
    df_train = pd.concat([df_ipo, df_togas_train], axis=0)

    tf_train_df = get_tf_eggim_patch_dataset(df_train, num_classes=3, preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    tf_val_df = get_tf_eggim_patch_dataset(df_togas.loc[val_idx], num_classes=3, preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    tf_test_df = get_tf_eggim_patch_dataset(df_togas.loc[test_idx], num_classes=3, preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    print("train, val, test size:")
    print(len(tf_train_df), len(tf_val_df), len(tf_test_df))
    tf_train_df = tf_train_df.batch(batch_size)
    tf_val_df = tf_val_df.batch(batch_size)
    tf_test_df = tf_test_df.batch(batch_size)

    n_classes = 3  # Number of classes in your dataset
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[CategoricalAccuracy(name='cat_accuracy'), Precision(name='precision'), Recall(name='recall'),
                           AUC(name='auc')])

    checkpoint_dir, callbacks = get_standard_callbacks('resnet50', learning_rate)
    model.fit(tf_train_df,
              validation_data=tf_val_df,
              epochs=num_epochs,
              callbacks=callbacks)
    model.load_weights(f'{checkpoint_dir}/weights.h5')
    model.evaluate(tf_test_df)


if __name__ == '__main__':
    main()
