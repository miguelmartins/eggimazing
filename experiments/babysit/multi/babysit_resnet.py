import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy

from tensorflow.keras import backend as K
from custom_models.cnns import simple_cnn_bn, base_resnet50
from custom_models.augmentation import basic_augmentation, basic_plus_color_augmentation
from custom_models.optimization_utilities import get_standard_callbacks
from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset


def main():
    target_dir = '../test_files/EGGIMazing/Dataset'
    batch_size = 32
    num_epochs = 400
    learning_rate = 1e-4
    num_folds = 5
    name = f'../logs/cnn_basic_no_da_togasipo_cv_{num_folds}'

    dp = DatasetProcessor(target_dir)
    df = dp.process()

    togas_ids_boolean = np.array([x.startswith('PT') for x in df['patient_id'].values])
    df_togas = df[togas_ids_boolean].reset_index(drop=True)
    df_ipo = df[~togas_ids_boolean].reset_index(drop=True)

    split = dp.smarter_multiple_ds_group_k_splits(df_togas,
                                                  df_ipo,
                                                  k=num_folds,
                                                  train_size=0.6,
                                                  test_size=0.4,
                                                  internal_train_size=0.8,
                                                  random_state=42)

    test_idx = 4
    print("FOLD ", test_idx)
    i = 0
    for df_train, df_val, df_test in split:
        if i < test_idx:
            i += 1
        else:
            break
    fold = 'test_fold'
    y_train = df_train['eggim_square']
    class_counts = np.bincount(y_train)

    # Compute class weights
    class_weights_manual = {i: len(y_train) / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}
    # Convert the class weights dictionary to a tensor
    weights = tf.constant(list(class_weights_manual.values()), dtype=tf.float32)

    # Custom loss function that applies the weights
    def weighted_categorical_crossentropy(y_true, y_pred):
        # Compute the standard categorical crossentropy loss
        cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply the weights to the loss
        weights_applied = tf.reduce_sum(weights * y_true, axis=-1)

        return cce_loss * weights_applied

    tf_train_df = get_tf_eggim_patch_dataset(df_train, num_classes=3, augmentation_fn=basic_plus_color_augmentation,
                                             preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    tf_val_df = get_tf_eggim_patch_dataset(df_val, num_classes=3,
                                           preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    tf_test_df = get_tf_eggim_patch_dataset(df_test, num_classes=3,
                                            preprocess_fn=tf.keras.applications.resnet.preprocess_input)

    tf_train_df = tf_train_df.batch(batch_size)
    tf_val_df = tf_val_df.batch(batch_size)
    tf_test_df = tf_test_df.batch(batch_size)

    n_classes = 3  # Replace with the number of classes you have
    model = base_resnet50(input_shape=(224, 224, 3), n_classes=n_classes)
    # Compile the model with Adam optimizer
    # simple da
    # 3/3 [==============================] - 0s 38ms/step - loss: 0.5910 - cat_accuracy: 0.7778 - precision: 0.7805 - recall: 0.7111 - auc: 0.9097
    # color da
    # 3/3 [==============================] - 0s 38ms/step - loss: 0.7556 - cat_accuracy: 0.7222 - precision: 0.7733 - recall: 0.6444 - auc: 0.8593
    # no da
    # 3/3 [==============================] - 0s 43ms/step - loss: 0.6302 - cat_accuracy: 0.7667 - precision: 0.7791 - recall: 0.7444 - auc: 0.9096
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=weighted_categorical_crossentropy,
                  metrics=[CategoricalAccuracy(name='cat_accuracy'), Precision(name='precision'), Recall(name='recall'),
                           AUC(name='auc')])

    name_fold = name + f'fold_{fold}'
    checkpoint_dir, callbacks = get_standard_callbacks(name_fold, learning_rate)
    model.fit(tf_train_df,
              validation_data=tf_val_df,
              epochs=num_epochs,
              callbacks=callbacks)
    # around 70 no class weight
    model.load_weights(f'{checkpoint_dir}/weights.h5')
    model.evaluate(tf_test_df)


if __name__ == '__main__':
    main()
