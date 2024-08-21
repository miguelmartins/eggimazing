import itertools

import numpy as np
import tensorflow as tf
from keras.metrics import Precision, Recall, AUC
from custom_models.cnns import simple_cnn_bn
from custom_models.augmentation import basic_plus_color_augmentation, basic_augmentation
from custom_models.optimization_utilities import get_standard_callbacks
from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset
from optimization.custom_losses import weighted_binary_crossentropy


def main():
    target_dir = '../test_files/EGGIMazing/Dataset'
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    num_folds = 5
    name = f'../logs/cnn_basic_no_da_togasipo_cv_{num_folds}'
    n_classes = 2  # Replace with the number of classes you have
    dp = DatasetProcessor(target_dir)
    df = dp.process(merge_eggim_square=True)

    togas_ids_boolean = np.array([x.startswith('PT') for x in df['patient_id'].values])
    df_togas = df[togas_ids_boolean].reset_index(drop=True)
    df_ipo = df[~togas_ids_boolean].reset_index(drop=True)

    split = dp.patient_k_group_split(df_togas,
                                     df_ipo,
                                     k=num_folds,
                                     train_size=0.9,
                                     test_size=0.1,
                                     internal_train_size=0.5,
                                     random_state=42)

    test_idx = 2
    df_train, df_val, df_test = next(itertools.islice(split, test_idx, test_idx + 1))
    fold = 'test_fold'
    y_train = df_train['eggim_square']
    class_counts = np.bincount(y_train)

    # Compute class weights
    class_weights_manual = {i: len(y_train) / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}
    # Convert the class weights dictionary to a tensor
    weights = tf.constant(list(class_weights_manual.values()), dtype=tf.float32)
    print("Weights", weights)
    # Custom loss function that applies the weights

    tf_train_df = get_tf_eggim_patch_dataset(df_train, num_classes=n_classes,
                                             augmentation_fn=basic_augmentation)
    tf_val_df = get_tf_eggim_patch_dataset(df_val, num_classes=n_classes)
    tf_test_df = get_tf_eggim_patch_dataset(df_test, num_classes=n_classes)

    tf_train_df = tf_train_df.batch(batch_size)
    tf_val_df = tf_val_df.batch(batch_size)
    tf_test_df = tf_test_df.batch(batch_size)

    model = simple_cnn_bn(input_shape=(224, 224, 3), n_classes=n_classes, gap=True)
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=weighted_binary_crossentropy(class_weights_manual),
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall'),
                           AUC(name='auc')])

    name_fold = name + f'fold_{fold}'
    print(model.summary())
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

# 1/1 [==============================] - 0s 142ms/step - loss: 0.6193 - accuracy: 0.6667 - precision: 0.7647 - recall: 0.6842 - auc: 0.7416