import numpy as np
import tensorflow as tf
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy

from custom_models.cnns import simple_cnn, simple_cnn_bn
from custom_models.optimization_utilities import get_standard_callbacks
from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset


def main():
    target_dir = '../test_files/EGGIMazing/Dataset'
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    togas_split_size = 0.7
    np.random.seed(42)

    dp = DatasetProcessor(target_dir)
    df = dp.process()
    df_ipo = df[[not x.startswith('2024') for x in df['patient_id'].values]].reset_index(drop=True)

    df_togas = df[[x.startswith('2024') for x in df['patient_id'].values]].reset_index(drop=True)
    df_togas_pids = np.unique(df_togas['patient_id'].values)
    print(len(df_togas_pids))
    np.random.shuffle(df_togas_pids)

    id_cutoff_togas = int(togas_split_size * len(df_togas_pids))
    print("CUTFF", id_cutoff_togas)
    val_togas_ids, test_togas_ids = df_togas_pids[:id_cutoff_togas], df_togas_pids[:id_cutoff_togas]
    df_val = df_togas[[x in val_togas_ids for x in df_togas['patient_id']]].reset_index(drop=True)
    df_test = df_togas[[not x in val_togas_ids for x in df_togas['patient_id']]].reset_index(drop=True)

    # df_train = df.loc[train_idx]
    tf_train_df = get_tf_eggim_patch_dataset(df_ipo, num_classes=3)
    tf_val_df = get_tf_eggim_patch_dataset(df_val, num_classes=3)

    tf_test_df = get_tf_eggim_patch_dataset(df_test, num_classes=3)

    tf_train_df = tf_train_df.batch(batch_size)
    tf_val_df = tf_val_df.batch(batch_size)
    tf_test_df = tf_test_df.batch(batch_size)

    n_classes = 3  # Replace with the number of classes you have
    model = simple_cnn_bn(input_shape=(224, 224, 3), n_classes=n_classes)
    # Compile the model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[CategoricalAccuracy(name='cat_accuracy'), Precision(name='precision'), Recall(name='recall'),
                           AUC(name='auc')])

    checkpoint_dir, callbacks = get_standard_callbacks('simple_cnn', learning_rate)
    model.fit(tf_train_df,
              validation_data=tf_val_df,
              epochs=num_epochs,
              callbacks=callbacks)
    model.load_weights(f'{checkpoint_dir}/weights.h5')
    model.evaluate(tf_test_df)


if __name__ == '__main__':
    main()
