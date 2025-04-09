import numpy as np
import tensorflow as tf
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy

from custom_models.augmentation import basic_plus_color_augmentation, basic_augmentation
from custom_models.bilinear_cnns import fe_resnet
from custom_models.cnns import simple_cnn_bn, base_resnet50
from custom_models.optimization_utilities import get_standard_callbacks, get_recall_early
from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset
from optimization.custom_losses import weighted_categorical_crossentropy, focal_loss_with_class_weights


def main():
    target_dir = 'test_files/EGGIMazing/Dataset01'
    patient_ids = np.load('configs/test_patient_ids_2.npy', allow_pickle=True)
    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-4
    num_folds = len(patient_ids)
    name = f'logs/cv_embc2_aggressiveonly_togas_resnet_multi_{num_folds}'

    dp = DatasetProcessor(target_dir)
    df = dp.process()

    togas_ids_boolean = np.array([x.startswith('PT') for x in df['patient_id'].values])
    df_togas = df[togas_ids_boolean].reset_index(drop=True)
    df_ipo = df[~togas_ids_boolean].reset_index(drop=True)

    y_train = df_togas['eggim_square']
    class_counts = np.bincount(y_train)
    class_weights_manual = {i: len(y_train) / (len(class_counts) * class_counts[i]) for i in
                            range(len(class_counts))}
    weights = tf.constant(list(class_weights_manual.values()), dtype=tf.float32)

    split = dp.single_ds_patient_wise_split(df_togas,
                                            patient_ids,
                                            internal_train_size=0.8,
                                            target_variable='eggim_square',
                                            random_state=42)
    for fold, (df_train, df_val, df_test) in enumerate(split):
        tf_train_df = get_tf_eggim_patch_dataset(df_train,
                                                 num_classes=3,
                                                 augmentation_fn=basic_augmentation,
                                                 preprocess_fn=tf.keras.applications.resnet.preprocess_input)
        tf_val_df = get_tf_eggim_patch_dataset(df_val,
                                               num_classes=3,
                                               preprocess_fn=tf.keras.applications.resnet.preprocess_input)
        tf_test_df = get_tf_eggim_patch_dataset(df_test,
                                                num_classes=3,
                                                preprocess_fn=tf.keras.applications.resnet.preprocess_input)
        tf_train_df = tf_train_df.batch(batch_size)
        tf_val_df = tf_val_df.batch(batch_size)
        tf_test_df = tf_test_df.batch(batch_size)

        n_classes = 3  # Replace with the number of classes you have
        model = base_resnet50(input_shape=(224, 224, 3), n_classes=n_classes)

        # Compile the model with Adam optimizer 13:21

        class CombinedRecall(tf.keras.metrics.Metric):
            def __init__(self, name='combined_recall', **kwargs):
                super(CombinedRecall, self).__init__(name=name, **kwargs)
                self.recall_1 = tf.keras.metrics.Recall(class_id=1)
                self.recall_2 = tf.keras.metrics.Recall(class_id=2)

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.recall_1.update_state(y_true, y_pred, sample_weight)
                self.recall_2.update_state(y_true, y_pred, sample_weight)

            def result(self):
                return (self.recall_1.result() + self.recall_2.result()) / 2

            def reset_states(self):
                self.recall_1.reset_states()
                self.recall_2.reset_states()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=weighted_categorical_crossentropy(weights),
                      metrics=[CategoricalAccuracy(name='cat_accuracy'), Precision(name='precision'),
                               CombinedRecall()])

        name_fold = name + f'fold_{fold}'
        checkpoint_dir, callbacks = get_standard_callbacks(name_fold, learning_rate)
        model.fit(tf_train_df,
                  validation_data=tf_val_df,
                  epochs=num_epochs,
                  callbacks=callbacks)
        model.load_weights(f'{checkpoint_dir}/weights.h5')
        model.evaluate(tf_test_df)


if __name__ == '__main__':
    main()
