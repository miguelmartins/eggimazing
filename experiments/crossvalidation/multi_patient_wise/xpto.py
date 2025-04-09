import numpy as np
import tensorflow as tf
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy

from custom_models.augmentation import basic_plus_color_augmentation, basic_augmentation
from custom_models.bilinear_cnns import fe_resnet
from custom_models.cnns import simple_cnn_bn, base_resnet50
from custom_models.optimization_utilities import get_standard_callbacks
from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset
from optimization.custom_losses import weighted_categorical_crossentropy


def main():
    import os
    print(os.getcwd())
    target_dir = 'test_files/EGGIMazing/Dataset01'
    patient_ids = np.load('configs/test_patient_ids_2.npy', allow_pickle=True)
    print(np.load('configs/test_patient_ids_2.npy', allow_pickle=True))
    batch_size = 32
    num_epochs = 400
    learning_rate = 1e-4
    num_folds = len(patient_ids)
    name = f'../../../logs/cv_embc_patient_resnet_multi_{num_folds}'

    dp = DatasetProcessor(target_dir)
    df = dp.process()

    togas_ids_boolean = np.array([x.startswith('PT') for x in df['patient_id'].values])
    df_togas = df[togas_ids_boolean].reset_index(drop=True)
    df_ipo = df[~togas_ids_boolean].reset_index(drop=True)

    split = dp.patient_wise_split(df_togas,
                                  df_ipo,
                                  patient_ids,
                                  internal_train_size=0.9,
                                  target_variable='eggim_square',
                                  random_state=42)
    for fold, (_, _, df_test) in enumerate(split):
        #try:
        print(f"good {patient_ids[fold]}")
        tf_test_df = get_tf_eggim_patch_dataset(df_test,
                                                num_classes=3,
                                                preprocess_fn=tf.keras.applications.resnet.preprocess_input)

        tf_test_df = tf_test_df.batch(batch_size)

        #except Exception as e:
        #    print(e)
        #    print(f"Patient n {fold}; ID {patient_ids[fold]}")


if __name__ == '__main__':
    main()
