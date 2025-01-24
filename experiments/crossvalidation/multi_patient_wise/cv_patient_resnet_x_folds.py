from etl.load_dataset import DatasetProcessor, get_tf_eggim_patch_dataset
from custom_models.cnns import base_resnet50
import tensorflow as tf
from keras.metrics import Precision, Recall, AUC, CategoricalAccuracy
from optimization.custom_losses import weighted_categorical_crossentropy
from custom_models.augmentation import basic_augmentation
from custom_models.optimization_utilities import get_standard_callbacks
import numpy as np

target_dir = '../2025-01_EGGIM_Dataset3' 
patient_ids = np.load('configs/new_patient_ids.npy', allow_pickle=True)
dp = DatasetProcessor(target_dir)
df = dp.process()
togas_ids_boolean = np.array([x.startswith('PT') for x in df['patient_id'].values])
df_togas = df[togas_ids_boolean].reset_index(drop=True)
df_ipo = df[~togas_ids_boolean].reset_index(drop=True)

target_dir = '../2025-01_EGGIM_Dataset3'  # aqui foi o o que eu chamei no remote server, mas fica a criterio
patient_ids = np.load('configs/new_patient_ids.npy', allow_pickle=True)
dp = DatasetProcessor(target_dir)
df = dp.process()
togas_ids_boolean = np.array([x.startswith('PT') for x in df['patient_id'].values])
df_togas = df[togas_ids_boolean].reset_index(drop=True)
df_ipo = df[~togas_ids_boolean].reset_index(drop=True)


batch_size = 32
num_epochs = 2
learning_rate = 1e-4
num_folds = 5
n_classes =3

name = f'../logs/cv_patient_resnet_multi_{num_folds}'

split = dp.patient_wise_split_x_fold(df_togas,df_ipo,patient_ids,n_folds=num_folds,internal_train_size=0.9,target_variable='eggim_square',random_state=42)
i = 0
for fold, (df_train, df_val, df_test) in enumerate(split):
    tf_train_df = get_tf_eggim_patch_dataset(df_train,num_classes=3,
                                             augmentation_fn=basic_augmentation,
                                             preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    tf_val_df = get_tf_eggim_patch_dataset(df_val,
                                           num_classes=3,
                                           preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    tf_test_df = get_tf_eggim_patch_dataset(df_test,
                                            num_classes=3,
                                            preprocess_fn=tf.keras.applications.resnet.preprocess_input)
    y_train = df_train['eggim_square']
    class_counts = np.bincount(y_train)
    class_weights_manual = {i: len(y_train) / (len(class_counts) * class_counts[i]) for i in
                                range(len(class_counts))}
    weights = tf.constant(list(class_weights_manual.values()), dtype=tf.float32)

    tf_train_df = tf_train_df.batch(batch_size)
    tf_val_df = tf_val_df.batch(batch_size)
    tf_test_df = tf_test_df.batch(batch_size)

    n_classes = 3  # Replace with the number of classes you have
    model = base_resnet50(input_shape=(224, 224, 3), n_classes=n_classes)
        # Compile the model with Adam optimizer 13:21
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=weighted_categorical_crossentropy(weights),
                      metrics=[CategoricalAccuracy(name='cat_accuracy'), Precision(name='precision'),
                               Recall(name='recall'),
                               AUC(name='auc')])

    name_fold = name + f'fold_{fold}'
    checkpoint_dir, callbacks = get_standard_callbacks(name_fold, learning_rate)
    model.fit(tf_train_df,
                  validation_data=tf_val_df,
                  epochs=num_epochs,
                  callbacks=callbacks)
    model.load_weights(f'{checkpoint_dir}/weights.weights.h5')
    model.evaluate(tf_test_df)
       
