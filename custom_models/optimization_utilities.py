import tensorflow as tf
from datetime import datetime
import os


def get_standard_callbacks(checkpoint_name, min_lr):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f'{checkpoint_name}_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'weights.h5'),  # Path to save the model
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=True,
        verbose=1)

    lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                 factor=0.2,
                                                                 patience=3,
                                                                 min_lr=min_lr)

    return checkpoint_dir, [checkpoint_callback, lr_scheduler_callback]




def get_recall_early(checkpoint_name, min_lr):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f'{checkpoint_name}_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'weights.h5'),  # Path to save the model
        monitor='val_combined_recall',  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=True,
        mode='max',
        verbose=1)



    lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_combined_recall',
                                                                 factor=0.2,
                                                                 patience=3,
                                                                 min_lr=min_lr)

    return checkpoint_dir, [checkpoint_callback, lr_scheduler_callback]