import tensorflow as tf


def weighted_categorical_crossentropy(weights):
    def loss_fn(y_true, y_pred):
        # Compute the standard categorical crossentropy loss
        cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply the weights to the loss
        weights_applied = tf.reduce_sum(weights * y_true, axis=-1)

        return cce_loss * weights_applied

    return loss_fn


def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Calculate binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Get class weights
        weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]

        # Multiply the loss by the class weights
        return bce * weights

    return loss
