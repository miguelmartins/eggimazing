import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_categorical_crossentropy(weights):
    def loss_fn(y_true, y_pred):
        # Compute the standard categorical crossentropy loss
        cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply the weights to the loss
        weights_applied = tf.reduce_sum(weights * y_true, axis=-1)

        return cce_loss * weights_applied

    return loss_fn


def focal_loss_with_class_weights(alpha, gamma=2.0):
    """
    Combined Focal Loss with per-class weighting.

    Parameters:
    - alpha: A tensor or array of shape (num_classes,) specifying the weight for each class.
    - gamma: Focusing parameter for hard examples.

    Returns:
    - A loss function that can be used in model.compile()
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Convert labels to one-hot encoding if they are sparse
        if y_true.shape != y_pred.shape:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])

        # Calculate cross-entropy loss per class
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Apply per-class alpha weights
        weights = alpha * y_true  # Shape: (batch_size, num_classes)

        # Compute the focal modulation factor
        focal_modulation = tf.pow(1 - y_pred, gamma) * y_true  # Shape: (batch_size, num_classes)

        # Combine the focal loss with class weights
        loss = weights * focal_modulation * cross_entropy  # Shape: (batch_size, num_classes)

        # Sum the losses over classes
        loss = tf.reduce_sum(loss, axis=1)  # Shape: (batch_size,)

        # Return the mean loss over the batch
        return tf.reduce_mean(loss)

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
