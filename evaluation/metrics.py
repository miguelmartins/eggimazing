import numpy as np


def categorical_accuracy(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis=1))


def specificity_per_class(conf_matrix):
    specificity_per_class = []
    # Number of classes
    num_classes = conf_matrix.shape[0]

    for i in range(num_classes):
        # True Positives for class i
        TP = conf_matrix[i, i]

        # False Positives for class i
        FP = np.sum(conf_matrix[:, i]) - TP

        # False Negatives for class i
        FN = np.sum(conf_matrix[i, :]) - TP

        # True Negatives for class i
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        # Specificity for class i
        specificity = TN / (TN + FP)
        specificity_per_class.append(specificity)
    specificity_per_class = np.array(specificity_per_class)
    return specificity_per_class


def sensitivity_per_class(conf_matrix):
    # Initialize an array to store sensitivity for each class
    sensitivity_per_class = []

    # Number of classes
    num_classes = conf_matrix.shape[0]

    for i in range(num_classes):
        # True Positives for class i
        TP = conf_matrix[i, i]

        # False Negatives for class i
        FN = np.sum(conf_matrix[i, :]) - TP

        # Sensitivity for class i
        sensitivity = TP / (TP + FN)
        sensitivity_per_class.append(sensitivity)
    return np.array(sensitivity_per_class)

