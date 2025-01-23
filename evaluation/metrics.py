import numpy as np


def categorical_accuracy(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis=1))


def compute_metrics(y_true, y_pred, y_true_ordinals, y_pred_ordinals,
                    metrics):  # split y_true before and call this for each model
    y_true_ordinal = np.argmax(y_true, axis=-1)  # [0 0 1] -> 2
    y_pred_ordinal = np.argmax(y_pred, axis=-1)
    y_true_ordinals.append(y_true_ordinal)
    y_pred_ordinals.append(y_pred_ordinal)
    y_pred_one_hot = np.zeros_like(y_pred)
    y_pred_one_hot[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1  # [0.2, 0.2, 0.6] -> [0, 0, 1]
    conf_matrix = confusion_matrix(y_true_ordinal, y_pred_ordinal,
                                   labels=[0, 1, 2])
    metrics.append([accuracy_per_class(conf_matrix),
                    specificity_per_class(conf_matrix), sensitivity_per_class_(conf_matrix),
                    conf_matrix])
    return metrics


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
        if TN + FN != 0:
            specificity = TN / (TN + FP)
        else:
            specificity = np.nan

        specificity_per_class.append(specificity)

    specificity_per_class = np.array(specificity_per_class)
    return specificity_per_class


def accuracy_per_class(conf_matrix):
    accuracy_per_class = []
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
        if TP + TN + FP + FN != 0:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        else:
            accuracy = np.nan

        accuracy_per_class.append(accuracy)

    accuracy_per_class = np.array(accuracy_per_class)
    return accuracy_per_class


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
        if TP + FN != 0:
            sensitivity = TP / (TP + FN)
        else:
            sensitivity = np.nan
        sensitivity_per_class.append(sensitivity)
    return np.array(sensitivity_per_class)
