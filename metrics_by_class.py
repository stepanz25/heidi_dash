from mlxtend.evaluate import confusion_matrix
import numpy as np
import pandas as pd

def generate_metrics_table(y_test, y_pred, class_index):
    """
    Generates a metrics table for the specified class index.

    Parameters:
        y_test (array-like): The true labels of the data.
        y_pred (array-like): The predicted labels of the data.
        class_index (int): Index of the class for which the metrics table is generated.

    Returns:
        metrics_df (pd.DataFrame): Data frame containing metrics (accuracy, precision, recall, and F1 score)
                                   for the specified class index.
    """
    positive_label = class_index
    

    confusion_mat = confusion_matrix(y_test, y_pred, binary=True, positive_label=positive_label)

    tn, fp, fn, tp = confusion_mat.ravel()

    accuracy = round((tp + tn) / (tp + tn + fp + fn), 2)
    precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)
    f1_score = round(2 * (precision * recall) / (precision + recall), 2)

    metrics_dict = {
        'Accuracy': [f"{accuracy:.2f}"],
        'Precision': [f"{precision:.2f}"],
        'Recall': [f"{recall:.2f}"],
        'F1-Score': [f"{f1_score:.2f}"],
    }

    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df