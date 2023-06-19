import plotly.graph_objects as go
import numpy as np
from mlxtend.evaluate import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, class_index):
    """
    Generates a visualization of the confusion matrix using Plotly.

    Parameters:
        y_test (array-like): The true labels of the data.
        y_pred (array-like): The predicted labels of the data.
        class_index (int): Index of the positive class for the confusion matrix.

    Returns:
        fig (plotly.graph_objects.Figure): The generated confusion matrix figure.

    Example:
        fig = plot_confusion_matrix(y_test, y_pred, class_index)
    """

    class_names = ['Sentosa', 'Versicolor', 'Virginica']
    positive_label = class_names[class_index]

    confusion_mat = confusion_matrix(y_test, y_pred, binary=True, positive_label=class_index)
    confusion_mat_text = confusion_mat.astype(str)
    modified_class_names = ['Others', positive_label]

    # Create the figure
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_mat,
            x=modified_class_names,
            y=modified_class_names,
            text=confusion_mat_text,
            texttemplate="%{text}",
            textfont={"size": 20},
            hovertemplate='True label: %{y}<br>Predicted label: %{x}<br>Count: %{text}',
            colorbar=dict(title=dict(text="")),
        )
    )

    fig.update_layout(
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label'),
    )

    return fig