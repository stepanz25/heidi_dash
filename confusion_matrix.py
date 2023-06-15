from mlxtend.evaluate import confusion_matrix
import plotly.graph_objects as go
import numpy as np


def plot_confusion_matrix(y_test, y_pred):
    """
    Generates a visualization of the confusion matrices for multiple positive labels using Plotly.

    Parameters:
        y_test (array-like): The true labels of the data.
        y_pred (array-like): The predicted labels of the data.

    Returns:
        fig (plotly.graph_objects.Figure): The generated confusion matrix figure.

    Example:
        fig = plot_confusion_matrix(y_test, y_pred)
    """

    class_names = ['Sentosa', 'Versicolor', 'Virginica']
    positive_labels = [0, 1, 2]

    # Create the figure
    fig = go.Figure()

    for positive_label in positive_labels:
        confusion_mat = confusion_matrix(y_test, y_pred, binary=True, positive_label=positive_label)
        
        positive_class_name = class_names[positive_label]
        other_class_names = [name for name in class_names if name != positive_class_name]
        modified_class_names = ['Others', positive_class_name]

        # Add heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=confusion_mat,
                x=modified_class_names,
                y=modified_class_names,
                text=confusion_mat,
                hovertemplate='True label: %{y}<br>Predicted label: %{x}<br>Count: %{text}',
                colorbar=dict(title=dict(text="")),
                visible=False  # Set the heatmap initially invisible
            )
        )
    
    # Set the visibility of the first heatmap to True
    fig.data[0].visible = True
    
    # Create the dropdown menu
    dropdown_buttons = []
    for i, positive_label in enumerate(positive_labels):
        dropdown_buttons.append(
            dict(
                label=class_names[positive_label],
                method="update",
                args=[{"visible": [i == j for j in range(len(positive_labels))]}],
            )
        )
    
    # Update the layout with the dropdown menu
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label'),
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": -100, "t": -30},
            showactive=True,
            x=0.95,
            xanchor="right",
            y=1.1,
            yanchor="top"
        )],
    )

    return fig