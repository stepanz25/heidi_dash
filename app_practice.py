import numpy as np
import pandas as pd
import xgboost as xgb
import dash
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc
import eli5
import shap
from dash.dependencies import Input, Output
import base64
import plotly.figure_factory as ff
import dash_table

#from shap_violin import plotly_shap_violin_plot
from shap_dependance import plotly_dependence_plot
from shap_scatter import plotly_shap_scatter_plot
from confusion_matrix import plot_confusion_matrix
from metrics_by_class import generate_metrics_table
from importances import plotly_importances_plot
from metrics_by_class import generate_metrics_table

def generate_metrics_table_by_class(test, pred, class_index):
    """
    Generates a table of metrics for a specific class in a classification problem.

    Args:
        test (array-like): True labels for the test set.
        pred (array-like): Predicted labels for the test set.
        class_index (int): Index of the class for which metrics are calculated.

    Returns:
        html.Div: A Div element containing a DataTable with the metrics table.

    """
    table = generate_metrics_table(y_test=test, y_pred=pred, class_index=class_index)
    return html.Div(
        dash_table.DataTable(
            data=table.to_dict("rows"),
            columns=[{"id": x, "name": x} for x in table.columns],
            style_data={
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_table={
                "overflowX": "auto",
                "minWidth": "100%",
                "margin-top": "200px",
            },
            style_cell={
                "textAlign": "center",
                "font-family": "Arial, sans-serif",
                "font-size": "14px",
                "padding": "8px",
            },
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
        ),
        id="update-table",
    )


def create_feature_importances_plot(feature_importances):
    """
    Creates a plot of feature importances.

    Args:
        feature_importances (array-like): Feature importances.

    Returns:
        plotly.graph_objects.Figure: A Figure object representing the feature importances plot.

    """
    fig = plotly_importances_plot(feature_importances)
    return fig


def create_confusion_matrix(test, pred, class_index):
    """
    Creates a confusion matrix plot for a specific class in a classification problem.

    Args:
        test (array-like): True labels for the test set.
        pred (array-like): Predicted labels for the test set.
        class_index (int): Index of the class for which the confusion matrix is calculated.

    Returns:
        plotly.graph_objects.Figure: A Figure object representing the confusion matrix plot.

    """
    fig = plot_confusion_matrix(test=test, pred=pred, class_index=class_index)
    return fig


def create_dependence_plot(class_index, index_feature, feature_column):
    """
    Creates a dependence plot showing the relationship between a feature and its SHAP values.

    Args:
        class_index (int): Index of the class for which the dependence plot is created.
        index_feature (int): Index of the feature in the dataset.
        feature_column (str): Name of the feature column.

    Returns:
        plotly.graph_objects.Figure: A Figure object representing the dependence plot.

    """
    df = pd.DataFrame(X_test, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    shap_df = pd.DataFrame(shap_values[class_index])[index_feature].values
    fig = plotly_dependence_plot(df[feature_column], shap_values=shap_df)
    return fig


def create_shap_scatter_plot(class_index):
    """
    Creates a scatter plot of SHAP values for a specific class.

    Args:
        class_index (int): Index of the class for which the scatter plot is created.

    Returns:
        plotly.graph_objects.Figure: A Figure object representing the SHAP scatter plot.

    """
    df = pd.DataFrame(X_test, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    shap_df = pd.DataFrame(shap_values[class_index],
                           columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    fig = plotly_shap_scatter_plot(df, shap_values_df=shap_df)
    return fig


# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
features = data.feature_names
columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost classifier
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
confusion_mat = confusion_matrix(y_test, y_pred)

table_data = pd.DataFrame.from_dict(
    {"Accuracy": [f"{accuracy:.2f}"],
    "Precision": [f"{precision:.2f}"],
    "Recall": [f"{recall:.2f}"],
    "F1-Score": [f"{f1:.2f}"]}
)
# Calculate percentages for each intersection
confusion_mat_percent = confusion_mat / confusion_mat.astype(float).sum(axis=1, keepdims=True)

# Convert confusion matrix to string format
confusion_mat_text = confusion_mat.astype(str)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Obtain feature importances
feature_importances = eli5.format_as_dataframe(eli5.explain_weights(model, feature_names=data.feature_names))

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Define the layout
app.layout = dbc.Container([
    html.H1("XGBoost Classification Dashboard", className="text-center mt-4 mb-4"),
    dbc.Card(
    [
        dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Model Evaluation",
                        id="tooltip-target",
                        style={"cursor": "pointer", "fontSize": "27px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "This section provides an overview of the evaluation process and performance metrics for XGBoost machine learning model. It aims to assess the model's accuracy and effectiveness in making predictions for classification problem.",
                target="tooltip-target",
                placement="top"
            ),
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Overall Performance Metrics",
                        id="tooltip-target1",
                        style={"cursor": "pointer", "fontSize": "20px", "verticalAlign": "middle"},
                    ),
                ],
            )),
            dbc.Tooltip(
    children=html.Div(
        [
            html.P("Overall Performance Metrics:"),
            html.Ul(
                [
                    html.Li("Accuracy: The percentage of correct predictions made by the model."),
                    html.Li("Precision: The proportion of correctly predicted positive instances out of all positive predictions."),
                    html.Li("Recall: The proportion of correctly predicted positive instances out of all actual positive instances."),
                    html.Li("F1 Score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.")
                ]
            )
        ],
        style={"line-height": "1.5"},
    ),
    target="tooltip-target1",
    autohide=False,
),
                                    dbc.CardBody(
                                        html.Div(
                                            dash_table.DataTable(
                                                data=table_data.to_dict("rows"),
                                                columns=[
                                                    {"id": x, "name": x} for x in table_data.columns
                                                ],
                                                style_data={
                                                    "whiteSpace": "normal",
                                                    "height": "auto",
                                                },
                                                style_table={
                                                    "overflowX": "auto",
                                                    "minWidth": "100%",
                                                    "margin-top": "0px",
                                                },
                                                style_cell={
                                                    "textAlign": "center",
                                                    "font-family": "Arial, sans-serif",
                                                    "font-size": "12px",
                                                    "padding": "8px",
                                                },
                                                style_header={
                                                    "backgroundColor": "rgb(230, 230, 230)",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            id="update-table",
                                        ),
                                    ),
                                ],
                                className="mb-4",
                                style={
                                    "border": "1",
                                    "width": "100%",
                                    "height": "160px",
                                    "margin-left": "0px",
                                    "margin-top": "0px",
                                    "text-align": "center",
                                },
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Global Feature Importances",
                        id="tooltip-target3",
                        style={"cursor": "pointer", "fontSize": "20px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "The feature importances highlight the significance of different features in the model's predictions. It identifies which features have the most influence on the model's output, allowing us to understand the key factors driving the predictions.",
                target="tooltip-target3",
                placement="top"
            ),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id="global_feature_plot",
                                            config={"displayModeBar": False},
                                            style={"height": "370px", "margin-top": "0px"},  # Adjust the height of the graph as needed
                                        ),
                                    ),
                                ],
                                className="mb-4",
                                style={
                                    "border": "1",
                                    "width": "100%",
                                    "height": "465px",
                                    "margin-left": "0px",
                                    "margin-top": "0px",  # Add margin-top to create space
                                    "text-align": "center",
                                },
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Confusion Matrix",
                        id="tooltip-target4",
                        style={"cursor": "pointer", "fontSize": "20px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "The confusion matrix shows the number of true negatives (predicted negative, observed negative), true positives (predicted positive, observed positive), false negatives (predicted negative, but observed positive) and false positives (predicted positive, but observed negative). The amount of false negatives and false positives determine the costs of deploying and imperfect model.",
                target="tooltip-target4",
                placement="top"
            ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                figure=go.Figure(
                                                    data=[
                                                        go.Heatmap(
                                                            z=confusion_mat_percent,
                                                            x=data.target_names,
                                                            y=data.target_names,
                                                            text=confusion_mat_text,
                                                            texttemplate="%{text}",
                                                            textfont={"size": 20},
                                                            hovertemplate='True label: %{y}<br>Predicted label: %{x}<br>Count: %{text}',
                                                            colorbar=dict(title=dict(text="")),
                                                        )
                                                    ],
                                                    layout=go.Layout(
                                                        xaxis=dict(title="Predicted label"),
                                                        yaxis=dict(title="True label"),
                                                    ),
                                                ),
                                                config={"displayModeBar": False},
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                                style={
                                    "border": "1",
                                    "width": "100%",
                                    "height": "650px",
                                    "margin-left": "0px",
                                    "margin-top": "0px",
                                    "text-align": "center",
                                },
                            ),
                        ],
                        width=6,
                    ),
                ]
            )
        ),
    ],
    className="mb-4",
    style={
        "border": "1",
        "width": "100%",
        "margin-left": "0px",
        "margin-top": "0px",
        "text-align": "center",
    },
),
    html.Div(className="mt-4"),
    dbc.Card(
        [
            dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Class Evaluation",
                        id="tooltip-target5",
                        style={"cursor": "pointer", "fontSize": "27px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "Class evaluation contains information related to the evaluation of a classification model's performance for each individual class or category in the dataset. It provides insights into how well the model performs for each specific class, allowing for a more detailed analysis of its predictive capabilities.",
                target="tooltip-target5",
                placement="top"
            ),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Select Positive Class:", style={"font-size": "18px", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id="dropdown",
                                options=[
                                    { 
                                        "label": html.Span(['Sentosa'], style={'color': 'Red', 'font-size': 15}),
                                        "value": 0
                                    },
                                    {
                                        "label": html.Span(['Versicolor'], style={'color': 'Orange', 'font-size': 15}),
                                        "value": 1
                                    },
                                    {
                                        "label": html.Span(['Virginica'], style={'color': 'Green', 'font-size': 15}),
                                        "value": 2
                                    },
                                ],
                                placeholder="Select a Class",
                                value=1
                            ),
                        ]),
                        html.Div(id="update-table")
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Confusion Matrix by Class",
                        id="tooltip-target6",
                        style={"cursor": "pointer", "fontSize": "20px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "The confusion matrix by class shows the number of true negatives (predicted negative, observed negative), true positives (predicted positive, observed positive), false negatives (predicted negative, but observed positive) and false positives (predicted positive, but observed negative) by class. The amount of false negatives and false positives determine the costs of deploying and imperfect model.",
                target="tooltip-target6",
                placement="top"
            ),
                            dbc.CardBody([
                                html.Div(dcc.Graph(id='confusion_matrix')),  # Confusion Matrix by Class
                            ])
                        ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '550px',
                                                    "margin-left": "0px", "margin-top": "0px", "text-align": "center"})
                    ], width=6)
                ])
            )
        ],
        className="mb-4",
        style={'border': '1', 'width': '100%', 'margin-left': '0px', 'margin-top': '0px', 'text-align': 'center'}
    ),
    html.Div(className="mt-4"),
    dbc.Card(
        [
            dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Feature Analysis",
                        id="tooltip-target7",
                        style={"cursor": "pointer", "fontSize": "27px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "The feature analysis contains information related to the analysis of feature importance in a machine learning model. It provides insights into the relevance and contribution of different features or variables in predicting the target variable.",
                target="tooltip-target7",
                placement="top"
            ),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Select Positive Class:", style={"font-size": "18px", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id="dropdown_dup",
                                options=[
                                    { 
                                        "label": html.Span(['Sentosa'], style={'color': 'Red', 'font-size': 15}),
                                        "value": 0
                                    },
                                    {
                                        "label": html.Span(['Versicolor'], style={'color': 'Orange', 'font-size': 15}),
                                        "value": 1
                                    },
                                    {
                                        "label": html.Span(['Virginica'], style={'color': 'Green', 'font-size': 15}),
                                        "value": 2
                                    },
                                ],
                                placeholder="Select a Class",
                                value=1
                            ),
                        ], style={"margin-bottom": "20px"}),
                        html.Div([
                            html.Label("Select Feature 1:", style={"font-size": "18px", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id='dropdown_index_feature',
                                options=[{'label': feature, 'value': i} for i, feature in enumerate(features)],
                                placeholder="Select a Feature:",
                                value=2
                            ),
                        ], style={"margin-bottom": "20px"}),
                        html.Div([
                            html.Label("Select Feature 2:", style={"font-size": "18px", "font-weight": "bold"}),
                            dcc.Dropdown(
                                id='dropdown_feature_column',
                                options=[{'label': col, 'value': col} for col in columns],
                                placeholder="Select a Feature:",
                                value="SepalLengthCm"
                            ),
                        ], style={"margin-bottom": "20px"}),
                        dbc.Card([
                            dbc.CardHeader(html.P(
                [
                    html.Span(
                        "Dependence Plot",
                        id="tooltip-target8",
                        style={"cursor": "pointer", "fontSize": "20px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "The dependence plot shows the relationship between a specific feature and the predicted outcome of a machine learning model. It helps us understand how changes in the feature's values impact the model's predictions.",
                target="tooltip-target8",
                placement="top"
            ),
                            dbc.CardBody([
                                html.Div(dcc.Graph(id='plot-dependency'))
                            ]),
                        ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '550px',
                                                    "margin-left": "0px", "margin-top": "25px", "text-align": "center"})
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.P(
                [
                    html.Span(
                        "SHAP Values",
                        id="tooltip-target9",
                        style={"cursor": "pointer", "fontSize": "20px"},
                    ),
                ],
            )),
            dbc.Tooltip(
                "A SHAP (SHapley Additive exPlanations) value plot is a visual representation of the SHAP values associated with each feature in a machine learning model. SHAP values provide a way to explain the prediction of an individual instance by quantifying the contribution of each feature to the prediction.",
                target="tooltip-target9",
                placement="top"
            ),
                            dbc.CardBody([
                                html.Div(dcc.Graph(id='plot-scatter')),
                            ])
                        ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '805px',
                                                    "margin-left": "0px", "margin-top": "0px", "text-align": "center"})
                    ], width=6)
                ])
            )
        ],
        className="mb-4",
        style={'border': '1', 'width': '100%', 'margin-left': '0px', 'margin-top': '0px', 'text-align': 'center'}
    )
], className="p-4")



# Define the callback function
@app.callback(Output("update-table", "children"),
              Output("confusion_matrix", "figure"),
              Output("global_feature_plot", "figure"),
              Input("dropdown", "value")
)

def update_metrics(class_index):
    
    # Calculate SHAP values for the selected feature
    confusion_matrix = plot_confusion_matrix(y_test, y_pred, class_index)
    metrics_table = generate_metrics_table_by_class(y_test, y_pred, class_index)
    global_importances = plotly_importances_plot(feature_importances)
    
    return metrics_table, confusion_matrix, global_importances

@app.callback(
              Output("plot-scatter", "figure"),
              Output("plot-dependency", "figure"),
              Input("dropdown_dup", "value"),
              Input("dropdown_index_feature", "value"),
              Input("dropdown_feature_column", "value")
)

def update_shap_plot(class_index_dup, index_feature, feature_column):

    # Calculate SHAP values for the selected feature
    dependency_plot = create_dependence_plot(class_index_dup, index_feature, feature_column)
    summary_plot = create_shap_scatter_plot(class_index_dup)
    
    return summary_plot, dependency_plot


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)