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
import matplotlib.pyplot as plt
import io

def sum_shap(feature_index):
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[feature_index], X_train, show=False)

    image_filename = "plot.png"
    plt.savefig(image_filename)
    plt.close(fig)

    return image_filename

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

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

# Calculate percentages for each intersection
confusion_mat_percent = confusion_mat / confusion_mat.astype(float).sum(axis=1, keepdims=True)

# Convert confusion matrix to string format
confusion_mat_text = confusion_mat.astype(str)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Obtain feature importances
feature_importances = eli5.format_as_dataframe(eli5.explain_weights(model, feature_names=data.feature_names))

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Define the layout
app.layout = dbc.Container([
    html.H1("XGBoost Classification Dashboard", className="text-center mt-4 mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Performance Metrics"),
                dbc.CardBody([
                    dbc.Table([
                        html.Tr([html.Th('Accuracy'), html.Td(f"{accuracy:.2f}")]),
                        html.Tr([html.Th('Precision'), html.Td(f"{precision:.2f}")]),
                        html.Tr([html.Th('Recall'), html.Td(f"{recall:.2f}")]),
                        html.Tr([html.Th('F1-Score'), html.Td(f"{f1:.2f}")])
                    ], className="table table-striped text-center", bordered=True, color="light")
                ], style={'border': '0', 'width': '100%', 'height': '400px', "margin-left": "0px",
                                "margin-top": "100px", "text-align": "center"})
            ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '550px', "margin-left": "0px",
                                "margin-top": "0px", "text-align": "center"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Confusion Matrix"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Heatmap(
                                    z=confusion_mat_percent,
                                    x=data.target_names,
                                    y=data.target_names,
                                    text=confusion_mat_text,
                                    hovertemplate='True label: %{y}<br>Predicted label: %{x}<br>Count: %{text}',
                                    colorbar=dict(title=dict(text="")),
                                )
                            ],
                            layout=go.Layout(
                                xaxis=dict(title="Predicted label"),
                                yaxis=dict(title="True label"),
                            )
                        ),
                        config={'displayModeBar': False}
                    )
                ], style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "0px",
                                "margin-top": "0px", "text-align": "center"})
            ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '550px', "margin-left": "0px",
                                "margin-top": "0px", "text-align": "center"})
        ], width=6)
    ]),
    html.Div(className="mt-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature Importances"),
                dbc.CardBody([
                    dbc.Table.from_dataframe(feature_importances, striped=True, bordered=True, hover=True)
                ])
            ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '500px', "margin-left": "0px",
                                "margin-top": "0px", "text-align": "center"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("SHAP Values"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="dropdown",
                        options=[
                            {"label": "Option 0", "value": 0},
                            {"label": "Option 1", "value": 1},
                            {"label": "Option 2", "value": 2}
                        ],
                        value=2
                    ),
                    html.Div(id="plot-container"),
                ])
            ], className="mb-4", style={'border': '1', 'width': '100%', 'height': '500px', "margin-left": "0px",
                                "margin-top": "0px", "text-align": "center"})
        ], width=6)
    ])
], className="p-4")

# Define the callback function
@app.callback(Output("plot-container", "children"), [Input("dropdown", "value")])

def update_shap_plot(feature_index):
    # Calculate SHAP values for the selected feature
    summary_plot = sum_shap(feature_index)
    
    with open(summary_plot, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Display the image
    return html.Img(src="data:image/png;base64,{}".format(encoded_image), style={'height': '250px', 'width': '100%', "margin-top": "60px"})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)