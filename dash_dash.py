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
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc


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


# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Create the SHAP summary plot
#shap_summary_plot = shap.summary_plot(shap_values[0], X_test, plot_type='bar', class_names=data.target_names)

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# App layout
app.layout = dbc.Container([

    dbc.Row([
        dbc.RadioItems(options=[{"label": x, "value": x} for x in [0, 1, 2]],
                       value=0,
                       inline=True,
                       id='radio-buttons-final')
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph( id='my-first-graph-final')
        ], width=6),
    ]),

], fluid=True)

# Add controls to build the interaction
@app.callback(
    Output('my-first-graph-final', 'figure'),
    Input('radio-buttons-final', 'value')
)
def update_graph(col_chosen):
    fig = shap.summary_plot(shap_values[col_chosen], X_train, plot_type="violin")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8070)
