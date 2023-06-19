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

def metrics_table():
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

    table_data = [
        {"Metric": "Accuracy", "Value": f"{accuracy:.2f}"},
        {"Metric": "Precision", "Value": f"{precision:.2f}"},
        {"Metric": "Recall", "Value": f"{recall:.2f}"},
        {"Metric": "F1-Score", "Value": f"{f1:.2f}"},
    ]

    return table_data