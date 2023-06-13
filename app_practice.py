import numpy as np
import pandas as pd
import joblib
import shap
import random
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash, html, dcc
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
import io
import plotly.io as pio
# Load data and perform some data pre-processing

X_test = pd.read_csv("data/x_test.csv")
X_test_enc = pd.read_csv("data/x_test_enc.csv")
feature_data = pd.read_csv("data/feature_data.csv")
pipe_rf = joblib.load("src/models/rf.joblib")


transparent = "#00000000"  # for transparent backgrounds
color1 = "#234075"  # blue
color2 = "#234075"  # border colors
plot_text_color = "#234075"  # plot axis and label color
title_color = "#e3a82b"  # general title and text color
border_radius = "5px"  # rounded corner radius
border_width = "3px"  # border width

my_external_stylesheets = [dbc.themes.YETI, 'src/assets/theme.css']

app = Dash(__name__, external_stylesheets=my_external_stylesheets,
           title="HEiDi Classifier")

server = app.server

# Define layout
app.layout = dbc.Container([
    dbc.Row([dbc.Col([
            dcc.Graph(id='fig')
        ], width={'size': 3, 'offset': 0})
    ], className='mb-4 mt-4'),
    dbc.Row(
        html.Div(
            [
                dbc.Button(
                    "Generate Data for New Patient",
                    color="primary",
                    id="generate-button",
                    style={
                        "backgroundColor": "#234075",
                        "marginTop": "20px",
                        "color": "#e3a82b",
                        # "fontWeight": "bold",
                        "fontSize": "20px"
                    }
                )
            ],
            className="d-grid gap-2 col-6 mx-auto",
        )
    )
    
   ])


@app.callback(Output("fig", "figure"),
              Input("generate-button", "n_clicks"))

def update_patient(n_clicks):
    if n_clicks is None:
        return go.Figure()
    else:
        data = np.array([[0.97, 0.03]])

        labels = ['Class A', 'Class B']
        colors = ['red', 'green']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=[0],
            x=data[0],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{val:.2f}' for val in data[0]],
            textposition='auto',
            textfont=dict(color='white', size=20),
            name=labels[0]
        ))

        fig.add_trace(go.Bar(
            y=[0],
            x=[1 - data[0][0]],
            orientation='h',
            marker=dict(color=colors[::-1]),
            text=[f'{val:.2f}' for val in [1 - data[0][0]]],
            textposition='auto',
            textfont=dict(color='white', size=20),
            name=labels[1]
        ))

        fig.update_layout(
            barmode='stack',
            yaxis=dict(showticklabels=False),
            xaxis=dict(
                tickvals=np.arange(0, 1.1, 0.1),
                title='Probability'
            ),
            title='Prediction Probabilities',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # Convert the plot to HTML
        #patient_pred = pio.to_html(fig)

        return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)