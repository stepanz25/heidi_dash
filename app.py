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

    dbc.Row([
        dbc.Col([html.Img(src="src/assets/logo1.jpg", height="150px")], style={"width": "25%"}),

        dbc.Col([html.H1('Classifier Explainer: Predicting HEiDi Triage', className='text-center',
                         style={'color': '#234075', 'textAlign': 'center', "font-weight": "bold", "fontSize": 40,
                                "margin-top": "80px"})], md=4, style={"color": "#234075", "width": "70%"}),
        dbc.Col([
            dbc.Button(
                "ⓘ",
                id="popover-target",
                className="sm",
                style={"border": color2, "background": f"{color1}95"},
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader("Welcome to Classifier Explainer!"),
                    dbc.PopoverBody([
                        html.P("This dashboard contains:"),
                        html.P("🏥 "),
                        html.P("🏥 "),
                        html.P("🏥 ")
                    ]),
                    dbc.PopoverBody([
                        html.P(" "),
                        html.P("🏥 ")
                    ])
                ],
                target="popover-target",
                trigger="legacy",
                placement="bottom"
            )
        ], style={'textAlign': 'right'})
    ]),

    dbc.Row([
            html.Div(
                children=[
                    html.H3("Prediction on Individual Patient",
                            style={"background": color1, "color": title_color,
                                'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),

                    html.Div([
                        html.Iframe(
                            id="patient-prediction",
                            style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "25%",
                                "margin-top": "20px", "text-align": "center"}
                        )
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                            "width": "100%", "height": "150px"}),

                    html.Div(
                        id="prediction-label",
                        children=[
                            html.Div(
                                id="prediction-text",
                                style={'border': '0', 'width': '100%', 'height': '40px', "margin-left": "100px",
                                "margin-top": "100px", "text-align": "center"}
                            )
                        ],
                        style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                            "width": "100%", "height": "35px", "margin-top": "20px"}
                    )
                ],
                className="panel",
            )
        ], style={"margin-top": "20px", "text-align": "center"}),

    dbc.Row([    
        dbc.Col(
            html.Div(
                children=[
                    html.H3("Patient Information",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%",
                                   "margin-top": "20px"}),
                    html.Div([
                        html.Iframe(
                            id="patient-table",
                            style={
                                "border-width": "1",
                                "width": "100%",
                                "height": "300px",
                                "top": "20%",
                                "left": "70%",
                                "margin-top": "0px"
                            },
                        ),
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "height": "310px"})
                ],
                className="panel",
            )
        ),

        dbc.Col(
            html.Div(
                children=[
                    html.H3("Feature Description",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%",
                                   "margin-top": "20px"}),
                    html.Div([
                        html.Iframe(
                            id="feature-table",
                            style={
                                "border-width": "1",
                                "width": "100%",
                                "height": "300px",
                                "top": "20%",
                                "left": "70%",
                                "margin-top": "0px"
                            },
                        ),
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "height": "310px"})
                ],
                className="panel",
            )
        )]),

    dbc.Row([
        dbc.Col(
            html.Div(
                children=[
                    html.H3("Feature Importances",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                    html.Div(
                        [
                            html.Iframe(
                                id="patient-shap",
                                style={'border': '0', 'width': '100%', 'height': '200px', "margin-left": "0px",
                                       "margin-top": "20px"}
                            )
                        ],
                        style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                               "width": "100%", "height": "190px"}),
                ],
                className="panel",
)
        )
    ], style={"margin-top": "20px"}),
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
    ),
],
    className="mt-4",
    
    )


@app.callback(Output("patient-table", "srcDoc"),
              Output('patient-prediction', "srcDoc"),
              Output("feature-table", "srcDoc"),
              Output("patient-shap", "srcDoc"),
              Output("prediction-label", "children"),
              Input("generate-button", "n_clicks"))

def update_patient(n_clicks):
    num=5
    
    if n_clicks is None:
        return "", "", "", "", ""
    else:
        choosen_actual = X_test.iloc[[num]].T.reset_index()
        choosen_actual.columns = ['Feature Name', 'Value']
        choosen_actual.index = choosen_actual.index + 1
        styled_df = choosen_actual.style \
            .set_properties(**{'text-align': 'center', 'font-size': '14px', 'width': '290px', 'height': '7px',
                               'font-family': 'Helvetica'}) \
            .set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#234075'), ('color', 'white'), ('font-family', 'Helvetica')]},
            {'selector': 'td', 'props': [('border', '1px solid #e3a82b'), ('font-family', 'Helvetica')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('font-size', '20px'),
                                              ('font-family', 'Helvetica'), ('color', '#234075'),
                                              ('margin-bottom', '10px')]}
        ])

        choosen_instance = X_test_enc.loc[[num]]
        probability = pipe_rf.named_steps['randomforestclassifier'].predict_proba(choosen_instance)
        probability_red = probability[0][0] * 100
        probability_green = probability[0][1] * 100
        prediction = pipe_rf.named_steps['randomforestclassifier'].classes_[np.argmax(probability)]

        X = 50.0

        if probability_red >= X:
            prediction_label = html.Div(
                [
                    "Physician is >= {:.2f}% likely to classify this individual as ".format(probability_red),
                    html.Span("RED", style={"color": "red", "text-transform": "uppercase", "font-weight": "bold"}),
                    " - requiring immediate emergency attention."
                ],
                style={"font-size": "larger"}
            )
        elif probability_green >= X:
            prediction_label = html.Div(
                [
                    "Physician is >= {:.2f}% likely to classify this individual as ".format(probability_green),
                    html.Span("GREEN", style={"color": "green", "text-transform": "uppercase", "font-weight": "bold"}),
                    " - being safe to manage in the community."
                ], 
                style={"font-size": "larger"}
            )
        else:
            prediction_label = html.Div(
                "The model is unable to predict with >=50% confidence, therefore no recommendation has been provided.",
                style={"font-size": "larger"}
            )

        out = pd.DataFrame(probability, columns=['class A', 'class B'])
        out = out.style \
            .set_properties(**{'text-align': 'center', 'font-size': '14px', 'width': '300px', 'height': '7px',
                               'font-family': 'Helvetica'}) \
            .set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#234075'), ('color', 'white'), ('font-family', 'Helvetica')]},
            {'selector': 'td', 'props': [('border', '1px solid #e3a82b'), ('font-family', 'Helvetica')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('font-size', '20px'),
                                              ('font-family', 'Helvetica'), ('color', '#234075'),
                                              ('margin-bottom', '10px')]}
        ]) \
            .set_caption(f'\nSoft Predictions for Patient with Index {num}\n')
        out = out.hide()
        out = out.format('{:.3f}')

        feature_data.reset_index(drop=True, inplace=True)
        feature_data.index = feature_data.index + 1
        feature_table = feature_data.style \
            .set_properties(
            **{'text-align': 'center', 'font-size': '14px', 'height': '7px', 'font-family': 'Helvetica'}) \
            .set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#234075'), ('color', 'white'), ('font-family', 'Helvetica')]},
            {'selector': 'td', 'props': [('border', '1px solid #e3a82b'), ('font-family', 'Helvetica')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('font-size', '20px'),
                                              ('font-family', 'Helvetica'), ('color', '#234075'),
                                              ('margin-bottom', '10px')]},
            {'selector': '.col0', 'props': [('width', '40%')]},  # Adjust the width of the first column
            {'selector': '.col1', 'props': [('width', '60%')]},  # Adjust the width of the second column
        ])

        rf_explainer = shap.TreeExplainer(pipe_rf.named_steps['randomforestclassifier'])
        shap_values = rf_explainer.shap_values(choosen_instance)
        force_plot = shap.force_plot(rf_explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body><div style='color: #234075; font-family: Helvetica; font-size: 14px;'>{force_plot.html()}</div></body>"

        #phy_decision = X_test.iloc[[num]]['Physician.Disposition'].values[0]
        #decision = "The Physician suggested {}".format(phy_decision)

        #print(decision)

        return styled_df.to_html(escape=False), out.to_html(), feature_table.to_html(), shap_html, prediction_label

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)