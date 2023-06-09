import pandas as pd
import joblib
import shap
import random
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash, html

# Load data and perform some data pre-processing

X_test = pd.read_csv("../data/x_test.csv")
X_test_enc = pd.read_csv("../data/x_test_enc.csv")
pipe_rf = joblib.load("../src/models/rf.joblib")

transparent = "#00000000"  # for transparent backgrounds
color1 = "#234075"  # blue
color2 = "#234075"  # border colors
plot_text_color = "#234075"  # plot axis and label color
title_color = "#e3a82b"  # general title and text color
border_radius = "5px"  # rounded corner radius
border_width = "3px"  # border width

external_stylesheets = [dbc.themes.YETI, '/assets/theme.css']

app = Dash(__name__, external_stylesheets=external_stylesheets,
           title="HEiDi Classifier")

server = app.server
# Define layout
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([html.Img(src="/assets/logo1.jpg", height="150px")], style={"width": "25%"}),

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
        dbc.Col(
            html.Div(
                children=[
                    html.H3("Patient Information",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                    html.Div([
                        html.Iframe(
                            id="patient-table",
                            style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px",
                                   "margin-top": "20px"}

                        ),
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "width": "100%", "height": "470px"})
                ],
                className="panel",
            )
        ),
        dbc.Col(
            html.Div(
                children=[
                    html.H3("Prediction on Individual Patient",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),

                    html.Div([
                        html.Iframe(
                            id="patient-prediction",
                            style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px",
                                   "margin-top": "20px"}
                        )
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "width": "100%", "height": "470px"}),
                ],
                className="panel",
            )
        )
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Feature Description",
                    style={"background": color1, "color": title_color,
                           'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
            html.Div([
                html.Iframe(
                    id="doctor-decision",
                    style={
                        "border-width": "1",
                        "width": "100%",
                        "height": "300px",
                        "top": "20%",
                        "left": "70%",
                        "margin-top": "25px"
                    },
                ),
            ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius, "height": "500px"})
        ], md=4, style={"width": "55%"}),

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
                                style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px",
                                       "margin-top": "20px"}
                            )
                        ],
                        style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                               "width": "100%", "height": "500px"})
                ],
                className="panel",
            )
        )
    ], style={"margin-top": "20px"}),
    dbc.Row(
        html.Div(
            [
                dbc.Button("Generate data for new patient", color="primary",
                           id="generate-button"),
            ],
            className="d-grid gap-2 col-6 mx-auto",
        )
    )
],
    className="mt-4")


@app.callback(Output("patient-table", "srcDoc"),
              Output('patient-prediction', "srcDoc"),
              Output("patient-shap", "srcDoc"),
              Output("doctor-decision", "srcDoc"),
              Input("generate-button", "n_clicks"))
def update_patient(n_clicks):
    if n_clicks is None:
        return "", "", "", ""
    else:
        num = random.randint(0, 9)
        choosen_actual = X_test.iloc[[num]].T.reset_index()
        choosen_actual.columns = ['feat_name', 'value']

        choosen_instance = X_test_enc.loc[[num]]
        out = pd.DataFrame(pipe_rf.named_steps['randomforestclassifier'].predict_proba(choosen_instance))
        out.columns = ['class A', 'class B']
        out = out.style.set_caption("Prediction Probabilities")

        rf_explainer = shap.TreeExplainer(pipe_rf.named_steps['randomforestclassifier'])
        shap_values = rf_explainer.shap_values(choosen_instance)
        force_plot = shap.force_plot(rf_explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

        phy_decision = X_test.iloc[[num]]['Physician.Disposition'].values[0]
        decision = "The Physician suggested {}".format(phy_decision)

        return choosen_actual.to_html(), out.to_html(), shap_html, decision
    # Run app


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
