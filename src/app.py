import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import shap
import eli5
import vaex
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash import Dash, html, dcc, dash_table
import dash_daq as daq


# Load data some data pre-processing
df = pd.read_csv('../data/master.csv')
# df = df.query('suicides_100k_pop != 0')


# Create app

external_stylesheets = [dbc.themes.YETI, '/assets/theme.css']

app = Dash(__name__, external_stylesheets=external_stylesheets,
           title="HEiDi Classifier")

server = app.server

transparent = "#00000000"        # for transparent backgrounds
color1 = "#234075"               # blue
color2 = "#234075"               # border colors
plot_text_color = "#ebe8e8"      # plot axis and label color
title_color = "#e3a82b"          # general title and text color
border_radius = "5px"            # rounded corner radius
border_width = "3px"             # border width

# Define layout
app.layout = dbc.Container([
    
    dbc.Row([
                dbc.Col([html.Img(src="/assets/logo1.jpg", height="150px")], style={"width": "25%"}),
                
                dbc.Col([html.H1('Classifier Explainer: Predicting HEiDi Triage', className='text-center',
                                 style = {'color':'#234075','textAlign':'center', "font-weight": "bold", "fontSize": 40, "margin-top": "80px"})], md=4, style={"color": "#234075", "width": "70%"}),
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
                                html.P(
                                    "🏥 "),
                                html.P(
                                    "🏥 "),
                                html.P(
                                    "🏥 ")
                            ]),
                            dbc.PopoverBody([
                                html.P(" "),
                                html.P(
                                    "🏥 ")

                            ])
                        ],
                        target="popover-target",
                        trigger="legacy",
                        placement="bottom"
                    )
                ], style = {'textAlign': 'right'})
    ]),

    
    dbc.Row([
                    dbc.Col(
                        html.Div(
                            children=[
                                html.H3("Triage Nurse Data",
                                        style={"background": color1, "color": title_color,
                                               'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                                html.Div([
                                    html.Iframe(
                                        style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px", "margin-top": "20px"})
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
                                        style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px", "margin-top": "20px"})
                                ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                                          "width": "100%", "height": "470px"}),
                            ],
                            className="panel",
                        )
                    )
    ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Feature Importances",
                        style={"background": color1, "color": title_color,
                               'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                    html.Div([
                        html.Iframe(
                            id="plot_directors",
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
                                html.H3("Feature Description",
                                        style={"background": color1, "color": title_color,
                                               'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                                html.Div([
                                    html.Iframe(
                                        style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px", "margin-top": "20px"})
                                ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                                          "width": "100%", "height": "500px"}),
                            ],
                            className="panel",
                        )
                    )
            ], style={"margin-top": "20px"})
            ],
            className="mt-4",
            )

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)