import pandas as pd
import numpy as np
import joblib

import shap
import eli5

import plotly.express as px
import plotly.graph_objs as go
import altair as alt

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash, html, dcc, dash_table
import dash_daq as daq

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.compose import make_column_transformer

# Load data and perform some data pre-processing

adult_df_large = pd.read_csv("../data/adult.csv")
train_df, test_df = train_test_split(adult_df_large, test_size=0.2, random_state=42)
train_df_nan = train_df.replace("?", np.NaN)
test_df_nan = test_df.replace("?", np.NaN)

X_train = train_df_nan.drop(columns=["income"])
y_train = train_df_nan["income"]

X_test = test_df_nan.drop(columns=["income"])
y_test = test_df_nan["income"]

label_encoder = LabelEncoder()
y_train_num = label_encoder.fit_transform(y_train)
y_test_num = label_encoder.transform(y_test)

# Defining thr theme

transparent = "#00000000"        # for transparent backgrounds
color1 = "#234075"               # blue
color2 = "#234075"               # border colors
plot_text_color = "#234075"      # plot axis and label color
title_color = "#e3a82b"          # general title and text color
border_radius = "5px"            # rounded corner radius
border_width = "3px"             # border width

# Custom plot functions

# Plotting permutation importances

def get_permutation_importance(model, title):
    X_train_perm = X_train.drop(columns=["race", "education.num", "fnlwgt"])
    result = permutation_importance(model, X_train_perm, y_train_num, n_repeats=10, random_state=123)
    perm_sorted_idx = result.importances_mean.argsort()

    df = pd.DataFrame({
        'importance': result.importances[perm_sorted_idx].tolist(),
        'feature': X_train_perm.columns[perm_sorted_idx].tolist()
    })
    df_exploded = df.explode('importance').reset_index(drop=True)

    chart = alt.Chart(df_exploded).mark_boxplot(extent=2.5, color="#234075").encode(
        y=alt.Y('feature:N', sort=alt.SortField(field='importance', order='descending'), title=title),
        x=alt.X('importance:Q', title='Importance Score')
    ).properties(width=400, height=390).configure(background=transparent
    ).configure_axis(
        labelColor=plot_text_color,
        titleColor=plot_text_color
    ).interactive()
    return chart.to_html()

# Create app

external_stylesheets = [dbc.themes.YETI, '/assets/theme.css']

app = Dash(__name__, external_stylesheets=external_stylesheets,
           title="HEiDi Classifier")

server = app.server

# Load the models

pipe_xgb = joblib.load("/Users/stepan_zaiatc/xgboost.joblib")

# Define layout
app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col([html.Img(src="/assets/logo1.jpg", height="150px")], style={"width": "25%"}),
        
        dbc.Col([html.H1('Classifier Explainer: Predicting HEiDi Triage', className='text-center',
                         style={'color': '#234075', 'textAlign': 'center', "font-weight": "bold", "fontSize": 40, "margin-top": "80px"})], md=4, style={"color": "#234075", "width": "70%"}),
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
                    html.H3("Triage Nurse Data",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                    html.Div([
                        html.Iframe(
                            style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px", "margin-top": "20px"}
                        )
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
                            style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px", "margin-top": "20px"}
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
                    html.H3("Feature Importances",
                            style={"background": color1, "color": title_color,
                                    'textAlign': 'center', 'border-radius': border_radius, "width": "100%"}),
                    html.Div([
                        html.Iframe(
                            id="feature_permutation",
                            style={'border': '0', 'width': '100%', 'height': '500px', "margin-left": "30px", "margin-top": "20px"},
                            srcDoc=get_permutation_importance(model=pipe_xgb, title="Permuted Features")
                        )
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "width": "100%", "height": "500px"})
                ],
                className="panel",
            )
        )
    ], style={"margin-top": "20px"})
],
className="mt-4")
fdf
@app.callback(Output("feature_permutation", "srcDoc"))
def update_output(model):
    feature_permutation = get_permutation_importance(model=pipe_xgb)
    return feature_permutation

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)