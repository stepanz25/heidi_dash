# Import required libraries
import numpy as np
import pandas as pd
import joblib
import shap
import random
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
import io
import plotly.io as pio
from dash.dependencies import Input, Output
from dash import Dash, html, dcc


def generate_bar_chart(value1, value2, height=230, width=1250):
    """
    Generate a bar chart with two bars representing the given values.

    Args:
        value1 (float): The first value to be represented as a bar.
        value2 (float): The second value to be represented as a bar.
        height (int, optional): The desired height of the plot in pixels. Default is 230.
        width (int, optional): The desired width of the plot in pixels. Default is 1250.

    Returns:
        go.Figure: The generated bar chart as a Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[''],
        x=[value1],
        orientation='h',
        marker=dict(
            color='rgba(255, 0, 0, 0.6)',  # Set the color to red
            line=dict(color='rgba(255, 0, 0, 1.0)', width=3)
        ),
        text=[f'<span style="color: black;">{value1}%</span>'],  # Add the percentage value as text with black color
        textposition='auto',
        textfont=dict(
            size=20,
            color='white',
        ),
        showlegend=False  # Hide the legend for this trace
    ))

    fig.add_trace(go.Bar(
        y=[''],
        x=[value2],
        orientation='h',
        marker=dict(
            color='rgba(0, 100, 0, 0.6)',  # Set the color to dark green
            line=dict(color='rgba(0, 100, 0, 1.0)', width=3)
        ),
        text=[f'<span style="color: black;">{value2}%</span>'],  # Add the percentage value as text with black color
        textposition='auto',
        textfont=dict(
            size=20,
            color='white',
        ),
        showlegend=False  # Hide the legend for this trace
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,  # Set the height of the plot
        width=width,  # Set the width of the plot
        title={
            'text': "Soft Probability Distribution",
            'x': 0.5,
            'y': 0.9,
            'font': {
                'size': 24,
                'color': '#234075',
                'family': 'Arial, sans-serif'
            }}
    )

    return fig


def generate_horizontal_bar_chart(dataframe):
    """
    Generate a horizontal bar chart based on SHAP values in a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the SHAP values and feature names.

    Returns:
        plotly.graph_objs._figure.Figure: The Plotly figure object representing the horizontal bar chart.
    """
    # Sort the dataframe by "SHAP values" in descending order
    sorted_df = dataframe.sort_values(by="SHAP Values", ascending=True)

    # Create the horizontal bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_df["SHAP Values"],
        y=sorted_df["Feature Name"],
        orientation='h',
        marker=dict(color="#234075")
    ))

    # Set the layout options
    fig.update_layout(
        xaxis_title="SHAP Values",
        yaxis_title="Feature Name",
        height=500,
        margin=dict(l=100, r=20, t=70, b=20),
        plot_bgcolor='white',  # Set the background color to white
        paper_bgcolor='white',  # Set the plot's paper background color to white
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Show gridlines on the x-axis
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Show gridlines on the y-axis
    )

    # Show the plot
    return fig


def b64_image(image_filename):
    """
    Convert an image file to base64 encoding.

    Args:
        image_filename (str): The filename of the image file.

    Returns:
        str: The base64 encoded string representing the image.
    """
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


# Load the test dataset
X_test = pd.read_csv("data/x_test.csv")
X_test_enc = pd.read_csv("data/x_test_enc.csv")
feature_data = pd.read_csv("data/feature_data.csv")

# Load the trained model
pipe_rf = joblib.load("src/models/rf.joblib")

# Define color codes
positive_color = "#00FF00"
negative_color = "#FF0000"
transparent = "#00000000"  # for transparent backgrounds
color1 = "#234075"  # blue
color2 = "#234075"  # border colors
plot_text_color = "#234075"  # plot axis and label color
title_color = "#e3a82b"  # general title and text color
border_radius = "5px"  # rounded corner radius
border_width = "3px"  # border width

# Define external stylesheets for the Dash app
my_external_stylesheets = [dbc.themes.YETI, 'src/assets/theme.css']

# Create a Dash app instance
app = Dash(__name__, external_stylesheets=my_external_stylesheets,
           title="HEiDi Classifier")

# Get the underlying Flask server for deployment
server = app.server

# Define layout
app.layout = dbc.Container([

    # Header section
    dbc.Row([
        dbc.Col([html.Img(src=b64_image("src/assets/logo1.jpg"), height="150px")], style={"width": "25%"}),
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
                    dbc.PopoverHeader(html.H4("Welcome to Classifier Explainer!", style={"font-size": "25px"})),
                    dbc.PopoverBody([
                        html.P("This dashboard contains:"),
                        html.P("🏥 Patient Information: View the details of a randomly selected patient."),
                        html.P(
                            "🏥 Prediction Results: See the soft predictions for the selected patient, indicating the probabilities for each class."),
                        html.P(
                            "🏥 Feature Table: Explore the features and their corresponding values for the selected patient."),
                        html.P(
                            "🏥 SHAP Plot: Visualize the SHAP values to understand the impact of each feature on the model's prediction."),
                        html.P(
                            "🏥 Prediction Label: Get a recommendation based on the model's prediction for the selected patient."),
                    ]),
                    dbc.PopoverHeader("You can do the following with this dashboard:"),
                    dbc.PopoverBody([
                        html.P(
                            "🏥 Generate Random Patient: Click the 'Generate' button to select a new patient randomly."),
                        html.P(
                            "🏥 Explore Patient Information: Review the patient's features and values in the patient information section."),
                        html.P(
                            "🏥 Analyze Prediction Results: Examine the soft predictions table to understand the probabilities for each class."),
                        html.P(
                            "🏥 Understand Feature Importance: Study the feature table and SHAP plot to identify the important features and their impact on the prediction."),
                        html.P(
                            "🏥 Get Prediction Recommendation: Read the prediction label to determine the model's recommendation for the patient."),
                    ])
                ],
                target="popover-target",
                trigger="legacy",
                placement="bottom"
            )
        ], style={'textAlign': 'right'})
    ]),

    # Prediction on Individual Patient section
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
                          "width": "100%", "height": "150px", "margin-bottom": "20px"}),

                html.Div([
                    html.Iframe(
                        id="soft_prob_bar",
                        style={
                            "border-width": "1",
                            "width": "100%",
                            "height": "550px",
                            "top": "100%",
                            "left": "0%",
                            "margin-top": "0px"  # Adjust the margin-top value to create space
                        },
                    ),
                ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                          "height": "200px"}),

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

    # Patient Information and Feature Description section
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

    # Feature Importances section
    dbc.Row([
        dbc.Col(
            html.Div(
                children=[
                    html.H3("SHAP Force Plot",
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
    dbc.Row([
        dbc.Col(
            html.Div(
                children=[
                    html.H3("Feature Importances",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%",
                                   "margin-top": "20px"}),
                    html.Div([
                        html.Iframe(
                            id="shap-table",
                            style={
                                "border-width": "1",
                                "width": "100%",
                                "height": "540px",
                                "top": "20%",
                                "left": "70%",
                                "margin-top": "0px"
                            },
                        ),
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "height": "550px"})
                ],
                className="panel",
            )
        ),

        dbc.Col(
            html.Div(
                children=[
                    html.H3("Top 10 Feature Importances",
                            style={"background": color1, "color": title_color,
                                   'textAlign': 'center', 'border-radius': border_radius, "width": "100%",
                                   "margin-top": "20px"}),
                    html.Div([
                        html.Iframe(
                            id="top_shap_bar",
                            style={
                                "border-width": "1",
                                "width": "100%",
                                "height": "550px",
                                "top": "20%",
                                "left": "70%",
                                "margin-top": "0px"
                            },
                        ),
                    ], style={"border": f"{border_width} solid {color2}", 'border-radius': border_radius,
                              "height": "550px"})
                ],
                className="panel",
            )
        )]),

    # Generate Data for New Patient button
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
              Output("shap-table", "srcDoc"),
              Output("top_shap_bar", "srcDoc"),
              Output("soft_prob_bar", "srcDoc"),
              Output("prediction-label", "children"),
              Input("generate-button", "n_clicks"))
def update_patient(n_clicks):
    """
    Callback function to update the patient information, prediction results, feature table, SHAP plot, and prediction label.

    Parameters:
    - n_clicks (int): The number of times the "generate" button is clicked.

    Returns:
    - Tuple of strings: Updated patient table HTML, prediction HTML, feature table HTML, SHAP plot HTML, and prediction label.

    """

    if n_clicks is None:
        # Return empty strings if generate button is not clicked
        return "", "", "", "", "", "", "", ""
    else:
        # Generate a random number between 0 and 20 to select a patient from the test set
        num = random.randint(0, 20)

        # Get the actual features and values of the selected patient
        choosen_actual = X_test.iloc[[num]].T.reset_index()
        choosen_actual.columns = ['Feature Name', 'Value']
        choosen_actual.index = choosen_actual.index + 1

        # Style the actual patient table
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

        # Get the encoded features of the selected patient
        choosen_instance = X_test_enc.loc[[num]]

        # Predict the probability of each class using the random forest classifier
        probability = pipe_rf.named_steps['randomforestclassifier'].predict_proba(choosen_instance)
        probability_red = probability[0][0] * 100
        probability_green = probability[0][1] * 100

        # Define the threshold for prediction labels
        X = 50.0

        if probability_red >= X:
            # Generate the prediction label for high probability of red class
            prediction_label = html.Div(
                [
                    "Physician is >= {:.2f}% likely to classify this individual as ".format(probability_red),
                    html.Span("RED", style={"color": "red", "text-transform": "uppercase", "font-weight": "bold"}),
                    " - requiring immediate emergency attention."
                ],
                style={"font-size": "larger"}
            )
        elif probability_green >= X:
            # Generate the prediction label for high probability of green class
            prediction_label = html.Div(
                [
                    "Physician is >= {:.2f}% likely to classify this individual as ".format(probability_green),
                    html.Span("GREEN", style={"color": "green", "text-transform": "uppercase", "font-weight": "bold"}),
                    " - being safe to manage in the community."
                ],
                style={"font-size": "larger"}
            )
        else:
            # Generate the prediction label when no recommendation is provided
            prediction_label = html.Div(
                "The model is unable to predict with >=50% confidence, therefore no recommendation has been provided.",
                style={"font-size": "larger"}
            )

        # Generate the soft predictions table
        out = pd.DataFrame(probability, columns=['md_red', 'md_green'])
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

        # Format and style the feature table
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

        # Generate the SHAP plot using the TreeExplainer
        rf_explainer = shap.TreeExplainer(pipe_rf.named_steps['randomforestclassifier'])
        shap_values = rf_explainer.shap_values(choosen_instance)
        force_plot = shap.force_plot(rf_explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=False,
                                     plot_cmap=[positive_color, negative_color])
        shap_html = f"<head>{shap.getjs()}</head><body><div style='color: #234075; font-family: Helvetica; font-size: 14px;'>{force_plot.html()}</div></body>"

        shap_value_df = pd.DataFrame(data=np.abs(shap_values[1].T), columns=["SHAP Values"])
        shap_value_df['Feature Name'] = X_test_enc.columns.values
        shap_value_df = shap_value_df[["Feature Name", "SHAP Values"]].sort_values(by="SHAP Values", ascending=False)
        top_shap_value_df = shap_value_df[:10]
        shap_value_df = shap_value_df.style \
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
            .set_caption(f'\nSHAP Values for Patient with Index {num}\n')

        top_shap_bar = generate_horizontal_bar_chart(top_shap_value_df)

        soft_prob_bar = generate_bar_chart(probability_red, probability_green)

        return styled_df.to_html(
            escape=False), out.to_html(), feature_table.to_html(), shap_html, shap_value_df.to_html(), top_shap_bar.to_html(), soft_prob_bar.to_html(), prediction_label


# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8045)