import dash
from dash import Dash, html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import base64

# Generate the plot and save it as an image
def generate_plot(option):
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, option, 2]))
    fig.update_layout(title="Plot", xaxis_title="X", yaxis_title="Y")
    image_filename = "plot.png"
    fig.write_image(image_filename)
    return image_filename

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Plot as Image with Dropdown"),
    dcc.Dropdown(
        id="dropdown",
        options=[
            {"label": "Option 1", "value": 1},
            {"label": "Option 2", "value": 2},
            {"label": "Option 3", "value": 3}
        ],
        value=2
    ),
    html.Div(id="plot-container")
])

# Define the callback to update the plot
@app.callback(Output("plot-container", "children"), [Input("dropdown", "value")])
def update_plot(option):
    image_filename = generate_plot(option)

    # Read the image as base64
    with open(image_filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Display the image
    return html.Img(src="data:image/png;base64,{}".format(encoded_image))

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8200)
