import numpy as np
import pandas as pd
import plotly.graph_objs as go

def plotly_importances_plot(
    importance_df,
    descriptions=None,
    round=3,
    target="target",
    units="",
    title=None,
    xaxis_title=None,
):
    """Return feature importance plot

    Args:
        importance_df (pd.DataFrame): DataFrame with columns "feature" and "weight"
        descriptions (dict, optional): dict of descriptions of each feature.
        round (int, optional): Rounding to apply to floats. Defaults to 3.
        target (str, optional): Name of target variable. Defaults to "target".
        units (str, optional): Units of target variable. Defaults to "".
        title (str, optional): Title for graph. Defaults to None.
        xaxis_title (str, optional): Title for x-axis Defaults to None.

    Returns:
        Plotly fig
    """

    if title is None:
        title = "Feature Importance"

    longest_feature_name = importance_df["feature"].str.len().max()

    imp = importance_df.sort_values("weight")

    feature_names = [
        str(len(imp) - i) + ". " + col
        for i, col in enumerate(imp["feature"].astype(str).values.tolist())
    ]

    importance_values = imp["weight"]

    data = [
        go.Bar(
            y=feature_names,
            x=importance_values,
            #text=importance_values.round(round),
            text=descriptions[::-1]
            if descriptions is not None
            else None,  # don't know why, but order needs to be reversed
            # textposition='inside',
            # insidetextanchor='end',
            hoverinfo="text",
            orientation="h",
        )
    ]

    layout = go.Layout(title=title, plot_bgcolor="#fff", showlegend=False, xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True, showgrid=True, zeroline=True)
    fig.update_xaxes(automargin=True, title=xaxis_title, showgrid=True, zeroline=True)

    left_margin = longest_feature_name * 7
    if np.isnan(left_margin):
        left_margin = 100

    fig.update_layout(
        height=200 + len(importance_df) * 20,
        margin=go.layout.Margin(l=left_margin, r=40, b=40, t=40, pad=4),
    )
    return fig