import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.graph_objs as go


def plotly_dependence_plot(
    X_col,
    shap_values,
    interact_col=None,
    interaction=False,
    na_fill=-999,
    round=3,
    units="",
    highlight_index=None,
    idxs=None,
):
    """Returns a dependence plot showing the relationship between feature col_name
    and shap values for col_name. Do higher values of col_name increase prediction
    or decrease them? Or some kind of U-shape or other?

    Args:
        X_col (pd.Series): pd.Series with column values.
        shap_values (np.ndarray): shap values generated for X_col
        interact_col (pd.Series): pd.Series with column marker values. Defaults to None.
        interaction (bool, optional): Is this a plot of shap interaction values?
            Defaults to False.
        na_fill (int, optional): value used for filling missing values.
            Defaults to -999.
        round (int, optional): Rounding to apply to floats. Defaults to 2.
        units (str, optional): Units of the target variable. Defaults to "".
        highlight_index (str, int, optional): index row of X to highlight in
            the plot. Defaults to None.
        idxs (pd.Index, optional): list of descriptors of the index, e.g.
            names or other identifiers. Defaults to None.

    Returns:
        Plotly fig
    """
    assert len(X_col) == len(
        shap_values
    ), f"Column(len={len(X_col)}) and Shap values(len={len(shap_values)}) and should have the same length!"
    if idxs is not None:
        assert len(idxs) == X_col.shape[0]
        idxs = pd.Index(idxs).astype(str)
    else:
        idxs = X_col.index.astype(str)

    if highlight_index is not None:
        if isinstance(highlight_index, int):
            highlight_idx = highlight_index
            highlight_name = idxs[highlight_idx]
        elif isinstance(highlight_index, str):
            assert (
                highlight_index in idxs
            ), f"highlight_index should be int or in idxs, {highlight_index} is neither!"
            highlight_idx = idxs.get_loc(highlight_index)
            highlight_name = highlight_index

    col_name = X_col.name

    if interact_col is not None:
        text = np.array(
            [
                f"{idxs.name}={index}<br>{X_col.name}={col_val}<br>{interact_col.name}={col_col_val}<br>SHAP={shap_val:.{round}f}"
                for index, col_val, col_col_val, shap_val in zip(
                    idxs, X_col, interact_col, shap_values
                )
            ]
        )
    else:
        text = np.array(
            [
                f"{idxs.name}={index}<br>{X_col.name}={col_val}<br>SHAP={shap_val:.{round}f}"
                for index, col_val, shap_val in zip(idxs, X_col, shap_values)
            ]
        )

    data = []

    X_col = X_col.copy().replace({na_fill: np.nan})
    y = shap_values
    if interact_col is not None and not is_numeric_dtype(interact_col):
        for onehot_col in interact_col.unique().tolist():
            data.append(
                go.Scattergl(
                    x=X_col[interact_col == onehot_col].replace({na_fill: np.nan}),
                    y=shap_values[interact_col == onehot_col],
                    mode="markers",
                    marker=dict(size=7, showscale=False, opacity=0.6),
                    showlegend=True,
                    opacity=0.8,
                    hoverinfo="text",
                    name=onehot_col,
                    text=[
                        f"{idxs.name}={index}<br>{X_col.name}={col_val}<br>{interact_col.name}={interact_val}<br>SHAP={shap_val:.{round}f}"
                        for index, col_val, interact_val, shap_val in zip(
                            idxs,
                            X_col[interact_col == onehot_col],
                            interact_col[interact_col == onehot_col],
                            shap_values[interact_col == onehot_col],
                        )
                    ],
                )
            )
    elif interact_col is not None and is_numeric_dtype(interact_col):
        if na_fill in interact_col:
            data.append(
                go.Scattergl(
                    x=X_col[interact_col != na_fill],
                    y=shap_values[interact_col != na_fill],
                    mode="markers",
                    text=text[interact_col != na_fill],
                    hoverinfo="text",
                    marker=dict(
                        size=7,
                        opacity=0.6,
                        color=interact_col[interact_col != na_fill],
                        colorscale="Bluered",
                        colorbar=dict(title=interact_col.name),
                        showscale=True,
                    ),
                )
            )
            data.append(
                go.Scattergl(
                    x=X_col[interact_col == na_fill],
                    y=shap_values[interact_col == na_fill],
                    mode="markers",
                    text=text[interact_col == na_fill],
                    hoverinfo="text",
                    marker=dict(size=7, opacity=0.35, color="grey"),
                )
            )
        else:
            data.append(
                go.Scattergl(
                    x=X_col,
                    y=shap_values,
                    mode="markers",
                    text=text,
                    hoverinfo="text",
                    marker=dict(
                        size=7,
                        opacity=0.6,
                        color=interact_col,
                        colorscale="Bluered",
                        colorbar=dict(title=interact_col.name),
                        showscale=True,
                    ),
                )
            )

    else:
        data.append(
            go.Scattergl(
                x=X_col,
                y=shap_values,
                mode="markers",
                text=text,
                hoverinfo="text",
                marker=dict(size=7, opacity=0.6),
            )
        )

    if interaction:
        title = f"Interaction plot for {X_col.name} and {interact_col.name}"
    else:
        title = f"Dependence plot for {X_col.name}"

    layout = go.Layout(
        title=title,
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(title=col_name),
        yaxis=dict(title=f"SHAP value ({units})" if units != "" else "SHAP value"),
    )

    fig = go.Figure(data, layout)

    if interact_col is not None and not is_numeric_dtype(interact_col):
        fig.update_layout(showlegend=True)

    if highlight_index is not None:
        fig.add_trace(
            go.Scattergl(
                x=[X_col.iloc[highlight_idx]],
                y=[shap_values[highlight_idx]],
                mode="markers",
                marker=dict(
                    color="LightSkyBlue",
                    size=25,
                    opacity=0.5,
                    line=dict(color="MediumPurple", width=4),
                ),
                name=f"{idxs.name} {highlight_name}",
                text=f"{idxs.name} {highlight_name}",
                hoverinfo="text",
                showlegend=False,
            ),
        )
    fig.update_traces(selector=dict(mode="markers"))
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    return fig
