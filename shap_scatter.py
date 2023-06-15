import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import plotly.graph_objs as go
from plotly.subplots import make_subplots



def matching_cols(cols1, cols2):
    """returns True if cols1 and cols2 match."""
    if isinstance(cols1, pd.DataFrame):
        cols1 = cols1.columns
    if isinstance(cols2, pd.DataFrame):
        cols2 = cols2.columns
    if len(cols1) != len(cols2):
        return False
    if (pd.Index(cols1) == pd.Index(cols2)).all():
        return True
    return False


def plotly_shap_scatter_plot(
    X,
    shap_values_df,
    display_columns=None,
    title="Shap values",
    idxs=None,
    highlight_index=None,
    na_fill=-999,
    round=3,
    max_cat_colors=5,
):
    """Generate a shap values summary plot where features are ranked from
    highest mean absolute shap value to lowest, with point clouds shown
    for each feature.

    Args:

        X (pd.DataFrame): dataframe of input features
        shap_values_df (pd.DataFrame): dataframe shap_values with same columns as X
        display_columns (List[str]): list of feature to be displayed. If None
            default to all columns in X.
        title (str, optional): Title to display above graph.
            Defaults to "Shap values".
        idxs (List[str], optional): List of identifiers for each row in X.
            Defaults to None.
        highlight_index ({str, int}, optional): Index to highlight in graph.
            Defaults to None.
        na_fill (int, optional): Fill value used to fill missing values,
            will be colored grey in the graph.. Defaults to -999.
        round (int, optional): rounding to apply to floats. Defaults to 3.
        index_name (str): identifier for idxs. Defaults to "index".

    Returns:
        Plotly fig
    """
    assert matching_cols(
        X, shap_values_df
    ), "X and shap_values_df should have matching columns!"
    if display_columns is None:
        display_columns = X.columns.tolist()
    if idxs is not None:
        assert len(idxs) == X.shape[0]
        idxs = pd.Index(idxs).astype(str)
    else:
        idxs = X.index.astype(str)
    index_name = idxs.name

    length = len(X)
    if highlight_index is not None:
        if isinstance(highlight_index, int):
            assert highlight_index >= 0 and highlight_index < len(
                X
            ), "if highlight_index is int, then should be between 0 and {len(X)}!"
            highlight_idx = highlight_index
            highlight_index = idxs[highlight_idx]
        elif isinstance(highlight_index, str):
            assert str(highlight_index) in idxs, f"{highlight_index} not found in idxs!"
            highlight_idx = np.where(idxs == str(highlight_index))[0].item()
        else:
            raise ValueError("Please pass either int or str highlight_index!")

    # make sure that columns are actually in X:
    display_columns = [col for col in display_columns if col in X.columns]
    min_shap = shap_values_df.min().min()
    max_shap = shap_values_df.max().max()
    shap_range = max_shap - min_shap
    min_shap = min_shap - 0.01 * shap_range
    max_shap = max_shap + 0.01 * shap_range

    fig = make_subplots(
        rows=len(display_columns),
        cols=1,
        subplot_titles=display_columns,
        shared_xaxes=True,
    )

    for i, col in enumerate(display_columns):
        if is_numeric_dtype(X[col]):
            # numerical feature get a single bluered plot
            fig.add_trace(
                go.Scattergl(
                    x=shap_values_df[col],
                    y=np.random.rand(length),
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=X[col].replace({na_fill: np.nan}),
                        colorscale="Bluered",
                        showscale=True,
                        opacity=0.3,
                        colorbar=dict(
                            title="feature value <br> (red is high)",
                            tickfont=dict(color="rgba(0, 0, 0, 0)"),
                        ),
                    ),
                    name=col,
                    showlegend=False,
                    opacity=0.8,
                    hoverinfo="text",
                    text=[
                        f"{index_name}={i}<br>{col}={value}<br>shap={shap:.{round}f}"
                        for i, shap, value in zip(
                            idxs, shap_values_df[col], X[col].replace({na_fill: np.nan})
                        )
                    ],
                ),
                row=i + 1,
                col=1,
            )
        else:
            color_cats = list(X[col].value_counts().index[:max_cat_colors])
            n_color_cats = len(color_cats)
            colors = [
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ]
            colors = colors * (1 + int(n_color_cats / len(colors)))
            colors = colors[:n_color_cats]
            for cat, color in zip(color_cats, colors):
                fig.add_trace(
                    go.Scattergl(
                        x=shap_values_df[col][X[col] == cat],
                        y=np.random.rand((X[col] == cat).sum()),
                        mode="markers",
                        marker=dict(
                            size=5,
                            showscale=False,
                            opacity=0.3,
                            color=color,
                        ),
                        name=cat,
                        showlegend=False,
                        opacity=0.8,
                        hoverinfo="text",
                        text=[
                            f"{index_name}={i}<br>{col}={cat}<br>shap={shap:.{round}f}"
                            for i, shap in zip(
                                idxs[X[col] == cat], shap_values_df[col][X[col] == cat]
                            )
                        ],
                    ),
                    row=i + 1,
                    col=1,
                )
            if X[col].nunique() > max_cat_colors:
                fig.add_trace(
                    go.Scattergl(
                        x=shap_values_df[col][~X[col].isin(color_cats)],
                        y=np.random.rand((~X[col].isin(color_cats)).sum()),
                        mode="markers",
                        marker=dict(
                            size=5,
                            showscale=False,
                            opacity=0.3,
                            color="grey",
                        ),
                        name="Other",
                        showlegend=False,
                        opacity=0.8,
                        hoverinfo="text",
                        text=[
                            f"{index_name}={i}<br>{col}={col_val}<br>shap={shap:.{round}f}"
                            for i, shap, col_val in zip(
                                idxs[~X[col].isin(color_cats)],
                                shap_values_df[col][~X[col].isin(color_cats)],
                                X[col][~X[col].isin(color_cats)],
                            )
                        ],
                    ),
                    row=i + 1,
                    col=1,
                )

        if highlight_index is not None:
            fig.add_trace(
                go.Scattergl(
                    x=[shap_values_df[col].iloc[highlight_idx]],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        color="LightSkyBlue",
                        size=20,
                        opacity=0.5,
                        line=dict(color="MediumPurple", width=4),
                    ),
                    name=f"{index_name} {highlight_index}",
                    text=f"index={highlight_index}<br>{col}={X[col].iloc[highlight_idx]}<br>shap={shap_values_df[col].iloc[highlight_idx]:.{round}f}",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
        fig.update_xaxes(
            showgrid=False, zeroline=False, range=[min_shap, max_shap], row=i + 1, col=1
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=i + 1, col=1
        )

    fig.update_layout(
        title=title + "<br>",
        height=100 + len(display_columns) * 50,
        margin=go.layout.Margin(l=40, r=40, b=40, t=100, pad=4),
        hovermode="closest",
        plot_bgcolor="#fff",
    )
    return fig
