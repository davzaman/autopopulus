# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import transforms, ticker
from IPython.display import display

from utils import (
    EXPERIMENT_GRID_VARS,
    PREDICT_METRIC_DIMENSIONS,
    all_predict_but,
    compress_legend,
    format_names,
    PRETTY_NAMES,
)

########################
#    Impute Metrics    #
########################
# %%
impute_data = pd.read_pickle(
    "/home/davina/Private/repos/autopopulus/guild_impute_results.pkl"
)

# %%
test_impute_data = impute_data[
    (impute_data["split"] == "test")
    & (impute_data["feature_space"] == "original")
    & (impute_data["filter_subgroup"] == "missingonly")
    & (impute_data["reduction"] == "CW")
    & (impute_data["feature_type"] == "mixed")
]


# %%
fig = px.scatter(
    format_names(test_impute_data),
    x=PRETTY_NAMES["method"]["name"],
    y="val",
    color=PRETTY_NAMES["feature-map"]["name"],
    facet_col=PRETTY_NAMES["mechanism"]["name"],
    facet_col_spacing=0.06,
    # facet_row=PRETTY_NAMES["metric"]["name"],
    facet_row="metric_name",
    facet_row_spacing=0.3,
    symbol=PRETTY_NAMES["score_to_probability_func"]["name"],
    symbol_sequence=["circle", "x-open"],
    size=PRETTY_NAMES["percent-missing"]["name"],
    size_max=5,
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    labels={"val": "Metric", "Method": "Imputation Method"},
    # title="Imputation Performance",
)
fig.add_trace(
    go.Scatter(
        x=[np.nan],
        y=[np.nan],
        legendgroup="A",
        mode="markers",
        marker=dict(size=5, color="black"),
        name="33% Missing",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[np.nan],
        y=[np.nan],
        legendgroup="A",
        mode="markers",
        marker=dict(size=10, color="black"),
        name="66% Missing",
    ),
    row=1,
    col=1,
)
fig.update_layout(legend_itemsizing="trace")

# Get rid of mechanism= and metric=
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_layout(title_x=0.5, title=dict(text="Imputation Performance", font=dict(size=10), yref='container'))
fig.update_layout(
    legend=dict(
        font=dict(size=10),
        yanchor="top",
        y=-0.5,
        xanchor="center",
        x=0.5,
        orientation="h",
    ),
    legend_title=dict(text="", font=dict(size=12)),
)
fig.update_yaxes(matches=None, showticklabels=True)
fig.update_xaxes(tickangle=45)
compress_legend(fig)
fig

#########################
#    Predict Metrics    #
#########################
# %%
predict_data = pd.read_pickle(
    "/home/davina/Private/repos/autopopulus/guild_predict_results.pkl"
)
predict_data.set_index(EXPERIMENT_GRID_VARS)[["val"] + PREDICT_METRIC_DIMENSIONS]


# %%
stats = (
    predict_data.groupby(
        EXPERIMENT_GRID_VARS + PREDICT_METRIC_DIMENSIONS, dropna=False
    )["val"]
    .agg(["mean", "sem"])
    .assign(
        ci96_hi=lambda stats: stats["mean"] + 1.96 * stats["sem"],
        ci95_low=lambda stats: stats["mean"] - 1.96 * stats["sem"],
    )
)
display(stats)

# %%
limit_metrics = ["Brier-score", "PR-AUC", "ROC-AUC"]
flat_stats = stats.reset_index()
flat_stats = flat_stats[flat_stats["metric_name"].isin(limit_metrics)]
flat_stats["erry"] = flat_stats["ci96_hi"] - flat_stats["mean"]
flat_stats["erryminus"] = flat_stats["mean"] - flat_stats["ci95_low"]
fig = px.scatter(
    format_names(flat_stats),
    x=PRETTY_NAMES["method"]["name"],
    y="mean",
    error_y="erry",
    error_y_minus="erryminus",
    symbol=PRETTY_NAMES["predictor"]["name"],
    color=PRETTY_NAMES["feature-map"]["name"],
    # https://plotly.com/python/marker-style/#custom-marker-symbols
    symbol_sequence=["circle", "x-open"],
    facet_col=PRETTY_NAMES["mechanism"]["name"],
    facet_col_spacing=0.06,
    facet_row="metric_name",
    facet_row_spacing=0.04,
    size=PRETTY_NAMES["percent-missing"]["name"],
    size_max=5,
    # title="Prediction Performance",
    # x_title="Imputation Method",
    labels={"mean": "Mean", "Method": "Imputation Method"},
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
)
# Get rid of metric=
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None, showticklabels=True)
fig.update_xaxes(tickangle=45)

fig.add_trace(
    go.Scatter(
        x=[np.nan],
        y=[np.nan],
        legendgroup="A",
        mode="markers",
        marker=dict(size=5, color="black"),
        name="33% Missing",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[np.nan],
        y=[np.nan],
        legendgroup="A",
        mode="markers",
        marker=dict(size=10, color="black"),
        name="66% Missing",
    ),
    row=1,
    col=1,
)
fig.update_layout(legend_itemsizing="trace")
# fig.update_layout(legend=dict(font=dict(size=10)), legend_title=dict(text="", font=dict(size=12)))
compress_legend(fig)
fig.update_layout(
    title_x=0.5,
    legend_title=dict(text=""),
    legend=dict(yanchor="top", y=-0.5, xanchor="center", x=0.5, orientation="h"),
)
fig

# %%
# %%
