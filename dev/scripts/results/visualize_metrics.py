# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import transforms, ticker
from IPython.display import display

from utils import (
    EXPERIMENT_GRID_VARS,
    PREDICT_METRIC_DIMENSIONS,
    add_percent_missing_to_legend,
    all_predict_but,
    compress_legend,
    default_graph_format,
    format_names,
    PRETTY_NAMES,
    save_fig_to_svg,
)

tracker = "mlflow"
dataset = "crrt"
root = "/home/davina/Private/repos/autopopulus/guild_runs/"

########################
#    Impute Metrics    #
########################
# %%
impute_data = pd.read_pickle(join(root, f"{tracker}_{dataset}_impute_results.pkl"))

# %%
test_impute_data = impute_data[
    (impute_data["split"] == "test")
    & (impute_data["feature_space"] == "original")
    & (impute_data["filter_subgroup"] == "missingonly")
    & (impute_data["reduction"] == "CW")
    & (impute_data["feature_type"] == "mixed")
]
# the None feature-map from baselines should just be onehot cat
test_impute_data.loc[:, "feature-map"] = test_impute_data["feature-map"].fillna(
    "onehot_categorical"
)

agg_impute_data = (
    test_impute_data.groupby(
        EXPERIMENT_GRID_VARS[dataset] + ["reduction", "metric_name"], dropna=False
    )
    .mean()
    .reset_index()
)
agg_impute_data["combined_metric_name"] = (
    agg_impute_data["reduction"]
    + " "
    + agg_impute_data["metric_name"].str.replace("_", " ")
)


# %%
col = "mechanism"
row = "metric_name"
# https://github.com/plotly/dash-sample-apps/pull/516
# https://www.nature.com/articles/nmeth.1618
color_palette = [
    # "#E69F00",
    "#996900",  # adjust E69F00 a so greyscale is better (value=60%)
    "#56B4E9",  # value=91%
    # "#009E73",
    # "#F0E442",
    # "#0072B2"
    # "#D46027", # value=83%
    "#bf5722",  # adjust D46027, value=75
    "#CC7CA8",
]
fig = px.scatter(
    format_names(agg_impute_data),
    x=PRETTY_NAMES["method"]["name"],
    y="val",
    color=PRETTY_NAMES["feature-map"]["name"],
    facet_col=PRETTY_NAMES[col]["name"],
    # facet_col="combined_metric_name",
    # facet_col_spacing=0.08,  # when vert legend
    facet_col_spacing=0.04,
    # facet_row=PRETTY_NAMES["metric_name"]["name"],
    facet_row=PRETTY_NAMES[row]["name"],
    facet_row_spacing=0.05,
    symbol=PRETTY_NAMES["score_to_probability_func"]["name"],
    symbol_sequence=["circle", "x-open"],
    size=PRETTY_NAMES["percent-missing"]["name"],
    size_max=8,
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    labels={"val": "", "Method": ""},
    # title="Column-wise Imputation Performance per Missingness Mechanism",
    # width=2551/agg_impute_data[col].nunique()/2,
    # height=3295/agg_impute_data[row].nunique()*1.4,
    color_discrete_sequence=color_palette,
)
add_percent_missing_to_legend(fig)
default_graph_format(
    fig, l1_name="Feature Mapping", l2_name="Score to Probability Function"
)

fig

# %%
fig.write_image(join(root, f"{dataset}_impute_performance.pdf"), width=850)

#########################
#    Predict Metrics    #
#########################
# %%
predict_data = pd.read_pickle(join("root", f"{tracker}_{dataset}_predict_results.pkl"))
predict_data.set_index(EXPERIMENT_GRID_VARS[dataset])[
    ["val"] + PREDICT_METRIC_DIMENSIONS
]

predict_data.loc[:, "feature-map"] = predict_data["feature-map"].fillna(
    "onehot_categorical"
)

# %%
stats = (
    predict_data.groupby(
        EXPERIMENT_GRID_VARS[dataset] + PREDICT_METRIC_DIMENSIONS, dropna=False
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
    facet_col=PRETTY_NAMES[col]["name"] if dataset == "cure_ckd" else None,
    facet_col_spacing=0.06,
    facet_row=PRETTY_NAMES[row]["name"],
    facet_row_spacing=0.04,
    size=PRETTY_NAMES["percent-missing"]["name"] if dataset == "cure_ckd" else None,
    size_max=5,
    # title="Prediction Performance",
    # labels={"mean": "Mean", "Method": "Imputation Method"},
    labels={"mean": "", "Method": ""},
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    color_discrete_sequence=color_palette,
)
if dataset == "cure_ckd":
    add_percent_missing_to_legend(fig)
default_graph_format(fig, l1_name="Feature Mapping", l2_name="Predictor")
if dataset == "crrt":
    fig.update_layout(width=400, height=800, legend=dict(y=-0.1))

fig

# %%
if dataset == "crrt":
    fig.write_image(join(root, f"{dataset}_pred_performance.pdf"))
else:
    fig.write_image(join(root, f"{dataset}_pred_performance.pdf"), width=850)

# %%
time_data = pd.read_pickle(join(root, f"{tracker}_{dataset}_time_results.pkl"))
time_data = time_data.set_index(EXPERIMENT_GRID_VARS[dataset])[
    ["val", "step", "metric_name", "split"]
]

# %%
stats = (
    time_data.groupby(EXPERIMENT_GRID_VARS[dataset] + ["split"], dropna=False)["val"]
    .agg(["mean", "sem"])
    .assign(
        ci96_hi=lambda stats: stats["mean"] + 1.96 * stats["sem"],
        ci95_low=lambda stats: stats["mean"] - 1.96 * stats["sem"],
    )
)
display(stats)


# %%
flat_stats = stats.reset_index()
flat_stats["erry"] = flat_stats["ci96_hi"] - flat_stats["mean"]
flat_stats["erryminus"] = flat_stats["mean"] - flat_stats["ci95_low"]
fig = px.scatter(
    format_names(flat_stats),
    x=PRETTY_NAMES["method"]["name"],
    y="mean",
    error_y="erry",
    error_y_minus="erryminus",
    symbol=PRETTY_NAMES["score_to_probability_func"]["name"],
    color=PRETTY_NAMES["feature-map"]["name"],
    # https://plotly.com/python/marker-style/#custom-marker-symbols
    symbol_sequence=["circle", "x-open"],
    facet_col=PRETTY_NAMES["mechanism"]["name"],
    facet_col_spacing=0.06,
    # facet_row="metric_name",
    # facet_row_spacing=0.04,
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
fig.write_image(join(root, f"{dataset}_time_performance.pdf"))
