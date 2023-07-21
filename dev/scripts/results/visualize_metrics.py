# %%
# %load_ext autoreload
# %autoreload 2
import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], "/home/davina/Private/repos/autopopulus"))

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import transforms, ticker
from IPython.display import display

from dev.scripts.utils import (
    EXPERIMENT_GRID_VARS,
    IMPUTE_METRIC_DIMENSIONS,
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
dataset = "cure_ckd"
legend = False
root = "/home/davina/Private/repos/autopopulus/guild_runs/"

########################
#    Impute Metrics    #
########################
# %%
impute_data = pd.read_pickle(join(root, f"{tracker}_{dataset}_impute_results.pkl"))

# %%
# the None feature-map from baselines should just be onehot cat
impute_data.loc[:, "feature-map"] = impute_data["feature-map"].fillna(
    "onehot_categorical"
)
agg_impute_data = (
    impute_data.groupby(
        EXPERIMENT_GRID_VARS[dataset] + IMPUTE_METRIC_DIMENSIONS, dropna=False
    )
    .first()
    .reset_index()
)
agg_impute_data["combined_metric_name"] = (
    agg_impute_data["reduction"]
    + " "
    + agg_impute_data["metric_name"].str.replace("_", "&")
)


# %%
test_impute_data = agg_impute_data[
    (agg_impute_data["split"] == "test")
    & (agg_impute_data["feature_space"] == "original")
    & (agg_impute_data["filter_subgroup"] == "missingonly")
    & (agg_impute_data["reduction"] == "CW")
    & (agg_impute_data["feature_type"] == "mixed")
]
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
    format_names(test_impute_data),
    x=PRETTY_NAMES["method"]["name"],
    y="val",
    color=PRETTY_NAMES["feature-map"]["name"],
    facet_col=PRETTY_NAMES[col]["name"],
    # facet_col="combined_metric_name",
    # facet_col_spacing=0.08,  # when vert legend
    facet_col_spacing=0.04,
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
    height=400,
    color_discrete_sequence=color_palette,
)
add_percent_missing_to_legend(fig)
default_graph_format(
    fig, l1_name="Feature Mapping", l2_name="Score to Probability Function"
)
if not legend:
    fig.update_layout(showlegend=False)


fig

# %%
fig.write_image(join(root, f"{dataset}_impute_performance.pdf"), width=850)


# %%
#######################
#  EW vs CW  Metrics  #
#######################
reduction_row = "reduction"
ew_vs_cw = agg_impute_data[
    (agg_impute_data["split"] == "test")
    & (agg_impute_data["feature_space"] == "original")
    & (agg_impute_data["filter_subgroup"] == "missingonly")
    & (agg_impute_data["metric_name"].str.contains("MAAPE"))
    & (agg_impute_data["feature_type"] == "mixed")
]
fig = px.scatter(
    format_names(ew_vs_cw),
    x=PRETTY_NAMES["method"]["name"],
    y="val",
    color=PRETTY_NAMES["feature-map"]["name"],
    facet_col=PRETTY_NAMES[col]["name"],
    facet_col_spacing=0.04,
    facet_row=PRETTY_NAMES[reduction_row]["name"],
    facet_row_spacing=0.05,
    symbol=PRETTY_NAMES["score_to_probability_func"]["name"],
    symbol_sequence=["circle", "x-open"],
    size=PRETTY_NAMES["percent-missing"]["name"],
    size_max=8,
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    labels={"val": "", "Method": ""},
    color_discrete_sequence=color_palette,
    height=300,
    width=850,
)
add_percent_missing_to_legend(fig)
default_graph_format(
    fig, l1_name="Feature Mapping", l2_name="Score to Probability Function"
)
if not legend:
    fig.update_layout(showlegend=False)

fig

# %%
fig.write_image(join(root, f"{dataset}_impute_reduction_performance.pdf"))
ew_vs_cw.set_index(EXPERIMENT_GRID_VARS[dataset] + ["reduction"])["val"]


# %%
#######################
#   Mapped  Metrics   #
#######################
mapped_row = "feature_space"
mapped = agg_impute_data[
    (agg_impute_data["split"] == "test")
    # & (agg_impute_data["feature_space"] == "mapped")
    & (agg_impute_data["filter_subgroup"] == "missingonly")
    & (agg_impute_data["metric_name"].str.contains("MAAPE"))
    & (agg_impute_data["reduction"] == "CW")
    & (agg_impute_data["feature_type"] == "mixed")
]
# mapped = impute_data[
#     (impute_data["split"] == "test")
#     & (impute_data["filter_subgroup"] == "missingonly")
#     & (impute_data["reduction"] == "CW")
#     & (impute_data["feature_type"] == "mixed")
# ]
fig = px.scatter(
    format_names(mapped),
    x=PRETTY_NAMES["method"]["name"],
    y="val",
    color=PRETTY_NAMES["feature-map"]["name"],
    facet_col=PRETTY_NAMES[col]["name"],
    facet_col_spacing=0.04,
    # facet_row=PRETTY_NAMES[mapped_row]["name"],
    facet_row=mapped_row,
    facet_row_spacing=0.05,
    symbol=PRETTY_NAMES["score_to_probability_func"]["name"],
    symbol_sequence=["circle", "x-open"],
    size=PRETTY_NAMES["percent-missing"]["name"],
    size_max=8,
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    labels={"val": "", "Method": ""},
    color_discrete_sequence=color_palette,
    height=300,
    width=850,
)
add_percent_missing_to_legend(fig)
default_graph_format(
    fig, l1_name="Feature Mapping", l2_name="Score to Probability Function"
)
if not legend:
    fig.update_layout(showlegend=False)

fig

# %%
fig.write_image(join(root, f"{dataset}_impute_mapped_performance.pdf"))

# %%
###########################
#    Impute CI Metrics    #
###########################
grouped = impute_data[
    (impute_data["split"] == "test")
    & (impute_data["feature_space"] == "original")
    & (impute_data["filter_subgroup"] == "missingonly")
    & (impute_data["reduction"] == "CW")
    & (impute_data["feature_type"] == "mixed")
].groupby(EXPERIMENT_GRID_VARS[dataset] + IMPUTE_METRIC_DIMENSIONS, dropna=False)
impute_ci_stats = (
    grouped["val"]
    .agg(["mean", "sem"])
    .assign(
        ci96_hi=lambda stats: stats["mean"] + 1.96 * stats["sem"],
        ci95_low=lambda stats: stats["mean"] - 1.96 * stats["sem"],
    )
)
# limit to those with 100 steps (0 -> 99)
impute_ci_stats = impute_ci_stats[grouped["step"].max() == 99]
display(impute_ci_stats)
flat_stats = impute_ci_stats.reset_index()
flat_stats["erry"] = flat_stats["ci96_hi"] - flat_stats["mean"]
flat_stats["erryminus"] = flat_stats["mean"] - flat_stats["ci95_low"]
fig = px.scatter(
    format_names(flat_stats),
    x=PRETTY_NAMES["method"]["name"],
    y="mean",
    error_y="erry",
    error_y_minus="erryminus",
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
if not legend:
    fig.update_layout(showlegend=False)
fig

# %%
stats = []
mechanisms = flat_stats["mechanism"].unique()
metrics = flat_stats["metric_name"].unique()
fig = make_subplots(
    rows=len(metrics),
    cols=len(mechanisms),
    subplot_titles=mechanisms,
    shared_xaxes=True,
)
for i, mech in enumerate(mechanisms):
    ae_stats_for_mech = flat_stats[
        (flat_stats["mechanism"] == mech)
        & ~flat_stats["method"].str.contains("knn|simple")
    ].iloc[0]
    stats_for_mech = flat_stats[
        (flat_stats["mechanism"] == mech)
        & (flat_stats["percent-missing"] == ae_stats_for_mech["percent-missing"])
        & (
            flat_stats["score_to_probability_func"]
            == ae_stats_for_mech["score_to_probability_func"]
        )
    ]

    for j, metric in enumerate(metrics):
        mech_metric_stats = stats_for_mech[stats_for_mech["metric_name"] == metric]
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=mech_metric_stats["method"],
                y=mech_metric_stats["mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=mech_metric_stats["erry"],
                    arrayminus=mech_metric_stats["erryminus"],
                    visible=True,
                ),
                showlegend=False,
            ),
            row=j + 1,
            col=i + 1,
        )
    stats.append(stats_for_mech)
EXPERIMENT_GRID_VARS_TABLE_ORDER = (
    [  # order matters so can't use experiment_grid global arg
        "mechanism",
        "percent-missing",
        "score_to_probability_func",
        "method",
        "feature-map",
    ]
)
as_table = pd.concat(stats).set_index(EXPERIMENT_GRID_VARS_TABLE_ORDER)

formatted_ci_table = (
    format_names(as_table)
    .pivot(
        columns=PRETTY_NAMES["metric_name"]["name"],
    )
    .sort_index(
        level=[PRETTY_NAMES["mechanism"]["name"], PRETTY_NAMES["method"]["name"]],
        key=lambda column: column.map(
            lambda e: PRETTY_NAMES[
                "mechanism"
                if column.name == PRETTY_NAMES["mechanism"]["name"]
                else "method"
            ]["order"].index(e)
        ),
        inplace=False,
    )
)  # .style.format("{:.3f}", subset=["mean", "erry"])
display(formatted_ci_table)
fig

# %%
print(
    (
        formatted_ci_table["mean"].applymap("{:.4f}".format).astype("str")
        + " +/- "
        + formatted_ci_table["erry"].applymap("{:.4f}".format).astype("str")
    ).to_latex()
)

# %%
fig.write_image(join(root, f"{dataset}_impute_ci_performance.pdf"), width=850)

# %%
##############
#    Loss    #
##############
last_loss_data = (
    impute_data.groupby(
        EXPERIMENT_GRID_VARS[dataset] + IMPUTE_METRIC_DIMENSIONS, dropna=False
    )
    .last()
    .reset_index()
)
loss_data = last_loss_data[
    (last_loss_data["split"] == "val")
    & (last_loss_data["metric_name"].str.contains("loss"))
]
fig = px.line_3d(
    impute_data[
        (impute_data["split"] == "val")
        & (impute_data["metric_name"].str.contains("loss"))
    ],
    x="method",
    y="step",
    z="val",
    color="feature-map",
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    labels={"val": "", "Method": ""},
    # title="Column-wise Imputation Performance per Missingness Mechanism",
    # width=2551/agg_impute_data[col].nunique()/2,
    # height=3295/agg_impute_data[row].nunique()*1.4,
    color_discrete_sequence=color_palette,
)
display(fig)
fig = px.scatter(
    format_names(loss_data),
    x="step",
    y="val",
    color=PRETTY_NAMES["feature-map"]["name"],
    facet_col=PRETTY_NAMES[col]["name"],
    # facet_col="combined_metric_name",
    # facet_col_spacing=0.08,  # when vert legend
    facet_col_spacing=0.04,
    facet_row=PRETTY_NAMES["method"]["name"],
    # facet_row=PRETTY_NAMES[row]["name"],
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
    height=250,
    color_discrete_sequence=color_palette,
)
display(fig)
fig = px.scatter(
    format_names(loss_data),
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
    height=250,
    color_discrete_sequence=color_palette,
)
add_percent_missing_to_legend(fig)
default_graph_format(
    fig, l1_name="Feature Mapping", l2_name="Score to Probability Function"
)
if not legend:
    fig.update_layout(showlegend=False)
display(fig)

# %%
fig.write_image(join(root, f"{dataset}_loss_performance.pdf"), width=850)

#########################
#    Predict Metrics    #
#########################
# %%
predict_data = pd.read_pickle(join(root, f"{tracker}_{dataset}_predict_results.pkl"))
predict_data.set_index(EXPERIMENT_GRID_VARS[dataset])[
    ["val"] + PREDICT_METRIC_DIMENSIONS
]

predict_data.loc[:, "feature-map"] = predict_data["feature-map"].fillna(
    "onehot_categorical"
)

# %%
pred_stats = (
    predict_data.groupby(
        EXPERIMENT_GRID_VARS[dataset] + PREDICT_METRIC_DIMENSIONS, dropna=False
    )["val"]
    .agg(["mean", "sem"])
    .assign(
        ci96_hi=lambda stats: stats["mean"] + 1.96 * stats["sem"],
        ci95_low=lambda stats: stats["mean"] - 1.96 * stats["sem"],
    )
)
display(pred_stats)

# %%
limit_table_metrics = ["Brier-score"]
table_metrics = format_names(pred_stats)[
    pred_stats.index.get_level_values("metric_name").str.contains(
        "|".join(limit_table_metrics)
    )
].assign(erry=lambda df: df["ci96_hi"] - df["mean"])
display(table_metrics)
(
    table_metrics["mean"].map("{:.3f}".format).astype("str")
    + " +/- "
    + table_metrics["erry"].map("{:.3f}".format).astype("str")
).to_latex()

# %%
shape = "long"
if shape == "long":
    pred_col = "metric_name"
    pred_row = "mechanism"
elif shape == "wide":
    pred_col = "mechanism"
    pred_row = "metric_name"
limit_viz_metrics = ["ROC-AUC"]
flat_stats = pred_stats.reset_index()
flat_stats = flat_stats[flat_stats["metric_name"].isin(limit_viz_metrics)]
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
    symbol_sequence=["cross", "diamond-open"],
    facet_col=PRETTY_NAMES[pred_col]["name"],  # if dataset == "cure_ckd" else None,
    facet_col_spacing=0.02,
    facet_row=PRETTY_NAMES[pred_row]["name"] if dataset == "cure_ckd" else None,
    facet_row_spacing=0.01,
    size=PRETTY_NAMES["percent-missing"]["name"] if dataset == "cure_ckd" else None,
    size_max=8,
    # title="Prediction Performance",
    # labels={"mean": "Mean", "Method": "Imputation Method"},
    labels={"mean": "", "Method": ""},
    category_orders={
        info["name"]: info["order"] for info in PRETTY_NAMES.values() if "order" in info
    },
    color_discrete_sequence=color_palette,
    # long format
    width=350 if shape == "long" else 850,
    height=850 if shape == "long" else 350,
)
add_percent_missing_to_legend(fig)
default_graph_format(fig, l1_name="Feature Mapping", l2_name="Predictor")
if dataset == "crrt":
    fig.update_layout(width=350, height=350, legend=dict(y=-0.1))
if not legend:
    fig.update_layout(showlegend=False)
fig.update_traces(error_y_width=10, selector=dict(type="scatter"))
fig.update_traces(error_y_thickness=0.7, selector=dict(type="scatter"))

fig

# %%
if dataset == "crrt":
    fig.write_image(join(root, f"{dataset}_pred_performance.pdf"))
else:
    # fig.write_image(join(root, f"{dataset}_pred_performance.pdf"), width=850)
    fig.write_image(join(root, f"{dataset}_pred_performance.pdf"))

# %%
if dataset == "cure_ckd":
    semi_obs = pd.read_pickle(
        join(root, f"{tracker}_{dataset}_semi_obs_predict_results.pkl")
    )
    semi_obs.set_index(EXPERIMENT_GRID_VARS[dataset])[
        ["val"] + PREDICT_METRIC_DIMENSIONS
    ]

    semi_obs.loc[:, "feature-map"] = semi_obs["feature-map"].fillna(
        "onehot_categorical"
    )

    semi_obs_stats = (
        semi_obs.groupby(
            EXPERIMENT_GRID_VARS[dataset] + PREDICT_METRIC_DIMENSIONS, dropna=False
        )["val"]
        .agg(["mean", "sem"])
        .assign(
            ci96_hi=lambda stats: stats["mean"] + 1.96 * stats["sem"],
            ci95_low=lambda stats: stats["mean"] - 1.96 * stats["sem"],
        )
    )
    display(semi_obs_stats)

    limit_viz_metrics = ["ROC-AUC"]
    flat_stats = semi_obs_stats.reset_index()
    flat_stats = flat_stats[flat_stats["metric_name"].isin(limit_viz_metrics)]
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
        size_max=8,
        # title="Prediction Performance",
        # labels={"mean": "Mean", "Method": "Imputation Method"},
        labels={"mean": "", "Method": ""},
        category_orders={
            info["name"]: info["order"]
            for info in PRETTY_NAMES.values()
            if "order" in info
        },
        color_discrete_sequence=color_palette,
    )
    add_percent_missing_to_legend(fig)
    default_graph_format(fig, l1_name="Feature Mapping", l2_name="Predictor")
    if dataset == "crrt":
        fig.update_layout(width=400, height=800, legend=dict(y=-0.1))
    if not legend:
        fig.update_layout(showlegend=False)
    display(fig)
    # fig.write_image(join(root, f"{dataset}_pred_performance.pdf"), width=850)

    formatted_semi_obs_table = (
        format_names(
            semi_obs_stats.reset_index().set_index(
                EXPERIMENT_GRID_VARS_TABLE_ORDER + ["predictor"]
            )
        )
        .pivot(
            columns=PRETTY_NAMES["metric_name"]["name"],
        )
        .sort_index(
            level=[PRETTY_NAMES["mechanism"]["name"], PRETTY_NAMES["method"]["name"]],
            key=lambda column: column.map(
                lambda e: PRETTY_NAMES[
                    "mechanism"
                    if column.name == PRETTY_NAMES["mechanism"]["name"]
                    else "method"
                ]["order"].index(e)
            ),
            inplace=False,
        )
    )
    display(formatted_semi_obs_table)
    fig

    limit_semi_obs_metrics = ["ROC-AUC"]
    print(
        (
            formatted_semi_obs_table["mean"][limit_semi_obs_metrics]
            .applymap("{:.4f}".format)
            .astype("str")
            + " +/- "
            + (
                formatted_semi_obs_table["ci96_hi"][limit_semi_obs_metrics]
                - formatted_semi_obs_table["mean"][limit_semi_obs_metrics]
            )
            .applymap("{:.4f}".format)
            .astype("str")
        ).to_latex()
    )

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
    legend=legend,
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
