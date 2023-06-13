from json import loads
from typing import List
import numpy as np
import plotly.graph_objects as go
import pandas as pd

EXPERIMENT_GRID_VARS = [
    "method",
    "feature-map",
    # ampute dims
    "percent-missing",
    "mechanism",
    "score_to_probability_func",
]
IMPUTE_METRIC_DIMENSIONS = [
    "split",
    "feature_space",
    "filter_subgroup",
    "reduction",
    "feature_type",
    "metric_name",
]

PREDICT_METRIC_DIMENSIONS = ["predictor", "metric_name"]

PRETTY_NAMES = {
    "method": {
        "original": ["simple", "mice", "dvae", "vae", "dae", "batchswap", "vanilla"],
        "order": ["Simple", "MICE", "DVAE", "VAE", "DAE", "Batchswap", "Vanilla"],
        "baseline_order": ["None", "Simple", "MICE"],
        "ae_order": ["DVAE", "VAE", "DAE", "Batchswap", "Vanilla"],
        "name": "Method",
    },
    "feature-map": {
        "original": [
            "onehot_categorical",
            "target_encode_categorical",
            "discretize_continuous",
            None,
        ],
        "order": [
            "One Hot Categorical",
            "Target Encode Categorical",
            "Discretize Continuous",
            "None",
        ],
        "name": "Feature Mapping",
    },
    "replace-nan-with": {
        "original": [0, "simple"],
        "order": ["0", "Simple"],
        "name": "Replace NaN With",
    },
    # missing scenario
    "mechanism": {
        "order": ["MCAR", "MAR", "MNAR", "MNAR(G)", "MNAR(Y)"],
        "name": "Mechanism",
    },
    "percent-missing": {
        "original": [0.33, 0.66],
        "order": [33.0, 66.0],
        "name": "Percent Missing",
    },
    "score_to_probability_func": {
        "original": ["sigmoid-mid", "sigmoid-tail"],
        "order": ["Sigmoid (Mid)", "Sigmoid (Tail)"],
        "name": "Score to Probability Missing",
    },
    "metric": {"name": "Metric"},
    "dataset": {
        "original": ["cure_ckd", "crrt"],
        "order": ["CURE CKD", "CRRT"],
        "name": "Dataset",
    },
    "predictor": {
        "order": ["LGBM", "RF"],
        "original": ["lgbm", "rf"],
        "name": "Predictor",
    },
}


def save_fig_to_svg(fig):
    dims = (1480, 720)
    # fig.update_layout( autosize=False, width=dims[0], height=dims[1])
    # fig.write_image("impute_performance.svg",  width=dims[0], height=dims[1])
    fig.write_image("impute_performance.svg", width=2000)


def format_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(  # Format col names
        {k: v["name"] for k, v in PRETTY_NAMES.items()}, axis="columns"
    ).replace(  # Format values
        {
            v["name"]: dict(zip(v["original"], v["order"]))
            for v in PRETTY_NAMES.values()
            if "original" in v
        }
    )


# Filter for missing only
def impute_table_to_latex(table: pd.DataFrame, mask: pd.Series = None) -> str:
    if mask is not None:
        table = table[mask]
    table = table.pivot_table(
        index=[
            PRETTY_NAMES[x]["name"]
            for x in [
                "metric",
                "mechanism",
                "score_to_probability_func",
                "percent-missing",
            ]
        ],
        columns=[PRETTY_NAMES[x]["name"] for x in ["method", "feature-map"]],
        values="val",
        # dropna=False,
    )
    table = table[PRETTY_NAMES["method"]["order"]]
    # display(table)
    latex = table.to_latex(float_format="%.3f")
    #     print(latex)
    return latex


def compress_legend(fig):
    # high-level: color assignment column in scatterplot (e.g. feature_map)
    l1_categories = []
    # sub-level: symbol assignment column in scatterplot (e.g. c)
    l2_categories = []
    for trace in fig["data"]:
        try:
            feature_map, score_to_prob = trace["name"].split(",")

            if feature_map not in l1_categories:
                trace["name"] = feature_map
                trace["showlegend"] = True
                l1_categories.append(feature_map)
            else:
                trace["showlegend"] = False

            if score_to_prob not in l2_categories:
                marker = trace["marker"].to_plotly_json()
                marker["color"] = "black"
                fig.add_trace(
                    go.Scatter(
                        x=[np.nan],
                        y=[np.nan],
                        mode="markers",
                        marker=marker,
                        name=score_to_prob,
                    )
                )
                l2_categories.append(score_to_prob)
        except (
            ValueError
        ):  # custom added trace where the name doesn't have a , leave it alone
            pass


def all_impute_but(dim: str) -> List[str]:
    all_dims = EXPERIMENT_GRID_VARS + IMPUTE_METRIC_DIMENSIONS
    all_dims.remove(dim)
    return all_dims


def all_predict_but(dim: str) -> List[str]:
    all_dims = EXPERIMENT_GRID_VARS + PREDICT_METRIC_DIMENSIONS
    all_dims.remove(dim)
    return all_dims
