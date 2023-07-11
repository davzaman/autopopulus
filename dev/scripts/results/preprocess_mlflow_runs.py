# %%
# I only need this if i'm running this as interactive cells, instead of a script
from os import makedirs
from os.path import join
import sys


root = "/home/davina/Private/repos/autopopulus"

sys.path.insert(0, join(sys.path[0], root))

# %%
import mlflow
import pandas as pd
import re
from typing import Dict, Any


from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    PREDICT_METRIC_TAG_FORMAT,
)
from autopopulus.utils.cli_arg_utils import string_json_to_python


# %%
def all_scalars(run_id: str) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            dict(metric)
            for metric_name in metrics
            for metric in client.get_metric_history(run_id, metric_name)
        ],
    ).rename({"key": "metric", "value": "val"}, axis=1)
    df["run_id"] = run_id
    return df


def expand_amputation_pattern(all_info: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            list(
                all_info["amputation-patterns"]
                .apply(string_json_to_python)
                .apply(lambda l: l[0] if l is not None else l)
                .dropna()
            ),
            index=all_info[all_info["amputation-patterns"].notna()].index,
        )
        .assign(num_incomplete_vars=lambda df: df["incomplete_vars"].apply(len))
        .assign(
            num_observed_vars=lambda df: df["weights"].apply(
                lambda w: len(w) if isinstance(w, dict) else 0
            )
        )
    )


def expand_col(expand_rule: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    # add expanded col to the original df (filtered)
    return pd.concat(
        [
            df[expand_rule["filter"]],
            df[expand_rule["filter"]][expand_rule["column"]]
            .str.split("/", expand=True)
            .set_axis(expand_rule["expanded_columns"], axis=1),
        ],
        axis=1,
    )


# %%
tracking_uri = "/home/davina/Private/repos/autopopulus/mlruns"
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

# %%
mlflow.set_tracking_uri(tracking_uri)
dataset = "cure_ckd"
experiment_groups = {"crrt": ["crrt"], "cure_ckd": ["baseline", "ae", "ae-ec2"]}
all_runs = mlflow.search_runs(experiment_names=experiment_groups[dataset])
# runs = mlflow.search_runs(search_all_experiments=True)

# %%
is_metric_col = all_runs.columns.str.startswith("metrics")
metric_cols = all_runs.columns[is_metric_col]
metrics = metric_cols.str.replace("metrics.", "")

# %%
runs = all_runs[all_runs["status"] == "FINISHED"]

# %%
# Duplicate runs
dup_dims = [
    "tags.method",
    "tags.dataset",
    "tags.fully_observed",
    "tags.replace_nan_with",
    "tags.feature_map",
    "tags.percent_missing",
    "tags.amputation_patterns",
    "tags.bootstrap_evaluate_imputer",
]
dup_runs = runs[runs.duplicated(subset=dup_dims, keep=False)][["run_id"] + dup_dims]
dup_runs
# %%
run_scalars = pd.concat([all_scalars(run_id) for run_id in runs["run_id"]])

# %%
all_info = (
    runs.loc[:, ~is_metric_col]
    .merge(run_scalars, how="left", on="run_id")
    .drop(
        # tags.feature_map is what we want, not params.feature_map
        ["tags.feature_map"],
        axis=1,
    )
)


# %%
# Rename cols to match guild
all_info.columns = (
    all_info.columns.map(
        # underscores should be dashes to match guild
        lambda name: name.replace("_", "-")
        if "params" in name or "tags" in name
        else name
    )
    .str.replace("params.", "")
    .str.replace("tags.", "")
)

# %%
if dataset == "cure_ckd":
    all_info = pd.concat(
        [all_info, expand_amputation_pattern(all_info)], axis=1
    ).astype({"percent-missing": float})

# %%
all_info[all_info["metric"].isna()]


# %%
# drop messed up runs
all_info = all_info[all_info["metric"].notna()]

# %%
impute_data = expand_col(
    {  # imputation metrics
        "column": "metric",
        "filter": (  # exclude random and other sys metrics
            ~all_info["metric"].str.contains(
                "sys|hp_metric|epoch|step|restore|done|time|checkpoint|lgbm|rf"
            )
        ),
        # temp hack to replace name with metric_name to avoid col name collision
        "expanded_columns": re.sub("{|}", "", IMPUTE_METRIC_TAG_FORMAT)
        .replace("name", "metric_name")
        .split("/"),
    },
    all_info,
)

# %%
impute_data

# %%
output_dir = "guild_runs"
makedirs(output_dir, exist_ok=True)
impute_data.to_pickle(join(root, output_dir, f"mlflow_{dataset}_impute_results.pkl"))

# %%
predict_data = expand_col(
    {  # predict metrics
        "column": "metric",
        "filter": all_info["metric"].str.contains("lgbm|rf"),
        "expanded_columns": re.sub("{|}", "", PREDICT_METRIC_TAG_FORMAT)
        .replace("name", "metric_name")
        .split("/"),
    },
    all_info,
)

# %%
predict_data.to_pickle(join(root, output_dir, f"mlflow_{dataset}_predict_results.pkl"))

# %%
time_data = expand_col(
    {
        "column": "metric",
        "filter": all_info["metric"].str.contains("duration"),
        "expanded_columns": ["split", "metric_name"],
    },
    all_info,
)

# %%
time_data.to_pickle(join(root, output_dir, f"mlflow_{dataset}_time_results.pkl"))

# %%
