import re
from os import makedirs
from os.path import dirname, join
from typing import Any, Dict

import guild.ipy as guild
import pandas as pd
from guild import timerange

from autopopulus.utils.cli_arg_utils import string_json_to_python
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    PREDICT_METRIC_TAG_FORMAT,
)


def filter_by_time(query: str, df: pd.DataFrame) -> pd.DataFrame:
    # API: https://my.guild.ai/t/command-runs-delete/72#filter-by-run-start-time-8
    # Code pulled from: https://github.com/guildai/guildai/blob/dba6d124a8e796ae4b802a5d5a35d95805b0ff4c/guild/commands/runs_impl.py#L373
    start, end = timerange.parse_spec(query)
    if start is None:
        start = pd.NaT
    if end is None:
        end = pd.NaT
    # started_day = all_guild_runs["started"].apply(lambda d: d.date())
    mask = ~(
        (bool(start) & (df["started"] < start)) | (bool(end) & (df["started"] >= end))
    )
    return df[mask]


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


def get_guild_dfs() -> Dict[str, pd.DataFrame]:
    # config.set_guild_home(GUILD_HOME_PATH)
    all_guild_runs = guild.runs()
    all_flags = all_guild_runs.guild_flags()
    all_scalars = all_guild_runs.scalars_detail()
    return {"runs": all_guild_runs, "flags": all_flags, "scalars": all_scalars}


def combine_all_guild_dfs(guild_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_run_meta_data = pd.concat(
        [
            # Run object instead of runindex
            guild_data["runs"]["run"].apply(lambda runindex: runindex.run),
            # Keep op
            guild_data["runs"]["operation"],
            # Expand flags
            pd.DataFrame(
                guild_data["runs"]["run"]
                .apply(lambda run: run.run.get("flags"))
                .tolist(),
                index=guild_data["runs"].index,
            ),
        ],
        axis=1,
    )

    # get all scalars, ignoring duplicates from scalars from both main op and substep op
    # combine with run metadata
    all_info = (
        guild_data["scalars"]
        .merge(  # get operation name for each scalar
            all_run_meta_data,
            how="left",
            on="run",
        )
        .query("operation != 'main'")  # drop main run as it duplicates everything
        .rename({"tag": "metric", "path": "model"}, axis="columns")
        # logs copied over from ray/tune will have the prefix I don't want.
        .assign(metric=lambda df: df["metric"].str.replace("ray/tune/", ""))
        .assign(
            model=lambda df: df["model"].str.replace(
                "/lightning_logs/version_.*", "", regex=True
            )
        )
    )

    return pd.concat(
        [
            expand_col(  # split model into time dim and imputer name
                {
                    "column": "model",
                    "filter": [True] * len(all_info),  # all
                    "expanded_columns": ["data_type_time_dim", "imputer_model"],
                },
                all_info,
            ),
            # amputation_patterns
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
            ),
        ],
        axis=1,
    )


if __name__ == "__main__":
    guild_data = get_guild_dfs()
    expanded_info = combine_all_guild_dfs(guild_data)
    predict_data = expand_col(
        {  # predict metrics
            "column": "metric",
            "filter": expanded_info["operation"] == "predict",
            "expanded_columns": re.sub("{|}", "", PREDICT_METRIC_TAG_FORMAT)
            .replace("name", "metric_name")
            .split("/"),
        },
        expanded_info,
    )
    impute_data = expand_col(
        {  # imputation metrics
            "column": "metric",
            "filter": (expanded_info["operation"].isin(["impute", "evaluate"]))
            & (  # exclude random and other sys metrics
                ~expanded_info["metric"].str.contains(
                    "sys|hp_metric|epoch|step|restore|done|time|checkpoint"
                )
            ),
            # TODO[LOW]: change {name} -> {metric_name} everywhere
            # temp hack to replace name with metric_name to avoid col name collision
            "expanded_columns": re.sub("{|}", "", IMPUTE_METRIC_TAG_FORMAT)
            .replace("name", "metric_name")
            .split("/"),
        },
        expanded_info,
    )

    time_data = expand_col(
        {
            "column": "metric",
            "filter": expanded_info["operation"].isin(["impute", "evaluate"])
            & expanded_info["metric"].str.contains("duration"),
            "expanded_columns": ["split", "metric_name"],
        },
        expanded_info,
    )

    output_dir = "guild_runs"
    makedirs(output_dir, exist_ok=True)
    impute_data.to_pickle(join(output_dir, "guild_impute_results.pkl"))
    predict_data.to_pickle(join(output_dir, "guild_predict_results.pkl"))
    time_data.to_pickle(join(output_dir, "guild_time_results.pkl"))
