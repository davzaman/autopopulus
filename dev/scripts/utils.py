import asyncio
import itertools
import re
import subprocess
import sys
import time
from asyncio.subprocess import PIPE
from csv import DictWriter
from json import loads
from os import name as os_name
from os.path import basename, exists, join
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from autopopulus.utils.utils import rank_zero_print

EXPERIMENT_GRID_VARS = {
    "cure_ckd": [
        "method",
        "feature-map",
        # ampute dims
        "percent-missing",
        "mechanism",
        "score_to_probability_func",
    ],
    "crrt": ["method", "feature-map"],
}
EXPERIMENT_RUNNING_ARGS = [
    "method",
    "fully_observed",
    "dataset",
    "bootstrap_evaluate_imputer",
    "feature_map",
    "replace_nan_with",
    "percent_missing",
    "amputation_patterns",
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
        "original": [
            "simple",
            # "mice",
            "knn",
            "dvae",
            "vae",
            "dae",
            "batchswap",
            "vanilla",
        ],
        "order": [
            "Simple",
            # "MICE",
            "KNN",
            "DVAE",
            "VAE",
            "DAE",
            "Batchswap",
            "Vanilla",
        ],
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
            "Mixed Features",
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
    "metric_name": {"name": "Metric"},
    "reduction": {"name": "Metric Reduction"},
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


def save_fig_to_svg(fig, fname: str = "impute_performance.svg"):
    dims = (1480, 720)
    # fig.update_layout( autosize=False, width=dims[0], height=dims[1])
    # fig.write_image("impute_performance.svg",  width=dims[0], height=dims[1])
    fig.write_image(fname, width=2000)


def format_names(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "metric_name"] = df["metric_name"].str.replace("_", "&")
    df = df.rename(  # Format col names
        {k: v["name"] for k, v in PRETTY_NAMES.items()}, axis="columns"
    ).replace(  # Format values
        {
            v["name"]: dict(zip(v["original"], v["order"]))
            for v in PRETTY_NAMES.values()
            if "original" in v
        }
    )
    # deal with multiindex
    if isinstance(df.index, pd.MultiIndex):
        levels = []
        for level in df.index.levels:
            info = PRETTY_NAMES.get(level.name, {})
            mapper = dict(zip(info.get("original", {}), info.get("order", {})))
            levels.append(level.map(mapper if mapper else lambda x: x))
        df.index = df.index.set_levels(levels).set_names(
            [PRETTY_NAMES.get(n, {}).get("name", n) for n in df.index.names]
        )
    return df


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


def default_graph_format(fig, l1_name: str, l2_name: str):
    # Get rid of mechanism= and metric=
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(tickangle=45)
    compress_legend(fig, l1_name, l2_name)
    format_legend(fig)


def compress_legend(fig, l1_name: str, l2_name: str):
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
                # combine with traceorder: https://plotly.com/python/reference/#layout-legend-traceorder
                trace["legendgroup"] = "l1"
                trace["legendgrouptitle_text"] = l1_name
                # trace["legendwidth"] = 130
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
                        legendgroup="l2",
                        legendgrouptitle_text=l2_name,
                    ),
                )
                l2_categories.append(score_to_prob)
        except (
            ValueError
        ):  # custom added trace where the name doesn't have a , leave it alone
            pass


def format_legend(fig):
    fig.update_layout(
        legend=dict(
            font=dict(size=10),
            # yanchor="middle",
            # xanchor="right",
            # x=1.5,
            # y=0.5,
            # orientation="v",
            x=0.5,
            y=-0.2,
            orientation="h",
            yanchor="top",
            xanchor="center",
            traceorder="grouped",
            tracegroupgap=20,
        ),
        title_x=0.5,
        legend_title=dict(text="", font=dict(size=12)),
        font_family="Times New Roman",
    )


def add_percent_missing_to_legend(fig):
    for percent_missing, size in zip(["33%", "66%"], [5, 10]):
        fig.add_trace(
            go.Scatter(
                x=[np.nan],
                y=[np.nan],
                legendgroup="percent_missing",
                legendgrouptitle_text="Percent Missing",
                mode="markers",
                marker=dict(size=size, color="black"),
                name=percent_missing,
            ),
            # row=1, # col=1,
        )
    fig.update_layout(legend_itemsizing="trace")


def all_impute_but(dim: str) -> List[str]:
    all_dims = EXPERIMENT_GRID_VARS + IMPUTE_METRIC_DIMENSIONS
    all_dims.remove(dim)
    return all_dims


def all_predict_but(dim: str) -> List[str]:
    all_dims = EXPERIMENT_GRID_VARS + PREDICT_METRIC_DIMENSIONS
    all_dims.remove(dim)
    return all_dims


def cli_str(obj) -> str:
    # format for split() also same thign for CLI. the str repr of lists/etc has spaces which will create problems.
    return str(obj).replace(" ", "")


def listify_command_args(command_args: Dict[str, Any]):
    return {k: v if isinstance(v, list) else [v] for k, v in command_args.items()}


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


async def read_stream_and_display(stream, display) -> bytes:
    """
    Read from stream line by line until EOF, display, and capture the lines.
    Ref: https://stackoverflow.com/a/25960956/1888794
    """
    output = []
    while True:
        try:
            line = await stream.readline()
            if not line:
                break
            output.append(line)
            display(line)  # assume it doesn't block
        except (
            ValueError
        ) as e:  # ValueError: Separator is not found, and chunk exceed the limit
            msg_bytes = str(e).encode()
            output.append(msg_bytes)
            display(msg_bytes)
            continue
    return b"".join(output)


async def read_and_display(*cmd) -> Tuple[int, bytes, bytes]:
    """
    Capture cmd's stdout, stderr while displaying them as they arrive (line by line).
    Ref: https://stackoverflow.com/a/25960956/1888794
    """
    # start process, https://stackoverflow.com/a/55458913/1888794
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=PIPE, stderr=PIPE, limit=1024 * 256  # 256 KiB
    )

    # read child's stdout/stderr concurrently (capture and display)
    try:
        stdout, stderr = await asyncio.gather(
            read_stream_and_display(process.stdout, sys.stdout.buffer.write),
            read_stream_and_display(process.stderr, sys.stderr.buffer.write),
        )
    except Exception:
        process.kill()
        raise
    finally:
        # wait for the process to exit
        rc = await process.wait()
    return (rc, stdout, stderr)


class RunManager:
    def __init__(
        self,
        experiment_tracker: str,
        continue_runs: bool = True,
        guild_use_queues: int = 0,
    ) -> None:
        # run the event loop
        if os_name == "nt":
            self.loop = asyncio.ProactorEventLoop()  # for subprocess' pipes on Windows
            asyncio.set_event_loop(self.loop)
        else:
            self.loop = asyncio.get_event_loop()

        self.continue_runs = continue_runs
        self.experiment_tracker = experiment_tracker
        # if use_queues nonzero, will use queues, specify the number of queues (parralellism).
        self.guild_use_queues = guild_use_queues
        self.progress_file = "run_progress.csv"
        self.field_names = [
            "timestamp",
            "return_code",
            "command",
            "run_id",
        ] + EXPERIMENT_RUNNING_ARGS
        if self.continue_runs and exists(self.progress_file):
            return
        if experiment_tracker != "guild":
            with open(
                self.progress_file, "a" if self.continue_runs else "w", newline=""
            ) as csvfile:
                writer = DictWriter(csvfile, fieldnames=self.field_names)
                writer.writeheader()

    def __del__(self):
        self.close()

    def close(self):
        self.loop.close()

    def run_pipeline(self, command_args: Dict[str, Any]):
        if self.experiment_tracker == "guild":
            base = "guild run main "
            base += "--stage" if self.guild_use_queues else "--background"
            subprocess.run(
                base.split()
                + [f"{name}={cli_str(val)}" for name, val in command_args.items()]
            )
        else:
            for command_args in product_dict(**listify_command_args(command_args)):
                parent_hash = self.run("autopopulus/impute.py", command_args)
                sub_command_args = {**command_args, "parent-hash": parent_hash}
                self._kill_orphaned_ray_procs()
                self.run("autopopulus/evaluate.py", sub_command_args)
                self.run("autopopulus/predict.py", sub_command_args)

    def _default_to_existing_run(self, command: List[str]) -> str:
        if self.continue_runs:  # there should be no existing run with that command
            runs = pd.read_csv(self.progress_file)
            command_str = " ".join(command)
            existing_run = runs["command"] == command_str
            if any(existing_run):
                return runs[existing_run]["run_id"].squeeze()
            return None
        return None

    def _kill_orphaned_ray_procs(self):
        subprocess.run(
            "for pid in $(ps -ef | awk '($3 == 1 && $8 ~ /ray/){ print $2; }'); do kill -9 $pid; done",
            shell=True,
        )

    def run(self, pyfile: str, command_args: Dict[str, Any]) -> str:
        command = [sys.executable, pyfile]
        for name, val in command_args.items():
            command += [f"--{name}", cli_str(val)]

        if (run_id := self._default_to_existing_run(command)) is not None:
            rank_zero_print(f"Skipping {command} as it was already run...")
            return run_id

        rc, parent_hash = self.run_and_save_output(command, run_name=basename(pyfile))

        with open(self.progress_file, "a") as csvfile:
            writer = DictWriter(  # ignore parent-hash
                csvfile, fieldnames=self.field_names, extrasaction="ignore"
            )
            writer.writerow(
                {
                    "return_code": rc,
                    "command": " ".join(command),
                    "run_id": parent_hash,
                    "timestamp": time.strftime("%d-%m-%Y %H:%M:%S"),
                    **command_args,
                }
            )

        return parent_hash

    def run_and_save_output(self, cmd: List[str], run_name: str) -> Tuple[int, str]:
        rc, stdout, stderr = self.loop.run_until_complete(read_and_display(*cmd))
        stdout = stdout.decode("utf-8", errors="replace")
        stderr = stderr.decode("utf-8", errors="replace")
        parent_hash = re.search(r"(?:Logger Hash: )(\w+)\b", str(stdout))
        if parent_hash:
            parent_hash = parent_hash.groups()[-1]
            artifact_path = mlflow.get_run(
                run_id=parent_hash
            ).info.artifact_uri.replace("file://", "")
            with open(join(artifact_path, f"{run_name}.log"), "w") as f:
                f.write(stdout)
            with open(join(artifact_path, f"{run_name}.err"), "w") as f:
                f.write(stderr)
        return (rc, parent_hash)
