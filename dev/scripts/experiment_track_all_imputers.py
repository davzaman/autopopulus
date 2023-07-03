from csv import DictWriter
from os import name as os_name
from os.path import join, basename, exists
import time
from typing import Any, Dict, List, Tuple
import json
import re
import sys
import itertools
import subprocess
import asyncio
from asyncio.subprocess import PIPE

import mlflow
import pandas as pd

from autopopulus.utils.utils import rank_zero_print

# Enforce you're in the right env
output = subprocess.run(["conda", "list"], stdout=subprocess.PIPE).stdout.decode(
    "utf-8"
)
env_name = re.search(r"(?:/mambaforge/envs/)(\w+)\b", str(output)).groups()[-1]
assert (
    env_name == "ap"
), "You're not in the right conda environment. Did you forget to `mamba activate ap`?"

#############
#  Options  #
#############
"""
These describe the experiment grid.
All other (non-changing across experiments) params will be in the options.yml file.
"""
percent_missing = [0.33, 0.66]
feature_mapping = [
    "onehot_categorical",
    "target_encode_categorical",
    "discretize_continuous",
]
feature_mapping_variational = ["target_encode_categorical"]

# maps imputer method to list of imputer names under that group
imputer_groups = {
    "none": ["none"],
    # "baseline_imputers" :["simple","mice", "knn"],
    "baseline": ["simple", "knn"],
    "ae": ["vanilla", "dae", "batchswap"],
    "variational": ["vae", "dvae"],
}
# replace_nan_with = ["simple", "0"]
replace_nan_with = ["0"]

####################################
#  Select experiments to run here  #
####################################
# experiment switches: all experiments: none, baseline, ae, vae
# can use a mix of group names and also individual ones
chosen_methods = ["batchswap"]
experiment_tracker = "mlflow"
datasets = ["cure_ckd"]
# if use_queues nonzero, will use queues, specify the number of queues (parralellism).
guild_use_queues: int = 1
data_filtering = {
    "cure_ckd": {"all_data": False, "fully_observed": True},
    "crrt": {"all_data": True, "fully_observed": False},
}

# expand to individual names
chosen_methods = [
    method for name in chosen_methods for method in imputer_groups.get(name, [name])
]


async def read_stream_and_display(stream, display) -> bytes:
    """
    Read from stream line by line until EOF, display, and capture the lines.
    Ref: https://stackoverflow.com/a/25960956/1888794
    """
    output = []
    while True:
        line = await stream.readline()
        if not line:
            break
        output.append(line)
        display(line)  # assume it doesn't block
    return b"".join(output)


async def read_and_display(*cmd) -> Tuple[int, bytes, bytes]:
    """
    Capture cmd's stdout, stderr while displaying them as they arrive (line by line).
    Ref: https://stackoverflow.com/a/25960956/1888794
    """
    # start process
    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)

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
    def __init__(self, experiment_tracker: str, continue_runs: bool = True) -> None:
        # run the event loop
        if os_name == "nt":
            self.loop = asyncio.ProactorEventLoop()  # for subprocess' pipes on Windows
            asyncio.set_event_loop(self.loop)
        else:
            self.loop = asyncio.get_event_loop()

        self.continue_runs = continue_runs
        self.experiment_tracker = experiment_tracker
        self.progress_file = "run_progress.csv"
        self.field_names = [
            "timestamp",
            "return_code",
            "command",
            "run_id",
            "method",
            "dataset",
            "fully-observed",
            "replace-nan-with",
            "feature-map",
            "percent-missing",
            "amputation-patterns",
            "bootstrap-evaluate-imputer",
        ]
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

    def run_command(self, command_args: Dict[str, Any]):
        if self.experiment_tracker == "guild":
            base = "guild run main "
            base += "--stage" if guild_use_queues else "--background"
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
            "for pid in $(ps -ef | awk '($3 == 1 && $8 ~ /ray/){ print $2; }'); do kill -9 $pid; done".split()
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
        stdout, stderr = stdout.decode("utf-8"), stderr.decode("utf-8")
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


def cli_str(obj) -> str:
    # format for split() also same thign for CLI. the str repr of lists/etc has spaces which will create problems.
    return str(obj).replace(" ", "")


def listify_command_args(command_args: Dict[str, Any]):
    return {k: v if isinstance(v, list) else [v] for k, v in command_args.items()}


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def run_all():
    run_manager = RunManager(experiment_tracker=experiment_tracker)
    for dataset in datasets:
        # fully_observed=no uses entire dataset
        all_data = data_filtering[dataset]["all_data"]
        # fully_observed=yes will ampute and impute a missingness scenario
        fully_observed = data_filtering[dataset]["fully_observed"]
        # fully_observed = False

        if fully_observed:
            with open(f"dev/{dataset}_amputation_pattern_grid.txt", "r") as f:
                amputation_patterns = json.load(f)

        # https://my.guild.ai/t/command-run/146
        # https://my.guild.ai/t/running-cases-in-parallel/341
        for method in chosen_methods:
            print("======================================================")
            print(f"Staging {method} imputation...")
            if method == "none":
                run_manager.run_command(
                    {"method": "none", "fully-observed": "yes", "dataset": dataset}
                )
            else:
                command_args = {
                    "method": method,
                    "dataset": dataset,
                }
                # bootstrap evaluate the baseline models no matter what
                # manually pick the AE models to bootstrap eval later
                if method in imputer_groups["baseline"]:
                    command_args["bootstrap-evaluate-imputer"] = True
                # Added to the end if the conditions are met otherwise nothing happens
                if method in imputer_groups["ae"]:
                    command_args = {
                        **command_args,
                        "feature-map": feature_mapping,
                        "replace-nan-with": replace_nan_with,
                    }
                elif (
                    method in imputer_groups["variational"]
                ):  # on vae and dvae only try target_encode_categorical
                    command_args = {
                        **command_args,
                        "feature-map": feature_mapping_variational,
                        "replace-nan-with": replace_nan_with,
                    }
                # if command_args.get("feature-map", "") == "discretize_continuous":
                # command_args["uniform-prob"] = True

                if all_data:
                    # When multiple flags have list values, Guild generates the cartesian product of all possible flag combinations.
                    all_data_command_args = {**command_args, "fully-observed": "no"}
                    run_manager.run_command(all_data_command_args)
                if fully_observed:
                    fully_observed_command_args = {
                        **command_args,
                        "fully-observed": "yes",
                        "percent-missing": percent_missing,
                        "amputation-patterns": amputation_patterns,
                    }
                    run_manager.run_command(fully_observed_command_args)
    run_manager.close()

    if experiment_tracker == "guild" and guild_use_queues:
        print("======================================================")
        print("Starting Queues...")
        for _ in range(guild_use_queues):
            # https://stackoverflow.com/a/70072233/1888794
            subprocess.run("guild run queue run-once=yes -y", shell=True)
        # subprocess.run('guild view -h 127.0.0.1')


def rerun(command_args_list: List[Dict[str, Any]]):
    run_manager = RunManager(experiment_tracker=experiment_tracker)
    for command_args in command_args_list:
        run_manager.run_command(command_args)
    run_manager.close()


if __name__ == "__main__":
    # run_all()
    from pickle import load

    with open("retry_runs.pkl", "rb") as file:
        command_args_list = load(file)
    rerun(command_args_list)
