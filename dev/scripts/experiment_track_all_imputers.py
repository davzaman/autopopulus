from typing import Dict
import json
import re
import sys
import subprocess

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
    "none": "none",
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
# chosen_methods = ["none", "baseline", "ae", "vae"]
chosen_methods = ["baseline", "ae", "variational"]
experiment_tracker = "guild"
datasets = ["cure_ckd"]
# if use_queues nonzero, will use queues, specify the number of queues (parralellism).
guild_use_queues: int = 1
# fully_observed=no uses entire dataset
all_data = False
# fully_observed=yes will ampute and impute a missingness scenario
fully_observed = True
# fully_observed = False


# expand to individual names
chosen_methods = [
    method for name in chosen_methods for method in imputer_groups.get(name, [name])
]


def cli_str(obj) -> str:
    # format for split() also same thign for CLI. the str repr of lists/etc has spaces which will create problems.
    return str(obj).replace(" ", "")


def run_command(command_args: Dict[str, str]):
    if experiment_tracker == "guild":
        base = "guild run --force-flags main "
        base += "--stage" if guild_use_queues else "--background"
        subprocess.run(
            base.split() + [f"{name}={val}" for name, val in command_args.items()]
        )
    elif experiment_tracker == "aim":
        command = [sys.executable, "autopopulus/impute.py"]
        for name, val in command_args.items():
            command += [f"--{name}", str(val)]
        output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode("utf-8")
        aim_hash = re.search(r"(?:Aim Logger Hash: )(\w+)\b", str(output)).groups()[-1]
        command[1] = "autopopulus/evaluate.py"
        subprocess.run(command + ["--aim-hash", aim_hash])
        command[1] = "autopopulus/predict.py"
        subprocess.run(command + ["--aim-hash", aim_hash])
    elif experiment_tracker == "none":
        command = [sys.executable, "autopopulus/impute.py"]
        for name, val in command_args.items():
            command += [f"--{name}", str(val)]
        subprocess.run(command, stdout=subprocess.PIPE).stdout.decode("utf-8")
        command[1] = "autopopulus/evaluate.py"
        subprocess.run(command)
        command[1] = "autopopulus/predict.py"
        subprocess.run(command)


for dataset in datasets:
    if fully_observed:
        with open(f"dev/{dataset}_amputation_pattern_grid.txt", "r") as f:
            amputation_patterns = json.load(f)

    # https://my.guild.ai/t/command-run/146
    # https://my.guild.ai/t/running-cases-in-parallel/341
    for method in chosen_methods:
        print("======================================================")
        print(f"Staging {method} imputation...")
        if method == "none":
            run_command(
                {"method": "none", "fully-observed": "yes", "dataset": cli_str(dataset)}
            )
        else:
            command_args = {
                "method": cli_str(method),
                "dataset": cli_str(dataset),
            }
            # bootstrap evaluate the baseline models no matter what
            # manually pick the AE models to bootstrap eval later
            if method in imputer_groups["baseline"]:
                command_args["bootstrap-evaluate-imputer"] = cli_str(True)
            # Added to the end if the conditions are met otherwise nothing happens
            if method in imputer_groups["ae"]:
                command_args = {
                    **command_args,
                    "feature-map": cli_str(feature_mapping),
                    "replace-nan-with": cli_str(replace_nan_with),
                }
            elif (
                method in imputer_groups["variational"]
            ):  # on vae and dvae only try target_encode_categorical
                command_args = {
                    **command_args,
                    "feature-map": cli_str(feature_mapping_variational),
                    "replace-nan-with": cli_str(replace_nan_with),
                }
            # if command_args.get("feature-map", "") == "discretize_continuous":
            # command_args["uniform-prob"] = cli_str(True)

            if all_data:
                # When multiple flags have list values, Guild generates the cartesian product of all possible flag combinations.
                all_data_command_args = {**command_args, "fully-observed": "no"}
                run_command(all_data_command_args)
            if fully_observed:
                fully_observed_command_args = {
                    **command_args,
                    "fully-observed": "yes",
                    "percent-missing": cli_str(percent_missing),
                    "amputation-patterns": cli_str(amputation_patterns),
                }
                run_command(fully_observed_command_args)

if experiment_tracker == "guild" and guild_use_queues:
    print("======================================================")
    print("Starting Queues...")
    for _ in range(guild_use_queues):
        # https://stackoverflow.com/a/70072233/1888794
        subprocess.run("guild run queue run-once=yes -y", shell=True)
    # subprocess.run('guild view -h 127.0.0.1')
