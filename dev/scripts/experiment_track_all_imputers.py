from typing import Dict
import json
import re
import sys
import subprocess

# Enforce you're in the right env
output = subprocess.run(["conda", "list"], stdout=subprocess.PIPE).stdout.decode("utf-8")
env_name = re.search(r"(?:/mambaforge/envs/)(\w+)\b", str(output)).groups()[-1]
assert env_name == "ap", "You're not in the right conda environment. Did you forget to `mamba activate ap`?"

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
    "baseline": ["simple", "mice"],
    "ae": ["vanilla", "dae", "batchswap"],
    "vae": ["vae", "dvae"],
}
replace_nan_with = ["simple", "0"]

####################################
#  Select experiments to run here  #
####################################
# experiment switches: all experiments: none, baseline, ae, vae
# chosen_methods=[ "none", "baseline", "ae", "vae" ]
chosen_methods = ["none", "baseline", "ae", "vae"]
experiment_tracker = "guild"
# fully_observed=no uses entire dataset
all_data = True
# fully_observed=yes will ampute and impute a missingness scenario
fully_observed = False

if fully_observed:
    with open("dev/amputation_pattern_grid.txt", "r") as f:
        amputation_patterns = json.load(f)


def cli_str(obj) -> str:
    # format for split() also same thign for CLI. the str repr of lists/etc has spaces which will create problems.
    return str(obj).replace(" ", "")


def run_command(command_args: Dict[str, str]):
    if experiment_tracker == "guild":
        subprocess.run(
            "guild run main --background".split()
            + [f"{name}={val}" for name, val in command_args.items()]
        )
    elif experiment_tracker == "aim":
        command = [sys.executable, "autopopulus/impute.py"]
        for name, val in command_args.items():
            command += [f"--{name}", str(val)]
        output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode("utf-8")
        aim_hash = re.search(r"(?:Aim Logger Hash: )(\w+)\b", str(output)).groups()[-1]
        command[1] = "autopopulus/predict.py"
        subprocess.run(command + ["--aim-hash", aim_hash])


# https://my.guild.ai/t/command-run/146
# https://my.guild.ai/t/running-cases-in-parallel/341
for method in chosen_methods:
    print("======================================================")
    print(f"Staging {method} imputation...")
    if method == "none":
        run_command({"method": "none", "fully-observed": "yes"})
    else:
        command_args = {"method": cli_str(imputer_groups[method])}
        # Added to the end if the conditions are met otherwise nothing happens
        if method == "ae":
            command_args = {
                **command_args,
                "feature-map": cli_str(feature_mapping),
                "replace-nan-with": cli_str(replace_nan_with),
            }
        elif method == "vae":  # on vae and dvae only try target_encode_categorical
            command_args = {
                **command_args,
                "feature-map": cli_str(feature_mapping_variational),
                "replace-nan-with": cli_str(replace_nan_with),
            }

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

# print("======================================================")
# print("Starting Queues...")
# num_queues = 4
# for _ in range(num_queues):
# run_command('guild run queue  -y')
# run_command('guild view -h 127.0.0.1')
