from os import walk
from os.path import join, dirname, basename
from regex import search
import pandas as pd

tracking_uri = "/home/davina/Private/repos/autopopulus/mlruns"
# run_commands = pd.read_csv("/home/davina/Private/repos/autopopulus/run_progress.csv")
command_args_list = []
for root, dirs, files in walk(tracking_uri):
    for file in files:
        if file == "failed":  # this is in tags
            # print(run_commands[run_commands["run_id"] == basename(dirname(root))])
            # build command args
            command_args = {}
            for tag in [
                "method",
                "fully_observed",
                "dataset",
                "bootstrap_evaluate_imputer",
                "feature_map",
                "replace_nan_with",
                "percent_missing",
                "amputation_patterns",
            ]:
                if tag in files:
                    with open(join(root, tag), "r") as f:
                        command_args[tag] = f.read()
            if command_args:
                command_args_list.append(command_args)

# print(len(command_args_list))
from pickle import dump

with open("retry_runs.pkl", "wb") as file:
    dump(command_args_list, file)
