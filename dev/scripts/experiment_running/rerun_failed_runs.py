from os import walk
from os.path import join

from dev.scripts.utils import EXPERIMENT_RUNNING_ARGS, RunManager

if __name__ == "__main__":
    experiment_tracker = "mlflow"

    tracking_uri = "/home/davina/Private/repos/autopopulus/mlruns"
    # run_commands = pd.read_csv("/home/davina/Private/repos/autopopulus/run_progress.csv")
    command_args_list = []
    for root, dirs, files in walk(tracking_uri):
        for file in files:
            if file == "failed":  # this is in tags
                # print(run_commands[run_commands["run_id"] == basename(dirname(root))])
                # build command args
                command_args = {}
                for tag in EXPERIMENT_RUNNING_ARGS:
                    if tag in files:
                        with open(join(root, tag), "r") as f:
                            command_args[tag] = f.read()
                if command_args:
                    command_args_list.append(command_args)

    run_manager = RunManager(experiment_tracker=experiment_tracker)
    for command_args in command_args_list:
        run_manager.run_pipeline(command_args)
    run_manager.close()
