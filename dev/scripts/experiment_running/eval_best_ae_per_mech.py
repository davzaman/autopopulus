import subprocess
from os.path import join
from typing import Union
import pandas as pd
from guild.run import Run

from dev.scripts.utils import EXPERIMENT_RUNNING_ARGS, RunManager


def get_best_performance_per_mechanism(
    impute_data: pd.DataFrame,
    # metric: str = "test/original/missingonly/CW/mixed/MAAPE_CategoricalError",
    metric: str = "test/original/missingonly/CW/mixed/MAAPE_CategoricalError",
):
    best_runs = impute_data.loc[
        impute_data[
            (impute_data["split"] == "test")
            & ~(impute_data["method"].isin(["mice", "simple", "knn"]))
        ]
        .groupby(["mechanism", "metric"])["val"]
        .idxmin()  # min because they're errors
    ]
    return best_runs[best_runs["metric"] == metric]


def eval_best_on_semi_observed(
    run_manager: RunManager, eval_run: Union[pd.Series, Run]
):
    """
    Evaluate the `main` parent operation associated with the best `eval` run
        on the remaining semi-observed subset of data.
    This will retrain the model, but with no tuning, using the best params from the impute step's tuning.
    We cannot bootstrap evaluate here since the test set will have NaNs (therefore no metrics).
    """
    if run_manager.experiment_tracker == "guild":
        # load up the model with the same params from tuning
        parent_hash = eval_run.get("env")["GUILD_RUNS_PARENT"]
        method = eval_run.get("flags")["method"]
        hparams_path = join(
            eval_run.get("env")["GUILD_HOME"],
            "runs",
            parent_hash,
            "impute",
            "STATIC",
            method,
            "params.json",
        )
        # rerun main operation with new flags
        # TODO: remove force-sourcecode?
        command = f"guild run --proto {parent_hash} method={method} tune-n-samples=0 evaluate-on-remaining-semi-observed=yes bootstrap-evaluate-imputer=no ae-hparams-from-checkpoint={hparams_path} experiment-name=BEST_AE_PER_MECH_SEMI_OBSERVED --force-sourcecode -y"
        subprocess.run(command.split())
    else:
        command_args = eval_run[
            [x.replace("_", "-") for x in EXPERIMENT_RUNNING_ARGS]
        ].to_dict()
        command_args.update(
            {
                "bootstrap-evaluate-imputer": "no",
                "experiment-name": "BEST_AE_PER_MECH_SEMI_OBSERVED",
                "tune-n-samples": "0",
                "evaluate-on-remaining-semi-observed": "yes",
                "ae-hparams-from-checkpoint": join(
                    eval_run["artifact_uri"].replace("file://", ""), "params.json"
                ),
            }
        )
        run_manager.run_pipeline(command_args)


def eval_best_bootstrap(run_manager: RunManager, eval_run: Union[pd.Series, Run]):
    """Rerun the `eval` op but with bootstrap evaluation turned on."""
    if run_manager.experiment_tracker == "guild":
        method = eval_run.get("flags")["method"]
        command = f"guild run --proto {run.id} method={method} bootstrap-evaluate-imputer=yes experiment-name=BEST_AE_PER_MECH -y"  # --force-sourcecode -y"
        subprocess.run(command.split())
    else:
        command_args = eval_run[
            [x.replace("_", "-") for x in EXPERIMENT_RUNNING_ARGS]
        ].to_dict()
        command_args.update(
            {
                "bootstrap-evaluate-imputer": "yes",
                "experiment-name": "BEST_AE_PER_MECH",
                "parent-hash": eval_run["run_id"],
            }
        )
        run_manager.run("autopopulus/evaluate.py", command_args)


if __name__ == "__main__":
    tracker = "mlflow"
    dataset = "cure_ckd"
    impute_data = pd.read_pickle(
        f"/home/davina/Private/repos/autopopulus/guild_runs/{tracker}_{dataset}_impute_results.pkl"
    )
    best_for_mech = get_best_performance_per_mechanism(impute_data)
    run_manager = RunManager(experiment_tracker=tracker, continue_runs=False)
    for run in best_for_mech["run"]:
        # want to rerun per mechanism even if the autoencoder type is the same
        eval_best_on_semi_observed(run_manager, run)
        eval_best_bootstrap(run_manager, run)
    run_manager.close()
