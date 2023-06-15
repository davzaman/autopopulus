import subprocess
from os.path import join
import pandas as pd
from guild.run import Run


def get_best_performance_per_mechanism(
    impute_data: pd.DataFrame,
    # metric: str = "test/original/missingonly/CW/mixed/MAAPE_CategoricalError",
    metric: str = "test/original/missingonly/CW/mixed/MAAPECategoricalError",
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


def eval_best_on_semi_observed(eval_run: Run):
    """
    Evaluate the `main` parent operation associated with the best `eval` run
        on the remaining semi-observed subset of data.
    This will retrain the model, but with no tuning, using the best params from the impute step's tuning.
    We cannot bootstrap evaluate here since the test set will have NaNs (therefore no metrics).
    """
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
    command = f"guild run --proto {parent_hash} method={method} tune-n-samples=0 evaluate-on-remaining-semi-observed=yes bootstrap-evaluate-imputer=no ae-hparams-from-checkpoint={hparams_path} experiment-name=BEST_AE_PER_MECH --force-sourcecode -y"
    subprocess.run(command.split())


def eval_best_bootstrap(eval_run: Run):
    """Rerun the `eval` op but with bootstrap evaluation turned on."""
    method = eval_run.get("flags")["method"]
    command = f"guild run --proto {run.id} method={method} bootstrap-evaluate-imputer=yes experiment-name=BEST_AE_PER_MECH -y"  # --force-sourcecode -y"
    subprocess.run(command.split())


if __name__ == "__main__":
    impute_data = pd.read_pickle(
        "/home/davina/Private/repos/autopopulus/guild_runs/guild_impute_results.pkl"
    )
    best_for_mech = get_best_performance_per_mechanism(impute_data)
    for run in best_for_mech["run"]:
        # want to rerun per mechanism even if the autoencoder type is the same
        eval_best_on_semi_observed(run)
        eval_best_bootstrap(run)
