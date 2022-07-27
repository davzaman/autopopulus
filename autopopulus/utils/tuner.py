from argparse import Namespace
from logging import info
from shutil import rmtree
from typing import Dict, Any, Tuple
import torch
import ray.tune as tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from pytorch_lightning.utilities import rank_zero_info


# Local
from autopopulus.utils.log_utils import get_logdir, get_logger
from autopopulus.models.ap import AEImputer
from autopopulus.data import CommonDataModule

TUNE_LOG_DIR = "tune_results"


def create_autoencoder_with_tuning(
    args: Namespace,
    data: CommonDataModule,
    settings: Dict,
):
    log = get_logger(get_logdir(args))

    if args.runtune:
        best_model_config, best_model_path = run_tune(
            args, data, settings, args.experiment_name, args.tune_n_samples
        )
        # update settings with best config
        settings.update(best_model_config)
    ae_imputer = AEImputer.from_argparse_args(
        args,
        summarywriter=log,
        runtune=False,
        **settings,
    )
    # log config
    log.add_text("config", str(settings))
    if args.runtune:  # delete tune files
        rmtree(TUNE_LOG_DIR)

    # so we don't get cuda out of memory when trying to do this last step
    torch.cuda.empty_cache()

    ae_imputer.fit(data)
    return ae_imputer


def tune_model_ray(
    config,
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
):
    """NOTE: YOU CANNOT PASS THE SUMMARYWRITER HERE IT WILL CAUSE A PROBLEM WITH MULTIPROCESSING: RuntimeError: Queue objects should only be shared between processes through inheritance"""
    log = get_logger(get_logdir(args))
    ae_imputer = AEImputer.from_argparse_args(
        args,
        summarywriter=log,
        runtune=True,
        **settings,
        **config,
    )
    ae_imputer.fit(data)


def run_tune(
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
    experiment_name: str,
    tune_n_samples: int = 1,
) -> Tuple[Dict, str]:
    """Gets the config and logdir of the best model.
    https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html"""
    analysis = tune.run(
        tune.with_parameters(
            tune_model_ray,
            args=args,
            data=data,
            settings=settings,
        ),
        name=experiment_name,
        local_dir=TUNE_LOG_DIR,
        num_samples=tune_n_samples,
        scheduler=ASHAScheduler(),
        mode="min",
        metric="impute/val-MAAPE-missingonly",
        resources_per_trial={"cpu": 32, "gpu": 4},
        config={
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "l2_penalty": tune.loguniform(1e-5, 1),
            "max_epochs": tune.choice([100, 200]),
            "patience": tune.choice([3, 5, 10]),
            # assume discretized, so num inputs on the dataset is 56
            "hidden_layers": tune.grid_search(
                [
                    [0.5, 0.25, 0.5],
                    [0.5],
                    [1.0, 0.5, 1.0],
                    [1.5],
                    [1.0, 1.5, 1.0],
                ]
            ),
        },
    )

    for metricn in [
        "AE/val-loss",
        "impute/val-RMSE",
        "impute/val-RMSE-missingonly",
        "impute/val-MAAPE",
        "impute/val-MAAPE-missingonly",
    ]:
        rank_zero_info(
            f"Best {metricn} config is:",
            analysis.get_best_config(metric=metricn, mode="min"),
        )
        rank_zero_info(
            f"Best {metricn} logdir is:",
            analysis.get_best_logdir(metric=metricn, mode="min"),
        )

    return (
        analysis.get_best_config(metric="impute/val-MAAPE-missingonly", mode="min"),
        analysis.get_best_logdir(metric="impute/val-MAAPE-missingonly", mode="min"),
    )
