from argparse import Namespace
from os import getcwd
from os.path import join
from shutil import copytree, ignore_patterns, rmtree
from typing import Dict, Any, Tuple

from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import TensorBoardLogger

import torch

# from ray_lightning.tune import get_tune_resources

import ray
import ray.tune as tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback


# Local
from autopopulus.utils.log_utils import (
    copy_log_from_tune,
    get_logdir,
    get_serialized_model_path,
    TUNE_LOG_DIR,
)
from autopopulus.models.ap import AEImputer
from autopopulus.data import CommonDataModule


def create_autoencoder(
    args: Namespace, data: CommonDataModule, settings: Dict
) -> AEImputer:
    logdir = get_logdir(args)
    if args.tune_n_samples:
        best_tune_logdir, best_model_config = run_tune(
            args, data, settings, args.experiment_name, args.tune_n_samples
        )
        # log.add_text("best_model_config", str(best_model_config))
        # Copy serialized model and logged values to local path, ignoring tune artifacts
        copy_log_from_tune(best_tune_logdir, logdir)
        copytree(
            join(best_tune_logdir, "serialized_models"),
            join(getcwd(), "serialized_models"),
            # ignore=ignore_patterns(
            # "checkpoint_*", "params.*", "progress.csv", "result.json", "*tfevents*"
            # ),
            dirs_exist_ok=True,
        )

        # Load up (now we can do from local because we copied over)
        best_checkpoint = get_serialized_model_path(
            f"AEDitto_{data.data_type_time_dim.name}", "pt"
        )
        ae_imputer = AEImputer.from_checkpoint(
            args, ae_from_checkpoint=best_checkpoint, **best_model_config
        )

        # Cleanup
        torch.cuda.empty_cache()
        rmtree(TUNE_LOG_DIR)  # delete tune files

        return ae_imputer

    # If not tuning assume we've been given a specific setting for hyperparams
    logger = TensorBoardLogger(logdir)
    ae_imputer = AEImputer.from_argparse_args(
        args,
        logger=logger,
        tune_callback=None,
        **settings,
    )
    ae_imputer.fit(data)
    return ae_imputer


def tune_model_ray(
    config,
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
):
    """NOTE: YOU CANNOT PASS THE SUMMARYWRITER HERE IT WILL CAUSE A PROBLEM WITH MULTIPROCESSING: RuntimeError: Queue objects should only be shared between processes through inheritance"""
    logger = TensorBoardLogger(get_logdir(args))

    ae_imputer = AEImputer.from_argparse_args(
        args,
        logger=logger,
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
) -> Tuple[str, Dict]:
    # ref: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
    """Gets the checkpoint path and config of the best model.
    https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html"""
    if args.fast_dev_run or args.limit_data:
        # Will have to set cuda_visible devices before running the python code if true
        ray.init(local_mode=True)  # Local debugging for tuning
        # tune will try everything in the grid, so just do 1
        hidden_layers_grid = [[0.5]]
        max_epochs = [3]
    else:
        hidden_layers_grid = [
            [0.5, 0.25, 0.5],
            [0.5],
            [1.0, 0.5, 1.0],
            [1.5],
            [1.0, 1.5, 1.0],
        ]
        max_epochs = [100, 200]
    data_type_time_dim_name = data.data_type_time_dim.name
    # 4 gpus means 1 gpu for 4 parallel trials WITHOUT DDP training.
    resources_per_trial = {"cpu": 8, "gpu": 1}
    args.num_gpus = resources_per_trial["gpu"]
    args.num_cpus = resources_per_trial["cpu"]
    metric = f"impute/{data_type_time_dim_name}/original/val-CWMAAPE-missingonly"
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
        metric=metric,
        trial_name_creator=lambda trial: trial.trial_id,
        # keep_checkpoints_num=1,
        # checkpoint_at_end=True,  # checkpoitns the tune trials not the model
        resources_per_trial=resources_per_trial,
        config={
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "l2_penalty": tune.loguniform(1e-5, 1),
            # "max_epochs": tune.choice(max_epochs),
            # "patience": tune.choice([3, 5, 10]),
            # assume discretized, so num inputs on the dataset is 56
            "hidden_layers": tune.grid_search(hidden_layers_grid),
            "tune_callback": TuneReportCallback(
                [
                    f"AE/{data_type_time_dim_name}/val-loss",
                    metric,
                ],
                on="validation_end",
            ),
        },
    )

    return (
        analysis.get_best_logdir(metric=metric, mode="min"),
        analysis.get_best_config(metric=metric, mode="min"),
    )
