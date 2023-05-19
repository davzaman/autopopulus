from argparse import Namespace
from os import getcwd
from os.path import join
from shutil import copytree, ignore_patterns, rmtree
from typing import Dict, Any, List, Tuple
import logging

from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies.strategy import Strategy

import torch

import ray
import ray.tune as tune
import ray.air as air
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback as raytune_TuneReportCallback,
)

# Local
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    AutoencoderLogger,
    BasicLogger,
    copy_log_from_tune,
    get_serialized_model_path,
    TUNE_LOG_DIR,
)
from autopopulus.models.ap import AEImputer
from autopopulus.data import CommonDataModule


def create_autoencoder_with_tuning(
    args: Namespace, data: CommonDataModule, settings: Dict
) -> AEImputer:
    logdir = BasicLogger.get_logdir(**BasicLogger.get_base_context_from_args(args))
    best_tune_logdir, best_model_config = run_tune(args, data, settings)
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


def get_tune_grid(args: Namespace) -> Dict[str, Any]:
    if args.fast_dev_run or args.limit_data:
        # Will have to set cuda_visible devices before running the python code if true
        # ray.init(local_mode=True)  # Local debugging for tuning
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

    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "l2_penalty": tune.loguniform(1e-5, 1),
        # "max_epochs": tune.choice(max_epochs),
        # "patience": tune.choice([3, 5, 10]),
        # assume discretized, so num inputs on the dataset is 56
        "hidden_layers": tune.choice(hidden_layers_grid),
    }
    return config


def tune_model(
    config,
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
    metrics: List[str],
    callback: raytune_TuneReportCallback,
    strategy: Strategy,
):
    logger = AutoencoderLogger(args)

    ae_imputer = AEImputer.from_argparse_args(
        args,
        logger=logger,
        tune_callback=callback(metrics, on="validation_end"),
        strategy=strategy,
        **settings,
        **config,
    )
    ae_imputer.fit(data)


def run_tune(
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
) -> Tuple[str, Dict]:
    """
    Gets the checkpoint path and config of the best model.
    Uses the total hardware on the machine and the gpus per trial requested
        to determine how many CPUs per trial.
    num_workers is not dynamically determined.
        More workers means more parallel dataloading, but more overhead (slower start).

    https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html
    Ref: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
    """

    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init
    # dashboard default port: 8265, dashboard requires ray-default package
    # ray.init(include_dashboard=True, logging_level=logging.DEBUG)
    # ray.init(include_dashboard=True, local_mode=True)  # async actor does'nt work
    ray.init(include_dashboard=True)

    data_type_time_dim_name = data.data_type_time_dim.name
    ncpu_per_gpu = args.total_cpus_on_machine // args.total_gpus_on_machine

    # set args to pass in
    args.num_workers = min(4, ncpu_per_gpu)

    # get callback, strat, and resources per trial depending on ngpus per trial.
    if args.n_gpus_per_trial <= 1:
        args.num_gpus = args.n_gpus_per_trial

        # Set callback and strat
        callback = raytune_TuneReportCallback
        strategy = None  # no custom strategy, use pl defaults

        resources_per_trial = {"cpu": ncpu_per_gpu, "gpu": args.n_gpus_per_trial}
    else:
        # TODO: upgrade to ray 2.4.0
        print("Not implemented yet")
    metrics = [
        IMPUTE_METRIC_TAG_FORMAT.format(
            name="loss",
            feature_space="original",
            filter_subgroup="all",
            reduction="NA",
            split="val",
        ),
        IMPUTE_METRIC_TAG_FORMAT.format(
            name="MAAPE",
            feature_space="original",
            filter_subgroup="missingonly",
            reduction="CW",
            split="val",
        ),
    ]
    key_metric = metrics[1]

    # https://docs.ray.io/en/latest/tune/api_docs/execution.html#tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                tune_model,
                args=args,
                data=data,
                settings=settings,
                metrics=metrics,
                callback=callback,
                strategy=strategy,
            ),
            resources=resources_per_trial,
        ),
        tune_config=tune.TuneConfig(
            mode="min",
            metric=key_metric,
            scheduler=ASHAScheduler(),
            num_samples=args.tune_n_samples,
            trial_name_creator=lambda trial: trial.trial_id,
        ),
        run_config=air.RunConfig(
            name=args.experiment_name,
            local_dir=TUNE_LOG_DIR,
        ),
        param_space=get_tune_grid(args),
    )
    # https://docs.ray.io/en/latest/tune/api_docs/result_grid.html
    results = tuner.fit()
    # https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.result.Result
    best_result = results.get_best_result(metric=key_metric, mode="min")

    return (best_result.log_dir, best_result.config)
