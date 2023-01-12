from warnings import warn
from argparse import Namespace
from os import getcwd
from os.path import join
from shutil import copytree, ignore_patterns, rmtree
from typing import Dict, Any, Tuple

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

import torch
import nni
from nni.experiment import Experiment


# Local
from autopopulus.utils.log_utils import (
    copy_log_from_tune,
    get_logdir,
    get_serialized_model_path,
    TUNE_LOG_DIR,
)
from autopopulus.models.ap import AEImputer
from autopopulus.data import CommonDataModule


class NNIReportCallback(Callback):
    """Based on Optuna PytorchLightningPruningCallback and https://nni.readthedocs.io/en/stable/_modules/nni/nas/evaluator/pytorch/lightning.html"""

    def __init__(self, monitor: str) -> None:
        super().__init__()
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking and nni.get_current_parameter() is None:
            return

        if trainer.is_global_zero:
            nni.report_intermediate_result(self.get_metric())

    def on_fit_end(self):
        if nni.get_current_parameter() is not None:
            nni.report_final_result(self.get_metric())

    def on_validation_end(self):
        if nni.get_current_parameter() is not None:
            nni.report_final_result(self.get_metric())

    def get_metric(self, trainer: Trainer) -> float:
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            warn(
                "The metric '{}' is not in the evaluation logs. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            return
        return current_score.item()


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


def get_tune_grid(args: Namespace) -> Dict[str, Any]:
    if args.fast_dev_run or args.limit_data:
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

    # https://nni.readthedocs.io/en/stable/hpo/search_space.html
    config = {
        "learning_rate": {"_type": "loguniform", "_value": [1e-5, 1e-1]},
        "l2_penalty": {"_type": "loguniform", "_value": [1e-5, 1]},
        "hidden_layers": {"_type": "choice", "_value": hidden_layers_grid},
        # "max_epochs": {"_type": "choice", "_value": max_epochs},
        # "patience": {"_type": "choice", "_value": [3,5,10]},
    }

    return config


def tune_model(
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
):
    """NOTE: YOU CANNOT PASS THE SUMMARYWRITER HERE IT WILL CAUSE A PROBLEM WITH MULTIPROCESSING: RuntimeError: Queue objects should only be shared between processes through inheritance"""
    logger = TensorBoardLogger(get_logdir(args))

    config = nni.get_next_parameter()
    data_type_time_dim_name = data.data_type_time_dim.name
    ae_imputer = AEImputer.from_argparse_args(
        args,
        logger=logger,
        tune_callback=NNIReportCallback(f"AE/{data_type_time_dim_name}/val-loss")
        ** settings,
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
    # https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
    # https://nni.readthedocs.io/en/stable/reference/experiment_config.html
    search_space = get_tune_grid(args)
    experiment = Experiment("local")

    experiment.config.experiment_name = args.experiment_name
    experiment.config.trial_command = "python main.py"
    experiment.config.trial_code_directory = "."
    # https://nni.readthedocs.io/en/stable/hpo/search_space.html#search-space-types-supported-by-each-tuner
    experiment.config.tuner.name = "TPE"
    experiment.config.tuner.class_args["optimize_mode"] = "minimize"
    experiment.config.search_space = search_space
    experiment.config.max_trial_number = tune_n_samples
    experiment.config.trial_gpu_number = 2
    experiment.config.training_service.use_active_gpu = True
    experiment.config.trial_concurrency = 2

    experiment.run(4632)
    experiment.stop()

    return (
        analysis.get_best_logdir(metric=metric, mode="min"),
        analysis.get_best_config(metric=metric, mode="min"),
    )
