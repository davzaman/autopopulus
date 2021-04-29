from argparse import Namespace
from typing import Dict, List, Optional, Union

import os
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from tensorboardX import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator


from data.utils import CommonDataModule
from utils.impute_metrics import MAAPE, RMSE


def get_serialized_model_path(modeln: str, ftype: str = "pkl") -> str:
    """Path to dump serialized model, whether it's autoencoder, or predictive model."""
    dir_name = "serialized_models"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    serialized_model_path = os.path.join(dir_name, f"{modeln}.{ftype}")
    return serialized_model_path


def log_imputation_performance(
    results: List[pd.DataFrame],
    data: CommonDataModule,
    log: SummaryWriter,
    runtest: bool,
):
    """For a given imputation method, logs the performance for the following metrics (matches AE). Assumes results are in order: train, val, test."""
    metrics = {"RMSE": RMSE, "MAAPE": MAAPE}
    if runtest:
        stages = ["train", "val", "test"]
    else:
        stages = ["train", "val"]

    for stage_i, stage in enumerate(stages):
        est = results[stage_i]

        true = getattr(data, f"X_true_{stage}")
        # if the original dataset contains nans and we're not filtering to fully observed, need to fill in ground truth too for metric computation
        ground_truth_non_missing_mask = ~np.isnan(true)
        true = true.where(ground_truth_non_missing_mask, est)

        missing_mask = getattr(data, f"X_{stage}").isna()

        # START HERE
        for name, metricfn in metrics.items():
            metric = metricfn(est, true)
            metric_missing_only = metricfn(est, true, missing_mask)
            print(
                f"{name}: {metric}.\n Missing cols only, {name}: {metric_missing_only}"
            )
            log.add_scalar(f"impute/{stage}-{name}", metric)
            log.add_scalar(f"impute/{stage}-{name}-missingonly", metric_missing_only)


def get_logger(
    args: Namespace, predictive_model: Optional[str] = None
) -> SummaryWriter:
    """Get the universal logger for tensorboardX."""
    if not args.tbX_on:
        return
    logdir = (
        (
            f"F.O./"
            f"{args.percent_missing}/"
            f"{args.missingness_mechanism}/"
            f"{args.method}"
        )
        if args.fully_observed
        else f"full/{args.method}"
    )
    if predictive_model:
        logdir += f"/{predictive_model}"
    return SummaryWriter(logdir=logdir)
    # return SummaryWriter(logdir=logdir, write_to_disk=args.tbX_on)


def add_scalars(
    logger: SummaryWriter,
    tag_scalar_dict: Dict[str, float],
    global_step: Optional[int] = None,
    walltime: Optional[float] = None,
    prefix: Optional[str] = None,
) -> None:
    """Adds scalars from dict but not to same plot."""
    if not logger:
        return
    for tag, scalar_value in tag_scalar_dict.items():
        if prefix:
            logger.add_scalar(f"{prefix}/{tag}", scalar_value, global_step, walltime)
        else:
            logger.add_scalar(tag, scalar_value, global_step, walltime)


def add_all_text(
    logger: SummaryWriter,
    tag_scalar_dict: Dict[str, str],
    global_step: Optional[int] = None,
    walltime: Optional[float] = None,
) -> None:
    """Adds text from dict."""
    if not logger:
        return
    for tag, text_string in tag_scalar_dict.items():
        logger.add_text(tag, text_string, global_step, walltime)


def copy_log_from_tune(tune_log_path: str, logger: SummaryWriter):
    """Ray tune stores the results as "ray/tune/val-loss" for example. We want to copy these over locally so we can remove the tune files and readily compare the output later."""
    for summary in summary_iterator(tune_log_path):
        for step, v in enumerate(summary.summary.value):
            if "ray/tune" in v.tag:
                new_name = v.tag[len("ray/tune/") :]  # remove ray/tune/
                logger.add_scalar(new_name, v.simple_value, step)


class MyLogger(LightningLoggerBase):
    def __init__(self, summarywriter: SummaryWriter):
        super().__init__()
        self._experiment = summarywriter

    @property
    def name(self):
        return "MyLogger"

    @property
    @rank_zero_experiment
    def experiment(self) -> SummaryWriter:
        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        return self._experiment

    @property
    def version(self) -> Union[int, str]:
        return "1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                except Exception as e:
                    m = f"\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor."
                    type(e)(e.message + m)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Any code that needs to be run after training finishes goes here
        self.experiment.flush()
