from argparse import Namespace
from logging import FileHandler, StreamHandler, basicConfig, info, INFO
from shutil import copy
from typing import Dict, List, Optional, Union

from regex import search
from os.path import join, exists
from os import makedirs, walk
import sys
import numpy as np
import pandas as pd

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter
from torch import isnan, tensor

import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event


from autopopulus.data import CommonDataModule
from autopopulus.utils.impute_metrics import CWMAAPE, CWRMSE

TUNE_LOG_DIR = "tune_results"


def init_new_logger(fname: Optional[str] = None):
    handlers = [StreamHandler(sys.stdout)]
    if fname is not None:
        handlers.append(FileHandler(fname))

    basicConfig(
        level=INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # print to stdout and log to file
        handlers=handlers,
    )


def get_serialized_model_path(
    modeln: str, ftype: str = "pkl", trial_num: Optional[int] = None
) -> str:
    """Path to dump serialized model, whether it's autoencoder, or predictive model."""
    dir_name = "serialized_models"
    if trial_num is not None:
        dir_name = join(TUNE_LOG_DIR, f"trial_{trial_num}", dir_name)
    if not exists(dir_name):
        makedirs(dir_name)
    serialized_model_path = join(dir_name, f"{modeln}.{ftype}")
    return serialized_model_path


def log_imputation_performance(
    results: List[pd.DataFrame],
    data: CommonDataModule,
    log: SummaryWriter,
    runtest: bool,
):
    """For a given imputation method, logs the performance for the following metrics (matches AE). Assumes results are in order: train, val, test."""
    metrics = {"RMSE": CWRMSE, "MAAPE": CWMAAPE}
    if runtest:
        stages = ["train", "val", "test"]
    else:
        stages = ["train", "val"]

    for stage_i, stage in enumerate(stages):
        est = results[stage_i]

        true = data.splits["ground_truth"][stage]
        # if the original dataset contains nans and we're not filtering to fully observed, need to fill in ground truth too for metric computation
        ground_truth_non_missing_mask = ~np.isnan(true)
        true = true.where(ground_truth_non_missing_mask, est)

        orig = data.splits["data"][stage]
        if isinstance(orig, pd.DataFrame):
            orig = tensor(orig.values)
        missing_mask = isnan(orig).bool()

        # START HERE
        for name, metricfn in metrics.items():
            metric = metricfn(est, true)
            metric_missing_only = metricfn(est, true, missing_mask)
            rank_zero_info(
                f"{name}: {metric}.\n Missing cols only, {name}: {metric_missing_only}"
            )
            log.add_scalar(f"impute/{stage}-{name}", metric)
            log.add_scalar(f"impute/{stage}-{name}-missingonly", metric_missing_only)


def get_logdir(args: Namespace) -> str:
    """Get logging directory based on experiment settings."""
    prefix = f"{TUNE_LOG_DIR}/trial_{args.trial_num}/" if "trial_num" in args else ""
    # Missingness scenario could be 1 mech or mixed
    if args.amputation_patterns:
        pattern_mechanisms = ",".join(
            [pattern["mechanism"] for pattern in args.amputation_patterns]
        )
        dir_name = (
            f"{prefix}F.O./{args.percent_missing}/{pattern_mechanisms}/{args.method}"
        )
    else:
        dir_name = f"{prefix}full/{args.method}"
    if not exists(dir_name):
        makedirs(dir_name)
    return dir_name


def get_summarywriter(
    logdir: Optional[str], predictive_model: Optional[str] = None
) -> SummaryWriter:
    """Get the universal logger for tensorboard."""
    if not logdir:
        return
    if predictive_model:
        return SummaryWriter(log_dir=join(logdir, predictive_model))
    return SummaryWriter(log_dir=logdir)


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


def copy_log_from_tune(
    best_tune_logdir: str, logdir: str, logger: SummaryWriter = None
):
    """
    We want to copy these over locally so we can remove the tune files and readily compare the output later.
    Walk through the best tune run logdirectory,
        ignoring top-level (which is tune metadata we don't care about),
        and copy over all tfevents.
    """
    for root, dirs, files in walk(best_tune_logdir):
        if root != best_tune_logdir:  # ignore top-level
            for file in files:
                if search("tfevents", file):  # only tfevent files
                    copy(join(root, file), logdir)
