from argparse import Namespace
from typing import Callable, Dict, List, Optional, Union
from pandas import DataFrame
import numpy as np
from numpy.random import default_rng
from torch import isnan, tensor
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_warn

# Local
from autopopulus.utils.log_utils import IMPUTE_METRIC_TAG_FORMAT, BasicLogger
from autopopulus.task_logic import (
    baseline_static_imputation,
    baseline_longitudinal_imputation,
)
from autopopulus.utils.utils import rank_zero_print, resample_indices_only
from autopopulus.data import CommonDataModule
from autopopulus.utils.impute_metrics import MAAPEMetric, RMSEMetric, universal_metric


def baseline_imputation_logic(
    args: Namespace, data: CommonDataModule
) -> Dict[str, DataFrame]:
    """
    Wrapper for Baseline static methods: setup dataset and logging.
    """

    if hasattr(baseline_static_imputation, args.method):
        module = baseline_static_imputation
    else:
        module = baseline_longitudinal_imputation
    fn = getattr(module, args.method)
    # will create train/val/test
    data.setup("fit")

    imputed_data_per_split = fn(args, data)
    if any(
        [imputed_data.isna().any().any() for imputed_data in imputed_data_per_split]
    ):
        rank_zero_warn("NaNs still found in imputed data.")
    log = BasicLogger(
        args=args, experiment_name=args.experiment_name, verbose=args.verbose
    )
    evaluate_baseline_imputation(
        args, data, imputed_data_per_split, get_metrics(data), log
    )

    return imputed_data_per_split


def get_metrics(data: CommonDataModule) -> List[Dict[str, Union[str, Callable]]]:
    return [
        {
            "name": "RMSE",
            "fn": universal_metric(
                RMSEMetric(columnwise=True, nfeatures=data.nfeatures["original"])
            ),
            "reduction": "CW",
        },
        {
            "name": "MAAPE",
            "fn": universal_metric(
                MAAPEMetric(columnwise=True, nfeatures=data.nfeatures["original"])
            ),
            "reduction": "CW",
        },
        {"name": "RMSE", "fn": universal_metric(RMSEMetric()), "reduction": "EW"},
        {"name": "MAAPE", "fn": universal_metric(MAAPEMetric()), "reduction": "EW"},
    ]


def evaluate_baseline_imputation(
    args: Namespace,
    data: CommonDataModule,
    imputed_data_per_split: Dict[str, DataFrame],
    metrics: List[Dict[str, Union[str, Callable]]],
    log: BasicLogger,
):
    """This mirrors AEDitto.metric_logging_step"""
    if args.method == "none":  # no performance to log
        return
    if data.ground_truth_has_nans:  # same as semi_observed_training
        return

    for split, imputed_data in imputed_data_per_split.items():
        pred = imputed_data
        data = data.splits["data"][split]
        true = data.splits["ground_truth"][split]
        where_data_are_missing = isnan(data)
        if args.bootstrap_eval_imputer and split == "test":
            gen = default_rng(args.seed)
            for b in tqdm(range(args.num_bootstraps)):
                bootstrap_indices = resample_indices_only(len(true), gen)
                log_baseline_imputation_performance(
                    pred.iloc[bootstrap_indices],
                    true.iloc[bootstrap_indices],
                    where_data_are_missing.iloc[bootstrap_indices],
                    split,
                    metrics,
                    log,
                    global_step=b,
                )
        else:
            log_baseline_imputation_performance(
                pred, true, where_data_are_missing, split, metrics, log
            )
    log.close()


def log_baseline_imputation_performance(
    est,
    true,
    where_data_are_missing,
    split: str,
    metrics,
    log: BasicLogger,
    global_step: Optional[int] = None,
):
    """For a given imputation method, logs the performance for the following metrics (matches AE). Assumes results are in order: train, val, test."""
    # START HERE
    for metric in metrics:
        for filter_subgroup in ["all", "missingonly"]:
            if filter_subgroup == "missingonly":
                val = metric["fn"](est, true, where_data_are_missing)
            else:
                val = metric["fn"](est, true)
            log.add_scalar(
                val,
                metric["name"],
                global_step=global_step,
                context={
                    "step": "impute",
                    "split": split,
                    "feature_space": "original",
                    "filter_subgroup": filter_subgroup,
                    "reduction": metric["reduction"],
                },
                tb_name_format=IMPUTE_METRIC_TAG_FORMAT,
            )
