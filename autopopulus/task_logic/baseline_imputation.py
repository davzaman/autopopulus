from argparse import Namespace
from typing import Dict
from pandas import DataFrame
import numpy as np
from torch import isnan, tensor

# Local
from autopopulus.utils.log_utils import IMPUTE_METRIC_TAG_FORMAT, BasicLogger
from autopopulus.task_logic import (
    baseline_static_imputation,
    baseline_longitudinal_imputation,
)
from autopopulus.utils.utils import rank_zero_print
from autopopulus.data import CommonDataModule
from autopopulus.utils.impute_metrics import CWRMSE, EWMAAPE, EWRMSE


def baseline_imputation_logic(
    args: Namespace, data: CommonDataModule
) -> Dict[str, Dict[str, DataFrame]]:
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

    imputed_data = fn(args, data)

    ## LOGGING ##
    if (
        args.method != "none"
    ):  # Logging here if baseline experiment (not fully observed)
        log = BasicLogger(args=args, experiment_name=args.experiment_name)
        # if args.fully_observed:
        log_baseline_imputation_performance(imputed_data, data, log)
        log.close()

    return imputed_data


def log_baseline_imputation_performance(
    results: Dict[str, DataFrame],
    data: CommonDataModule,
    log: BasicLogger,
):
    """For a given imputation method, logs the performance for the following metrics (matches AE). Assumes results are in order: train, val, test."""
    metrics = [
        {"name": "RMSE", "fn": CWRMSE, "reduction": "CW"},
        {"name": "RMSE", "fn": EWRMSE, "reduction": "EW"},
        {"name": "MAAPE", "fn": EWMAAPE, "reduction": "EW"},
    ]
    for split, imputed_data in results.items():
        est = imputed_data

        true = data.splits["ground_truth"][split]
        # if the original dataset contains nans and we're not filtering to fully observed, need to fill in ground truth too for metric computation
        ground_truth_non_missing_mask = ~np.isnan(true)
        true = true.where(ground_truth_non_missing_mask, est)

        orig = data.splits["data"][split]
        if isinstance(orig, DataFrame):
            orig = tensor(orig.values)
        missing_mask = isnan(orig).bool()

        # START HERE
        for metric in metrics.items():
            val = metric["fn"](est, true)
            val_missing_only = metric["fn"](est, true, missing_mask)
            rank_zero_print(
                f"{metric['name']}: {val}.\n Missing cols only, {metric['name']}: {val_missing_only}"
            )
            log.add_scalar(
                val,
                metric["name"],
                context={
                    "step": "impute",
                    "split": split,
                    "feature_space": "original",
                    "filter_subgroup": "all",
                    "reduction": metric["reduction"],
                },
                tb_name_format=IMPUTE_METRIC_TAG_FORMAT,
            )
            log.add_scalar(
                val_missing_only,
                metric["name"],
                context={
                    "step": "impute",
                    "split": split,
                    "feature_space": "original",
                    "filter_subgroup": "missingonly",
                    "reduction": metric["reduction"],
                },
                tb_name_format=IMPUTE_METRIC_TAG_FORMAT,
            )
