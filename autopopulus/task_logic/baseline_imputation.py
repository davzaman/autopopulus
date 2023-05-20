from argparse import Namespace
from typing import Dict, Optional
from pandas import DataFrame
import numpy as np
from numpy.random import default_rng
from torch import isnan, tensor
from tqdm import tqdm

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
        metrics = [
            {
                "name": "RMSE",
                "fn": universal_metric(
                    RMSEMetric(columnwise=True, nfeatures=data.nfeatures["original"])
                ),
                "reduction": "CW",
            },
            {
                "name": "RMSE",
                "fn": universal_metric(
                    MAAPEMetric(columnwise=True, nfeatures=data.nfeatures["original"])
                ),
                "reduction": "CW",
            },
            {"name": "RMSE", "fn": universal_metric(RMSEMetric()), "reduction": "EW"},
            {"name": "MAAPE", "fn": universal_metric(MAAPEMetric()), "reduction": "EW"},
        ]
        for split, imputed_data in imputed_data.items():
            est = imputed_data

            true = data.splits["ground_truth"][split]
            # if the original dataset contains nans and we're not filtering to fully observed, need to fill in ground truth too for metric computation
            ground_truth_non_missing_mask = ~np.isnan(true)
            true = true.where(ground_truth_non_missing_mask, est)

            orig = data.splits["data"][split]
            # if isinstance(orig, DataFrame):
            #     orig = tensor(orig.values)
            missing_mask = orig.isna()
            if args.bootstrap_eval_imputer and split == "test":
                gen = default_rng(args.seed)
                for b in tqdm(range(args.num_bootstraps)):
                    bootstrap_indices = resample_indices_only(len(true), gen)
                    log_baseline_imputation_performance(
                        est.iloc[bootstrap_indices],
                        true.iloc[bootstrap_indices],
                        missing_mask.iloc[bootstrap_indices],
                        split,
                        metrics,
                        log,
                        global_step=b,
                    )
            else:
                log_baseline_imputation_performance(
                    est, true, missing_mask, split, metrics, log
                )
        log.close()

    return imputed_data


def log_baseline_imputation_performance(
    est,
    true,
    missing_mask,
    split: str,
    metrics,
    log: BasicLogger,
    global_step: Optional[int] = None,
):
    """For a given imputation method, logs the performance for the following metrics (matches AE). Assumes results are in order: train, val, test."""
    # START HERE
    for metric in metrics:
        for filter_subgroup in ["all", "missingonly"]:
            args = [est, true]
            if filter_subgroup == "missingonly":
                args.append(missing_mask)
            val = metric["fn"](*args)
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
