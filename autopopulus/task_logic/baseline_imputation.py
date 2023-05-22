from argparse import Namespace
from typing import Dict, Optional
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
from models.sklearn_model_utils import TransformScorer


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
    evaluate_baseline_imputation(args, data, imputed_data_per_split)
    return imputed_data_per_split


def evaluate_baseline_imputation(
    args: Namespace,
    data: CommonDataModule,
    imputed_data_per_split: Dict[str, DataFrame],
):
    ## LOGGING ##
    if (
        args.method != "none"
    ):  # Logging here if baseline experiment (not fully observed)
        log = BasicLogger(
            args=args, experiment_name=args.experiment_name, verbose=args.verbose
        )
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
                "name": "MAAPE",
                "fn": universal_metric(
                    MAAPEMetric(columnwise=True, nfeatures=data.nfeatures["original"])
                ),
                "reduction": "CW",
            },
            {"name": "RMSE", "fn": universal_metric(RMSEMetric()), "reduction": "EW"},
            {"name": "MAAPE", "fn": universal_metric(MAAPEMetric()), "reduction": "EW"},
        ]
        for split, imputed_data in imputed_data_per_split.items():
            if imputed_data.isna().any().any():
                rank_zero_warn("NaNs still found in imputed data.")
            (
                pred,
                true,
                where_data_are_missing,
            ) = TransformScorer.get_imputed_data_from_model_output(
                X=data.splits["data"][split],
                X_pred=imputed_data,
                X_true=data.splits["ground_truth"][split],
                return_where_data_are_missing=True,
            )
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
            args.append(missing_mask if filter_subgroup == "missingonly" else None)
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
