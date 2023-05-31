from argparse import Namespace
from typing import Callable, Dict, List, Optional, Union
from pandas import DataFrame
import numpy as np
import pickle as pk
from os import makedirs
from os.path import dirname
from tqdm import tqdm
from numpy.random import default_rng
from pytorch_lightning.utilities import rank_zero_warn

# Local
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    BasicLogger,
    get_serialized_model_path,
)
from autopopulus.task_logic import (
    baseline_static_imputation,
    baseline_longitudinal_imputation,
)
from autopopulus.data import CommonDataModule
from autopopulus.utils.impute_metrics import MAAPEMetric, RMSEMetric, universal_metric
from autopopulus.data.types import DataT
from autopopulus.utils.utils import resample_indices_only


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
        [
            imputed_data.isna().any().any()
            for imputed_data in imputed_data_per_split.values()
        ]
    ):
        rank_zero_warn("NaNs still found in imputed data.")
    save_test_data(args, data)
    for split in ["train", "val"]:  # Test will be run in evaluate.py.
        evaluate_baseline_imputation(
            args,
            split=split,
            pred=imputed_data_per_split[split],
            input_data=data.splits["data"][split],
            true=data.splits["ground_truth"][split],
            nfeatures=data.nfeatures["original"],
            semi_observed_training=data.ground_truth_has_nans,
        )

    return imputed_data_per_split


def save_test_data(args: Namespace, data: CommonDataModule):
    """
    Save everything we need from datamodule so we don't have to save the whole thing.
    Reference evaluate_baseline_imputation() to see what we need to add here.
    Same in evaluate.py.
    Save the test split (same as AE saving test dataloader) + aux info.
    """
    test_dataloader_path = get_serialized_model_path(
        f"{args.data_type_time_dim.name}_test_dataloader", "pt"
    )
    makedirs(dirname(test_dataloader_path), exist_ok=True)
    with open(test_dataloader_path, "wb") as file:
        pk.dump(
            {
                "data": data.splits["data"]["test"],
                "ground_truth": data.splits["ground_truth"]["test"],
                "nfeatures": data.nfeatures["original"],
                "semi_observed_training": data.ground_truth_has_nans,
            },
            file,
        )


def get_baseline_metrics(nfeatures: int) -> List[Dict[str, Union[str, Callable]]]:
    return [
        {
            "name": "RMSE",
            "fn": universal_metric(RMSEMetric(columnwise=True, nfeatures=nfeatures)),
            "reduction": "CW",
        },
        {
            "name": "MAAPE",
            "fn": universal_metric(MAAPEMetric(columnwise=True, nfeatures=nfeatures)),
            "reduction": "CW",
        },
        {"name": "RMSE", "fn": universal_metric(RMSEMetric()), "reduction": "EW"},
        {"name": "MAAPE", "fn": universal_metric(MAAPEMetric()), "reduction": "EW"},
    ]


def evaluate_baseline_imputation(
    args: Namespace,
    split: str,
    pred: DataT,
    input_data: DataT,
    true: DataT,
    nfeatures: int,
    semi_observed_training: bool,
    bootstrap: bool = False,
):
    """This mirrors AEDitto.metric_logging_step."""
    if args.method == "none":  # no performance to log
        return
    if semi_observed_training:  # if data.ground_truth_has_nans
        return

    log: BasicLogger = BasicLogger(
        args=args, experiment_name=args.experiment_name, verbose=args.verbose
    )
    metrics: List[Dict[str, Union[str, Callable]]] = get_baseline_metrics(nfeatures)

    where_data_are_missing = np.isnan(input_data)
    if bootstrap:
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
