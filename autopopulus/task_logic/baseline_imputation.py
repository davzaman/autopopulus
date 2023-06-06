from argparse import Namespace
from typing import Callable, Dict, List, Optional, Union
from pandas import DataFrame
import numpy as np
import pickle as pk
from os import makedirs
from os.path import dirname
from pyparsing import col
from torchmetrics import MetricCollection
from tqdm import tqdm
from numpy.random import default_rng
from pytorch_lightning.utilities import rank_zero_warn
from autopopulus.data.transforms import list_to_tensor

# Local
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    MIXED_FEATURE_METRIC_FORMAT,
    BasicLogger,
    get_serialized_model_path,
)
from autopopulus.task_logic import (
    baseline_static_imputation,
    baseline_longitudinal_imputation,
)
from autopopulus.data import CommonDataModule
from autopopulus.utils.impute_metrics import (
    CategoricalErrorMetric,
    MAAPEMetric,
    RMSEMetric,
    universal_metric,
)
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
            col_idxs_by_type=data.col_idxs_by_type["original"],
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
                "col_idxs_by_type": data.col_idxs_by_type["original"],
                "semi_observed_training": data.ground_truth_has_nans,
            },
            file,
        )


def get_baseline_metrics(col_idxs_by_type) -> List[Dict[str, Union[str, Callable]]]:
    ctn_metric_names = [("RMSE", RMSEMetric), ("MAAPE", MAAPEMetric)]
    cat_metric_names = [("CategoricalError", CategoricalErrorMetric)]
    return [
        {
            "name": MIXED_FEATURE_METRIC_FORMAT.format(
                ctn_name=ctn_name, cat_name=cat_name
            ),
            "fn": universal_metric(
                MetricCollection(
                    {
                        "continuous": ctn_metric(
                            ctn_cols_idx=list_to_tensor(col_idxs_by_type["continuous"]),
                            columnwise=reduction == "CW",
                        ),
                        "categorical": cat_metric(
                            list_to_tensor(col_idxs_by_type["binary"]),
                            list_to_tensor(col_idxs_by_type["onehot"]),
                            columnwise=reduction == "CW",
                        ),
                    },
                    compute_groups=False,
                )
            ),
            "reduction": reduction,
        }
        for reduction in ["CW", "EW"]
        for ctn_name, ctn_metric in ctn_metric_names
        for cat_name, cat_metric in cat_metric_names
    ]


def evaluate_baseline_imputation(
    args: Namespace,
    split: str,
    pred: DataT,
    input_data: DataT,
    true: DataT,
    col_idxs_by_type: Dict[str, List[str]],
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
    metrics: List[Dict[str, Union[str, Callable]]] = get_baseline_metrics(
        col_idxs_by_type
    )

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
            if filter_subgroup == "missingonly" and where_data_are_missing.any().any():
                val = metric["fn"](est, true, where_data_are_missing)
            else:
                val = metric["fn"](est, true)
            context = {
                "step": "impute",
                "split": split,
                "feature_space": "original",
                "filter_subgroup": filter_subgroup,
                "reduction": metric["reduction"],
            }
            # metriccollection returns a dict
            if isinstance(val, Dict):
                combined = sum(val.values())
                # log the summed components for the mixed feature types
                log.add_scalar(
                    combined,
                    metric["name"],
                    global_step=global_step,
                    context={**context, **{"feature_type": "mixed"}},
                    tb_name_format=IMPUTE_METRIC_TAG_FORMAT,
                )
                # separately log each component as its metric type
                for feature_type, type_val in val.items():
                    log.add_scalar(
                        type_val,
                        metric["name"],
                        global_step=global_step,
                        context={**context, **{"feature_type": feature_type}},
                        tb_name_format=IMPUTE_METRIC_TAG_FORMAT,
                    )
            else:
                log.add_scalar(
                    val,
                    metric["name"],
                    global_step=global_step,
                    context={**context, **{"feature_type": "mixed"}},
                    tb_name_format=IMPUTE_METRIC_TAG_FORMAT,
                )
