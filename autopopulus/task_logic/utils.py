from logging import error
from enum import Enum
from typing import Callable, Union

from autopopulus.task_logic import (
    baseline_longitudinal_imputation,
    baseline_static_imputation,
)
from autopopulus.task_logic.ae_imputation import AE_METHOD_SETTINGS
from autopopulus.data.dataset_classes import CommonDataModule
from autopopulus.utils.impute_metrics import MAAPEMetric, universal_metric
from autopopulus.utils.log_utils import IMPUTE_METRIC_TAG_FORMAT


class ImputerT(Enum):
    AE = "ae"
    BASELINE = "baseline"

    def type(method: str) -> "ImputerT":
        if ImputerT.is_ae(method):
            return ImputerT.AE
        elif ImputerT.is_baseline(method):
            return ImputerT.BASELINE
        else:
            error(f"Method passed ({method}) is not a supported method.")

    @staticmethod
    def is_ae(method: str) -> bool:
        return method in AE_METHOD_SETTINGS

    @staticmethod
    def is_baseline(method: str) -> bool:
        return hasattr(baseline_static_imputation, method) or hasattr(
            baseline_longitudinal_imputation, method
        )


def get_tune_metric(
    imputer: ImputerT, data: CommonDataModule, data_feature_space: str
) -> Union[str, Callable]:
    """This must be called after data.setup"""
    if imputer == ImputerT.AE:  # tuner expects string
        if data.ground_truth_has_nans:
            return IMPUTE_METRIC_TAG_FORMAT.format(
                name="loss",
                # loss in only in mapped space (if mapping)
                feature_space=data_feature_space,
                filter_subgroup="all",
                reduction="NA",
                split="val",
            )
        else:
            return IMPUTE_METRIC_TAG_FORMAT.format(
                name="MAAPE",
                feature_space="original",
                filter_subgroup="missingonly",
                reduction="CW",
                split="val",
            )
    elif imputer == ImputerT.BASELINE:
        if (
            data.ground_truth_has_nans
        ):  # either heuristic or defaults, we go with defaults
            return None
        else:
            return universal_metric(  # sync with tuning metric for ae
                MAAPEMetric(columnwise=True, nfeatures=data.nfeatures["original"])
            )
