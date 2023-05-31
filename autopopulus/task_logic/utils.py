from logging import error
from enum import Enum

from autopopulus.task_logic import (
    baseline_longitudinal_imputation,
    baseline_static_imputation,
)
from autopopulus.task_logic.ae_imputation import AE_METHOD_SETTINGS


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
