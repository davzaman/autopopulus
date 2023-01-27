from argparse import Namespace
from typing import Dict
from pandas import DataFrame

# Local
from autopopulus.utils.log_utils import (
    get_logdir,
    get_summarywriter,
    log_imputation_performance,
)
from autopopulus.task_logic import (
    baseline_static_imputation,
    baseline_longitudinal_imputation,
)
from autopopulus.data import CommonDataModule


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
        log = get_summarywriter(get_logdir(args))
        if args.fully_observed and log:
            log_imputation_performance(imputed_data, data, log)
            log.close()

    return imputed_data
