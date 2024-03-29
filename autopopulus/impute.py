from argparse import Namespace
import sys
from typing import Callable

#### Traceback ####
from rich.traceback import install

install(theme="solarized-dark")

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


from autopopulus.data import CommonDataModule
from autopopulus.data.types import DataT  # Filter warnings
from autopopulus.datasets import DATA_LOADERS
from autopopulus.task_logic import baseline_imputation
from autopopulus.task_logic.ae_imputation import ae_imputation_logic
from autopopulus.task_logic.utils import (
    AE_METHOD_SETTINGS,
    BASELINE_DATA_SETTINGS,
    ImputerT,
)
from autopopulus.utils.get_set_cli_args import init_cli_args, load_cli_args
from autopopulus.utils.log_utils import (
    dump_artifact,
    init_sys_logger,
    mlflow_end,
    mlflow_init,
)
from autopopulus.utils.utils import rank_zero_print, seed_everything


def get_imputation_logic(args: Namespace) -> Callable[[Namespace, DataT], None]:
    imputer_type = ImputerT.type(args.method)
    if imputer_type == ImputerT.AE:
        return ae_imputation_logic
    elif imputer_type == ImputerT.BASELINE:
        return baseline_imputation.baseline_imputation_logic


def main():
    orig_command = sys.argv.copy()
    load_cli_args()
    args = init_cli_args()
    setattr(args, "orig_command", orig_command)
    # if args.verbose:
    #     rank_zero_print(args)
    seed_everything(args.seed)

    init_sys_logger()
    mlflow_init(args)

    # Assumes dataset name is kosher from argparse
    data_loader = DATA_LOADERS[args.dataset].from_argparse_args(args)
    data_settings = (
        AE_METHOD_SETTINGS[args.method].get("data", {})
        if args.method in AE_METHOD_SETTINGS
        else BASELINE_DATA_SETTINGS
    )
    # # To turn off transforms for debugging
    # data_settings = {
    #     "scale": False,
    #     "feature_map": None,
    #     "uniform_prob": False,
    # }
    ncpu_per_gpu = args.total_cpus_on_machine // args.total_gpus_on_machine
    num_workers = ncpu_per_gpu - 1
    data = CommonDataModule.from_argparse_args(
        args, dataset_loader=data_loader, **data_settings, num_workers=num_workers
    )

    imputed_data = get_imputation_logic(args)(args, data)
    labels = data.splits["label"]
    dump_artifact((imputed_data, labels), "imputed_data", "pkl")
    mlflow_end()


if __name__ == "__main__":
    main()
    rank_zero_print("Done!")
