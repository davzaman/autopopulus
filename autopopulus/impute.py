import pickle as pk
from argparse import Namespace
from os import makedirs
from os.path import dirname, join
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
from autopopulus.utils.log_utils import init_sys_logger
from autopopulus.utils.utils import rank_zero_print, seed_everything, should_ampute


def get_imputation_logic(args: Namespace) -> Callable[[Namespace, DataT], None]:
    imputer_type = ImputerT.type(args.method)
    if imputer_type == ImputerT.AE:
        return ae_imputation_logic
    elif imputer_type == ImputerT.BASELINE:
        return baseline_imputation.baseline_imputation_logic


def main():
    load_cli_args()
    args = init_cli_args()
    # if args.verbose:
    #     rank_zero_print(args)
    seed_everything(args.seed)

    init_sys_logger()

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

    data = CommonDataModule.from_argparse_args(
        args,
        dataset_loader=data_loader,
        ampute=should_ampute(args),
        **data_settings,
    )

    imputed_data = get_imputation_logic(args)(args, data)
    labels = data.splits["label"]
    pickled_imputed_data_path = join("serialized_models", "imputed_data.pkl")
    makedirs(dirname(pickled_imputed_data_path), exist_ok=True)
    with open(pickled_imputed_data_path, "wb") as file:
        pk.dump((imputed_data, labels), file)


if __name__ == "__main__":
    main()
    rank_zero_print("Done!")
