from logging import error
from os import makedirs
import sys
from typing import Callable
from argparse import ArgumentParser, Namespace
import pickle as pk
from os.path import join, exists, dirname

#### Traceback ####
from rich.traceback import install
from sklearn.exceptions import DataConversionWarning
from rich import (
    print,
)  # https://rich.readthedocs.io/en/stable/markup.html#console-markup
import warnings


warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


from pytorch_lightning.utilities import rank_zero_info

#### Local Module ####
from autopopulus.utils.cli_arg_utils import (
    load_cli_args,
    str2bool,
)
from autopopulus.utils.log_utils import get_logdir, init_new_logger
from autopopulus.utils.utils import (
    get_module_function_names,
    seed_everything,
    should_ampute,
)
from autopopulus.datasets import DATA_LOADERS, CureCKDDataLoader, CrrtDataLoader
from autopopulus.data import CommonDataModule
from autopopulus.models.ap import AEImputer
from autopopulus.task_logic import (
    baseline_static_imputation,
    baseline_longitudinal_imputation,
    baseline_imputation,
)
from autopopulus.task_logic.ae_imputation import AE_METHOD_SETTINGS, ae_imputation_logic
from autopopulus.models.ae import AEDitto
from autopopulus.models.prediction_models import Predictor
from autopopulus.data.types import DataT  # Filter warnings

install(theme="solarized-dark")


def get_imputation_logic(args: Namespace) -> Callable[[Namespace, DataT], None]:
    # nothing done...just fully observed
    if args.method == "none":
        return baseline_static_imputation.fully_observed
    else:
        if args.method in AE_METHOD_SETTINGS:
            return ae_imputation_logic
        elif hasattr(baseline_static_imputation, args.method) or hasattr(
            baseline_longitudinal_imputation, args.method
        ):
            return baseline_imputation.baseline_imputation_logic
        else:
            error(f"Method passed ({args.method}) is not a supported method.")


def main():
    load_cli_args()
    args = init_cli_args()
    # if args.verbose:
    #     print(args)
    seed_everything(args.seed)

    init_new_logger()

    pickled_imputed_data_path = join("serialized_models", "imputed_data.pkl")
    if args.imputed_data_from_pickle and exists(pickled_imputed_data_path):
        print("Loading pickled imputed data...")
        with open(pickled_imputed_data_path, "rb") as file:
            imputed_data, labels = pk.load(file)
    else:
        # Assumes dataset name is kosher from argparse
        data_loader = DATA_LOADERS[args.dataset].from_argparse_args(args)
        data_settings = (
            AE_METHOD_SETTINGS[args.method].get("data", {})
            if args.method in AE_METHOD_SETTINGS
            else baseline_static_imputation.BASELINE_DATA_SETTINGS
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
        makedirs(dirname(pickled_imputed_data_path), exist_ok=True)
        with open(pickled_imputed_data_path, "wb") as file:
            pk.dump((imputed_data, labels), file)

    rank_zero_info(f"Beginning downstream prediction on {args.data_type_time_dim}")

    predictor = Predictor.from_argparse_args(
        args, logdir=get_logdir(args), data_type_time_dim=args.data_type_time_dim
    )
    predictor.fit(imputed_data, labels)


def init_cli_args() -> Namespace:
    p = ArgumentParser()
    #### GENERAL ####
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for randomization",
    )
    p.add_argument(
        "--runtest",
        type=str2bool,
        default=False,
        help="Whether or not to run on the test set.",
    )
    p.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="print information to shell",
    )

    #### DATASET LOADING ####
    p.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="cure_ckd",
        choices=["cure_ckd", "crrt"],
        help="which dataset to use",
    )

    if "cure_ckd" in sys.argv:
        p = CureCKDDataLoader.add_data_args(p)
    if "crrt" in sys.argv:
        p = CrrtDataLoader.add_data_args(p)
    p.add_argument(
        "--mimic-limit",
        type=int,
        required="mimic3" in sys.argv,
        default=10,
        help="Limits the number of stays to read from the mimic3 datset.",
    )

    #### DATA HANDLING ####
    p = CommonDataModule.add_data_args(p)
    p.add_argument(
        "--val-test-size",
        type=float,
        # required=True,
        default=0.40,
        help="What percent of the dataset should be set aside for validation and test.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        # required=True,
        default=0.50,
        help="What percent of the validation+test portion should be set aside for the test set.",
    )

    #### IMPUTER ####
    p = AEImputer.add_imputer_args(p)
    p = AEDitto.add_imputer_args(p)
    p.add_argument(
        "--ae-from-checkpoint",
        type=str,
        default=None,
        help="Path to serialized model to load ae from checkpoint instead of training a new one.",
    )
    p.add_argument(
        "--method",
        type=str,
        # required=True,
        default="simple",
        choices=list(AE_METHOD_SETTINGS.keys())
        + get_module_function_names(baseline_static_imputation)
        + get_module_function_names(baseline_longitudinal_imputation)
        + ["none"],
        help="Which imputer to use, fully_observed for no imputation (include the fully observed flag in this case).",
    )
    # For MICE
    p.add_argument(
        "--mice-num-iterations",
        type=int,
        required="mice" in sys.argv or "--method=miceforest" in sys.argv,
        default=50,
        help="When using the mice imputer, you need to set how many datsets to complete.",
    )  # sklearn
    p.add_argument(
        "--mice-njobs",
        type=int,
        required="miceforest" in sys.argv,
        default=32,
        help="When using miceforest for mice imputation, set the number of jobs for parallelization.",
    )

    #### PREDICTION ####
    p = Predictor.add_prediction_args(p)
    p.add_argument(
        "--imputed-data-from-pickle",
        type=str,
        default=None,
        help="Path to pickled dictionary of split to imputed dataframe.",
    )

    # return p.parse_args()
    # Ignore unrecognized args
    return p.parse_known_args()[0]


if __name__ == "__main__":
    main()
    print("Done!")
