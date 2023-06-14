import sys
from argparse import ArgumentParser, Namespace
from os.path import isfile

import yaml

from autopopulus.data.dataset_classes import CommonDataModule
from autopopulus.datasets.ckd import CureCKDDataLoader
from autopopulus.datasets.crrt import CrrtDataLoader
from autopopulus.models.ae import AEDitto
from autopopulus.models.ap import AEImputer
from autopopulus.models.prediction_models import Predictor
from autopopulus.task_logic import (
    baseline_longitudinal_imputation,
    baseline_static_imputation,
)
from autopopulus.task_logic.utils import AE_METHOD_SETTINGS
from autopopulus.utils.cli_arg_utils import str2bool
from autopopulus.utils.utils import get_module_function_names


def load_cli_args(args_options_path: str = "options.yml"):
    """
    Modify command line args if desired, or load from YAML file.
    """
    if isfile(args_options_path):  # if file exists
        with open(args_options_path, "r") as f:
            res = yaml.safe_load(f)

        # sys.argv = [sys.argv[0]]
        for k, v in res.items():
            sys.argv += [f"--{k}", str(v)]


# This has to be in a separate script from `cli_arg_utils.py` because otherwise we get circular imports.
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
    p.add_argument(
        "--experiment-name",
        type=str,
        default="myexperiment",
        help="When running tuning, what experiment name to set. The guild file also shares this name.",
    )
    p.add_argument(
        "--aim-hash",
        type=str,
        default=None,
        help="Hash to continue the same Aim Run across impute/predict scripts/tasks.",
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
        # if this class' args are in the yml file/CLI then it will be a part of parse_known_args()[1] (it's unknown)
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
        "--test-size",
        type=float,
        # required=True,
        default=0.40,
        help="What percent of the dataset should be set aside for validation and test.",
    )
    p.add_argument(
        "--val-size",
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
        + get_module_function_names(baseline_longitudinal_imputation),
        help="Which imputer to use, fully_observed for no imputation (include the fully observed flag in this case).",
    )
    p.add_argument(
        "--bootstrap-evaluate-imputer",
        type=bool,
        default=False,
        help="Do bootstrap sampling to produce a distribution of metrics on the test set for autoencoder imputers.",
    )

    #### PREDICTION ####
    p = Predictor.add_prediction_args(p)

    # return p.parse_args()
    # Ignore unrecognized args
    return p.parse_known_args()[0]
