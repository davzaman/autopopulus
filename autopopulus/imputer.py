"""
Main driver of imputation experiments.
"""
import warnings  # Filter warnings
from argparse import ArgumentParser, Namespace
import sys
import numpy as np

#### Traceback ####
from rich.traceback import install
from sklearn.exceptions import DataConversionWarning

# https://rich.readthedocs.io/en/stable/markup.html#console-markup
from rich import print

#### Local Module ####
from models.ap import AEImputer
from data.utils import CommonDataModule
from utils.utils import YAMLStringListToList, seed_everything, str2bool
from task_logic.ae_experiments import (
    ap,
    dae_mvec,
    mida,
    vae_ifac,
)
from task_logic.baseline_experiments import (
    fully_observed,
    knn_imputation,
    mice_imputation,
    miceforest_imputation,
    simple_imputation,
)

install(theme="solarized-dark")
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

ae_methods = ["ap_new", "mida", "dae_mvec", "vae_ifac"]


def main():
    # Load args from guild
    args = init_args()
    if args.verbose:
        print(args)
    seed_everything(args.seed)
    run_experiment(args)


def run_experiment(args: Namespace):
    # nothing done...just fully observed
    if args.method == "none":
        fully_observed(args)
    # baseline imputer
    elif args.method == "simple":
        simple_imputation(args)
    elif args.method == "knn":
        knn_imputation(args)
    elif args.method == "mice":
        mice_imputation(args)
    elif args.method == "miceforest":
        miceforest_imputation(args)
    # AE methods
    elif args.method == "ap_new":
        ap(args)
    elif args.method == "mida":
        mida(args)
    elif args.method == "dae_mvec":
        dae_mvec(args)
    elif args.method == "vae_ifac":
        vae_ifac(args)
    else:
        assert False, args.method


def init_args() -> Namespace:

    p = ArgumentParser()
    # common
    p.add_argument(
        "--method",
        type=str,
        # required=True,
        default="simple",
        choices=["none", "simple", "knn", "mice", "miceforest"] + ae_methods,
        help="Which imputer to use, none for no imputation (include the fully observed flag in this case).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for randomization",
    )
    p.add_argument(
        "--tbX-on",
        type=str2bool,
        default=True,
        help="log to tensorboard",
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
        "--predictors",
        default="lr",
        action=YAMLStringListToList(choices=["lr", "rf", "dnn", "xgb"]),
        help="which ml classifiers to use for downstream prediction",
    )
    ## Predictive flags
    p.add_argument(
        "--num-bootstraps",
        type=int,
        # required=True,
        default=100,
        help="We do the predictive steps num_boostraps number of times to create a confidence interval on the metrics on the prediction performance.",
    )
    p.add_argument(
        "--confidence-level",
        type=float,
        # required=True,
        default=0.95,
        help="For the bootstrap confidence intervals, what is the confidence level we want to use.",
    )
    ## Impute flags
    p.add_argument(
        "--fully-observed",
        action="store_true",
        required=False,
        help="Filter down to fully observed dataset flag [TOGGLE].",
    )  # acts as switch
    p.add_argument(
        "--percent-missing",
        type=float,
        required="--fully-observed" in sys.argv and "--method=none" not in sys.argv,
        default=0.33,
        help="When filtering down to fully observed and amputing (imputer is not none), what percent of data should be missing.",
    )
    p.add_argument(
        "--missingness-mechanism",
        type=str,
        required="--fully-observed" in sys.argv and "--method=none" not in sys.argv,
        default="MCAR",
        choices=["MCAR", "MAR", "MNAR"],
        help="When filtering down to fully observed and amputing (imputer is not none), what missingness mechanism to use for amputation.",
    )
    p.add_argument(
        "--missing-cols",
        type=str,
        required="--fully-observed" in sys.argv and "--method=none" not in sys.argv,
        action=YAMLStringListToList(),
        help="Comma separated (with no spaces) list of columns in the dataset that will be masked when amputing.",
    )
    p.add_argument(
        "--observed-cols",
        type=str,
        required="--missingness-mechanism=MAR" in sys.argv,
        action=YAMLStringListToList(),
        help="Comma separated (with no spaces) list of columns in the dataset to use for masking when amputing under MAR.",
    )
    ## Dataset flags
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
    p.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="cure_ckd",
        choices=["cure_ckd", "covid_ckd", "mimic3", "synth"],
        help="which dataset to use",
    )
    # CKD
    p.add_argument(
        "--cohort",
        type=str,
        required="--dataset=cure_ckd" in sys.argv,
        default="ckd",
        choices=["ckd", "atrisk", "ckd_atrisk"],
        help="For CKD dataset: which cohort of patients by ckd status.",
    )
    p.add_argument(
        "--site-source",
        type=str,
        required="--dataset=cure_ckd" in sys.argv,
        default="ucla_providence",
        choices=["ucla", "providence", "ucla_providence"],
        help="For CKD datset: which hospital the data comes from.",
    )
    p.add_argument(
        "--covid-site-source",
        type=str,
        required="--dataset=covid_ckd" in sys.argv,
        default="ucla_providence",
        choices=["ucla", "providence", "ucla_providence"],
        help="For COVID CKD datset: which hospital the data comes from.",
    )
    p.add_argument(
        "--target",
        type=str,
        required="--dataset=cure_ckd" in sys.argv,
        default="rapid_decline_base_to_2",
        help="what is the target variable (y)",
    )
    # MIMIC
    p.add_argument(
        "--mimic-limit",
        type=int,
        required="--dataset=mimic3" in sys.argv,
        default=10,
        help="Limits the number of stays to read from the mimic3 datset.",
    )
    # For MICE
    p.add_argument(
        "--num-mice-iterations",
        type=int,
        required="--method=mice" in sys.argv or "--method=miceforest" in sys.argv,
        default=50,
        help="When using the mice imputer, you need to set how many datsets to complete.",
    )  # sklearn
    p.add_argument(
        "--njobs",
        type=int,
        required="--method=miceforest" in sys.argv,
        default=32,
        help="When using miceforest for mice imputation, set the number of jobs for parallelization.",
    )

    p = CommonDataModule.add_data_args(p)

    if np.array([ae_method in sys.argv for ae_method in ae_methods]).any():
        p = AEImputer.add_imputer_args(p)

    # return p.parse_args()
    # Ignore unrecognized args
    return p.parse_known_args()[0]


if __name__ == "__main__":
    main()
