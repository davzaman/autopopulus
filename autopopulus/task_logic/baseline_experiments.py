from argparse import Namespace

from joblib import dump
import miceforest as mf
import pandas as pd

## Sklearn
# Required for IterativeImputer, as it's experimental
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

## Local Modules
from data.transforms import simple_impute
from data.utils import CommonDataModule, get_ctn_cols, load_data
from utils.log_utils import (
    add_all_text,
    get_logger,
    log_imputation_performance,
    get_serialized_model_path,
)
from task_logic.predictions import run_predictions
from utils.utils import should_ampute


def fully_observed(args: Namespace) -> None:
    X, y = load_data(args)
    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=False,
    )
    # will create train/val/test
    data.setup()
    feature_names = list(data.X_train.columns)

    run_predictions(
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        data.X_test,
        data.y_test,
        args,
        feature_names,
    )


def simple_imputation(args: Namespace) -> None:
    X, y = load_data(args)
    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=should_ampute(args),
    )
    data.setup()
    feature_names = list(data.X_train.columns)

    ctn_columns = get_ctn_cols(args.dataset)
    X_train, X_val, X_test = simple_impute(
        data.X_train, ctn_columns, [data.X_val, data.X_test]
    )

    ## LOGGING ##
    log = get_logger(args)
    if args.fully_observed and log:
        log_imputation_performance([X_train, X_val, X_test], data, log, args.runtest)
        log.close()

    run_predictions(
        X_train,
        data.y_train,
        X_val,
        data.y_val,
        X_test,
        data.y_test,
        args,
        feature_names,
    )


def knn_imputation(args: Namespace) -> None:
    X, y = load_data(args)
    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=should_ampute(args),
    )
    data.setup()
    feature_names = list(data.X_train.columns)

    ## IMPUTE ##
    knn_imputer = KNNImputer()
    imputer = knn_imputer.fit(data.X_train)
    X_train = imputer.transform(data.X_train)
    X_val = imputer.transform(data.X_val)
    X_test = imputer.transform(data.X_test)

    # Add columns back in (sklearn erases) for rmse for missing only columns
    X_val = pd.DataFrame(X_val, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    ## LOGGING ##
    log = get_logger(args)
    if args.fully_observed and log:
        log_imputation_performance([X_train, X_val, X_test], data, log, args.runtest)
        log.close()

    run_predictions(
        X_train,
        data.y_train,
        X_val,
        data.y_val,
        X_test,
        data.y_test,
        args,
        feature_names,
    )


def mice_imputation(args: Namespace) -> None:
    """Uses sklearn instead of miceforest package."""
    X, y = load_data(args)
    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=should_ampute(args),
    )
    data.setup()
    feature_names = list(data.X_train.columns)

    ## IMPUTE ##
    imputer = IterativeImputer(
        max_iter=args.num_mice_iterations, random_state=args.seed
    )
    X_train = imputer.fit_transform(data.X_train)
    X_val = imputer.transform(data.X_val)
    X_test = imputer.transform(data.X_test)

    # Add columns back in (sklearn erases) for rmse for missing only columns
    X_val = pd.DataFrame(X_val, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    ## LOGGING ##
    # Serialize Model
    dump(imputer, get_serialized_model_path("mice"))

    log = get_logger(args)
    if args.fully_observed and log:
        log_imputation_performance([X_train, X_val, X_test], data, log, args.runtest)
        log.close()

    run_predictions(
        X_train,
        data.y_train,
        X_val,
        data.y_val,
        X_test,
        data.y_test,
        args,
        feature_names,
    )


def miceforest_imputation(args: Namespace) -> None:
    X, y = load_data(args)
    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=should_ampute(args),
    )
    data.setup()
    feature_names = list(data.X_train.columns)

    ## IMPUTE ##
    imputer = mf.KernelDataSet(
        data.X_train, save_all_iterations=True, random_state=args.seed
    )
    imputer.mice(args.num_mice_iterations, verbose=args.verbose, n_jobs=args.njobs)
    X_train = imputer.complete_data()
    X_val = imputer.impute_new_data(data.X_val).complete_data()
    X_test = imputer.impute_new_data(data.X_test).complete_data()

    ## LOGGING ##
    # Serialize Model
    dump(imputer, get_serialized_model_path("miceforest"))

    log = get_logger(args)
    if args.fully_observed and log:
        log_imputation_performance([X_train, X_val, X_test], data, log, args.runtest)
        add_all_text(log, {"mice_kernel": str(imputer)})
        log.close()

    run_predictions(
        X_train,
        data.y_train,
        X_val,
        data.y_val,
        X_test,
        data.y_test,
        args,
        feature_names,
    )
