from argparse import Namespace

## Local Modules
from data.utils import (
    CommonDataModule,
    get_ctn_cols,
    get_onehot_prefix_names,
    load_data,
)
from task_logic.predictions import run_predictions
from utils.impute_metrics import AccuracyPerBin
from utils.utils import should_ampute

from utils.tuner import create_autoencoder_with_tuning


def ap(args: Namespace):
    X, y = load_data(args)
    # n_features = len(X.columns)
    feature_names = list(X.columns)

    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        ampute=should_ampute(args),
        scale=True,
        discretize=True,
        uniform_prob=True,
        ctn_columns=get_ctn_cols(args.dataset),
        onehot_prefix_names=get_onehot_prefix_names(args.dataset),
    )
    # will create train/val/test
    data.setup()

    settings = {
        "mvec": False,
        "vae": False,
        "activation": "ReLU",
        "optimn": "Adam",
        "lossn": "BCE",
        "undiscretize_data": True,
        "n_features": len(data.columns_disc),
    }

    ae_imputer = create_autoencoder_with_tuning(args, data, settings)

    X_train = ae_imputer.transform(data.X_train, data.X_train_undisc)
    X_val = ae_imputer.transform(data.X_val, data.X_val_undisc)
    X_test = ae_imputer.transform(data.X_test, data.X_test_undisc)

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


def mida(args: Namespace) -> None:
    """Gondara paper. Ref: https://github.com/Harry24k/MIDA-pytorch and https://gist.github.com/lgondara/18387c5f4d745673e9ca8e23f3d7ebd3
    - Model is overcomplete + deep.
    - uses h2o.deeplearning by default one-hot encodes categorical variables
    - warm start with simple imputation
    """
    X, y = load_data(args)
    # n_features = len(X.columns)
    feature_names = list(X.columns)

    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        ampute=should_ampute(args),
        scale=True,
        discretize=False,
        uniform_prob=False,
        ctn_columns=get_ctn_cols(args.dataset),
        onehot_prefix_names=get_onehot_prefix_names(args.dataset),
    )
    # will create train/val/test
    data.setup()

    settings = {
        "lossn": "MSE",
        "optimn": "SGD",
        "activation": "TanH",
        "mvec": False,
        "vae": False,
        "dropout_corruption": 0.5,
        "replace_nan_with": "simple",
        "n_features": len(data.columns),
    }

    ae_imputer = create_autoencoder_with_tuning(args, data, settings)

    X_train = ae_imputer.transform(data.X_train)
    X_val = ae_imputer.transform(data.X_val)
    X_test = ae_imputer.transform(data.X_test)

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


def dae_mvec(args: Namespace):
    """Beaulieu-Jones paper.
    - They only had one layer so stuff like dropout doesn't make sense.
    - Repeated or temporal measurements were encoded as the
    mean, minimum, maximum, count, standard deviation and slope
    across each repeat.
    - Missing values are turned into 0.
    - Categorical vars are one-hot encoded.
    - Everything including continuous vars are sigmoided at the end.
    Ref: https://github.com/greenelab/DAPS/"""
    X, y = load_data(args)
    feature_names = list(X.columns)

    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=should_ampute(args),
        ctn_columns=None,  # Force sigmoid on all columns at the end
        onehot_prefix_names=get_onehot_prefix_names(args.dataset),
    )
    # will create train/val/test
    data.setup()

    settings = {
        "mvec": True,
        "vae": False,
        "dropout_corruption": 0.2,
        "replace_nan_with": 0,
        "lossn": "BCE",
        "optimn": "SGD",
        "activation": "sigmoid",
        "n_features": len(data.columns),
    }

    ae_imputer = create_autoencoder_with_tuning(args, data, settings)

    X_train = ae_imputer.transform(data.X_train)
    X_val = ae_imputer.transform(data.X_val)
    X_test = ae_imputer.transform(data.X_test)

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


def vae_ifac(args: Namespace):
    """McCoy paper. Ref: https://github.com/ProcessMonitoringStellenboschUniversity/IFAC-VAE-Imputation
    - They report RMSE on just missing columns according to the code.
    - AE is not over/undercomplete. All the hidden layers are the same size.
    - All data is continuous.
    - Reconstruction loss of ELBO is equivalent to MSE since we're assuming Normal dist.
    - Their paper reports replacing missing data with a single random value but they really just replace with 0.
    - Originally the paper was only on continuous data so it was only MSE, but we will do BCEMSE since we have categorical data too.
    """
    X, y = load_data(args)
    feature_names = list(X.columns)

    data = CommonDataModule.from_argparse_args(
        args,
        X=X,
        y=y,
        scale=True,
        discretize=False,
        uniform_prob=False,
        ampute=should_ampute(args),
        ctn_columns=get_ctn_cols(args.dataset),
        onehot_prefix_names=get_onehot_prefix_names(args.dataset),
    )
    # will create train/val/test
    data.setup()

    settings = {
        "vae": True,
        # "lossn": "MSE",
        "lossn": "BCEMSE",
        "activation": "ReLU",
        "replace_nan_with": 0,
        "n_features": len(data.columns),
    }

    ae_imputer = create_autoencoder_with_tuning(args, data, settings)

    X_train = ae_imputer.transform(data.X_train)
    X_val = ae_imputer.transform(data.X_val)
    X_test = ae_imputer.transform(data.X_test)

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
