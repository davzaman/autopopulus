from argparse import Namespace
from typing import Dict
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning.utilities import rank_zero_info

## Local Modules
from autopopulus.data import CommonDataModule
from autopopulus.utils.tuner import create_autoencoder_with_tuning
from autopopulus.models.ap import AEImputer


AE_METHOD_SETTINGS = {
    "ap_new": {
        "data": {"scale": True, "discretize": True, "uniform_prob": True},
        "train": {
            "mvec": False,
            "vae": False,
            "activation": "ReLU",
            "optimn": "Adam",
            "lossn": "BCE",
            "undiscretize_data": True,
        },
    },
    # Gondara paper. Ref: https://github.com/Harry24k/MIDA-pytorch and https://gist.github.com/lgondara/18387c5f4d745673e9ca8e23f3d7ebd3
    # - Model is overcomplete + deep.
    # - uses h2o.deeplearning by default one-hot encodes categorical variables
    # - warm start with simple imputation
    "mida": {
        "data": {"scale": True, "discretize": False, "uniform_prob": False},
        "train": {
            "lossn": "MSE",
            "optimn": "SGD",
            "activation": "Tanh",
            "mvec": False,
            "vae": False,
            "dropout_corruption": 0.5,
            "replace_nan_with": "simple",
        },
    },
    # Beaulieu-Jones paper.
    # - They only had one layer so stuff like dropout doesn't make sense.
    # - Repeated or temporal measurements were encoded as the
    # mean, minimum, maximum, count, standard deviation and slope
    # across each repeat.
    # - Missing values are turned into 0.
    # - Categorical vars are one-hot encoded.
    # - Everything including continuous vars are sigmoided at the end.
    # Ref: https://github.com/greenelab/DAPS/
    "dae_mvec": {
        "data": {
            "scale": True,
            "discretize": False,
            "uniform_prob": False,
            # "ctn_columns": None,
        },
        "train": {
            "mvec": True,
            "vae": False,
            "dropout_corruption": 0.2,
            "replace_nan_with": 0,
            "lossn": "BCE",
            "optimn": "SGD",
            "activation": "Sigmoid",
        },
    },
    # McCoy paper. Ref: https://github.com/ProcessMonitoringStellenboschUniversity/IFAC-VAE-Imputation
    # - They report RMSE on just missing columns according to the code.
    # - AE is not over/undercomplete. All the hidden layers are the same size.
    # - All data is continuous.
    # - Reconstruction loss of ELBO is equivalent to MSE since we're assuming Normal dist.
    # - Their paper reports replacing missing data with a single random value but they really just replace with 0.
    # - Originally the paper was only on continuous data so it was only MSE, but we will do BCEMSE since we have categorical data too.
    "vae_ifac": {
        "data": {"scale": True, "discretize": False, "uniform_prob": False},
        "train": {
            "vae": True,
            # "lossn": "MSE",
            "lossn": "BCEMSE",
            "activation": "ReLU",
            "replace_nan_with": 0,
        },
    },
}


def ae_transform(
    data_module: CommonDataModule, ae_imputer: AEImputer, split_name: str
) -> ndarray:
    split_dataloader = getattr(data_module, f"{split_name}_dataloader")
    return ae_imputer.transform(split_dataloader())


def ae_imputation_logic(
    args: Namespace, data: CommonDataModule
) -> Dict[str, Dict[str, DataFrame]]:
    """Output: top-level lookup: static/long. second-level: train/val/test."""
    # combine two dicts python 3.5+
    settings = AE_METHOD_SETTINGS[args.method]["train"]

    if args.ae_from_checkpoint:
        rank_zero_info(f"Loading AEImputer from {args.ae_from_checkpoint}")
        ae_imputer = AEImputer.from_checkpoint(args)
        data.setup("fit")
    else:
        ae_imputer = create_autoencoder_with_tuning(args, data, settings)

    transformed = {
        split_name: ae_transform(data, ae_imputer, split_name)
        for split_name in ["train", "val", "test"]
    }

    return transformed
