from abc import ABCMeta
from logging import error
from enum import Enum
from typing import Any, Callable, Dict, List, Union
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from torch import tensor

from autopopulus.data.dataset_classes import CommonDataModule
from autopopulus.utils.impute_metrics import MAAPEMetric, universal_metric
from autopopulus.utils.log_utils import IMPUTE_METRIC_TAG_FORMAT

# This should reflect everything in baseline_static_imputation
STATIC_BASELINE_METHODS = ["knn", "mice", "simple", "none"]
# Even if it cannot be tuned, so that I have a list of reference of what counts as baseline.
# TODO[LOW]: add one for longitudinal
STATIC_BASELINE_IMPUTER_MODEL_PARAM_GRID: Dict[ABCMeta, Dict[str, List[Any]]] = {
    KNNImputer: {
        "n_neighbors": [3, 5, 10],
        "weights": ["uniform", "distance"],
    },
    IterativeImputer: {"max_iter": [10, 50], "n_nearest_features": [5, 10, None]},
    SimpleImputer: {},
    None: {},
}

BASELINE_DATA_SETTINGS: Dict[str, Any] = {
    "scale": True,
    "feature_map": None,
    "uniform_prob": False,
}


AE_METHOD_SETTINGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "vanilla": {"train": {}},
    "dae": {"train": {"dropout_corruption": 0.3}},
    "batchswap": {"train": {"batchswap_corruption": 0.3}},
    "vae": {"train": {"variational": True}},
    "dvae": {"train": {"dropout_corruption": 0.3, "variational": True}},
    "ap_new": {
        "data": {
            "scale": True,
            "feature_map": "discretize_continuous",
            "uniform_prob": True,
        },
        "train": {
            "variational": False,
            "activation": "ReLU",
            "optimn": "Adam",
            "lossn": "BCE",
        },
    },
    # Gondara paper. Ref: https://github.com/Harry24k/MIDA-pytorch and https://gist.github.com/lgondara/18387c5f4d745673e9ca8e23f3d7ebd3
    # - Model is overcomplete + deep.
    # - uses h2o.deeplearning by default one-hot encodes categorical variables
    # - warm start with simple imputation
    "mida": {
        "data": {"scale": True, "feature_map": None, "uniform_prob": False},
        "train": {
            "lossn": "MSE",
            "optimn": "SGD",
            "activation": "Tanh",
            "variational": False,
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
    # WARNING: This is deprecated, mvec is deprecated.
    "dae_mvec": {
        "data": {
            "scale": True,
            "feature_map": None,
            "uniform_prob": False,
            # "ctn_columns": None,
        },
        "train": {
            "mvec": True,
            "variational": False,
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
    # - Originally the paper was only on continuous data so it was only MSE, but we will do CEMSE since we have categorical data too.
    "vae_ifac": {
        "data": {"scale": True, "feature_map": None, "uniform_prob": False},
        "train": {
            "variational": True,
            # "lossn": "MSE",
            "lossn": "CEMSE",
            "activation": "ReLU",
            "replace_nan_with": 0,
        },
    },
}


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
        return method in STATIC_BASELINE_METHODS


def get_tune_metric(
    imputer: ImputerT, data: CommonDataModule, data_feature_space: str
) -> Union[str, Callable]:
    """This must be called after data.setup"""
    if imputer == ImputerT.AE:  # tuner expects string
        if data.semi_observed_training:
            return IMPUTE_METRIC_TAG_FORMAT.format(
                name="loss",
                # loss in only in mapped space (if mapping)
                feature_space=data_feature_space,
                filter_subgroup="all",
                reduction="NA",
                split="val",
                feature_type="mixed",
            )
        else:
            return IMPUTE_METRIC_TAG_FORMAT.format(
                name="MAAPE",
                feature_space="original",
                filter_subgroup="missingonly",
                reduction="CW",
                split="val",
                feature_type="mixed",
            )
    elif imputer == ImputerT.BASELINE:
        if (
            data.semi_observed_training
        ):  # either heuristic or defaults, we go with defaults
            return None
        else:
            return universal_metric(  # sync with tuning metric for ae
                MAAPEMetric(
                    ctn_cols_idx=tensor(
                        data.col_idxs_by_type["original"]["continuous"]
                    ),
                    columnwise=True,
                )
            )
