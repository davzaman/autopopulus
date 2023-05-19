import re
from argparse import Namespace
from typing import Dict

import miceforest as mf
import pandas as pd
from joblib import dump

## Sklearn
# Required for IterativeImputer, as it's experimental
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

## Local Modules
from autopopulus.data import CommonDataModule
from autopopulus.data.transforms import SimpleImpute
from autopopulus.utils.impute_metrics import MAAPEMetric, universal_metric
from autopopulus.utils.log_utils import get_serialized_model_path
from autopopulus.models.sklearn_model_utils import TransformScorer, TunableEstimator

BASELINE_DATA_SETTINGS = {
    "scale": True,
    "feature_map": None,
    "uniform_prob": False,
}


BASELINE_IMPUTER_MODEL_PARAM_GRID = {
    "knn": {
        "n_neighbors": [3, 5, 10],
        "weights": ["uniform", "distance"],
    },
    "mice": {"max_iter": [10, 50], "n_nearest_features": [5, 10, None]},
}


def none(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    return {split: data.splits["data"][split] for split in ["train", "val", "test"]}


def simple(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    # nothing to tune
    imputer = SimpleImpute(data.dataset_loader.continuous_cols)
    imputer.fit(data.splits["data"]["train"])
    dump(imputer, get_serialized_model_path("simple"))
    return {  # impute data
        split: imputer.transform(data.splits["data"][split])
        for split in ["train", "val", "test"]
    }


def knn(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    imputer = TunableEstimator(
        KNNImputer(),
        BASELINE_IMPUTER_MODEL_PARAM_GRID["knn"],
        score_fn=TransformScorer(
            universal_metric(  # sync with tuning metric for ae
                MAAPEMetric(columnwise=True, nfeatures=data.nfeatures)
            ),
            higher_is_better=False,
        ),
    )
    imputer.fit(data.splits["data"])
    dump(imputer, get_serialized_model_path("knn"))
    return {
        # Add columns back in (sklearn erases) for rmse for missing only columns
        split: pd.DataFrame(
            # impute data
            imputer.transform(data.splits["data"][split]),
            columns=data.columns["original"],
        )
        for split in ["train", "val", "test"]
    }


def mice(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    """Uses sklearn instead of miceforest package."""
    imputer = TunableEstimator(
        IterativeImputer(random_state=args.seed),
        BASELINE_IMPUTER_MODEL_PARAM_GRID["mice"],
        score_fn=TransformScorer(
            universal_metric(  # sync with tuning metric for ae
                MAAPEMetric(columnwise=True, nfeatures=data.nfeatures)
            ),
            higher_is_better=False,
        ),
    )
    imputer.fit(data.splits["data"])
    dump(imputer, get_serialized_model_path("knn"))
    return {
        # Add columns back in (sklearn erases) for rmse for missing only columns
        split: pd.DataFrame(
            # impute data
            imputer.transform(data.splits["data"][split]),
            columns=data.columns["original"],
        )
        for split in ["train", "val", "test"]
    }


# TODO: this doesn't work with TunableEstimator
def miceforest(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    imputer = mf.KernelDataSet(
        data.splits["data"]["train"],
        save_all_iterations=True,
        random_state=args.seed,
    )
    imputer.mice(args.mice_num_iterations, verbose=args.verbose, n_jobs=args.njobs)
    X_train = imputer.complete_data()
    X_val = imputer.impute_new_data(data.splits["data"]["val"]).complete_data()
    X_test = imputer.impute_new_data(data.splits["data"]["test"]).complete_data()

    # Serialize Model
    dump(imputer, get_serialized_model_path("miceforest"))

    return {"train": X_train, "val": X_val, "test": X_test}
