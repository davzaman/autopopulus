from argparse import Namespace
from typing import Dict

from joblib import dump
import miceforest as mf
import pandas as pd

## Sklearn
# Required for IterativeImputer, as it's experimental
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

## Local Modules
from autopopulus.data.transforms import SimpleImpute
from autopopulus.data import CommonDataModule
from autopopulus.utils.log_utils import get_serialized_model_path


BASELINE_DATA_SETTINGS = {
    "scale": True,
    "feature_map": None,
    "uniform_prob": False,
}


def none(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    return {
        "train": data.splits["data"]["train"],
        "val": data.splits["data"]["val"],
        "test": data.splits["data"]["test"],
    }


def simple(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    imputer = SimpleImpute(data.dataset_loader.continuous_cols)
    X_train = imputer.fit_transform(data.splits["data"]["train"])
    X_val = imputer.transform(data.splits["data"]["val"])
    X_test = imputer.transform(data.splits["data"]["test"])

    dump(imputer, get_serialized_model_path("simple"))

    return {"train": X_train, "val": X_val, "test": X_test}


def knn(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    ## IMPUTE ##
    imputer = KNNImputer()
    X_train = imputer.fit_transform(data.splits["data"]["train"])
    X_val = imputer.transform(data.splits["data"]["val"])
    X_test = imputer.transform(data.splits["data"]["test"])

    # Add columns back in (sklearn erases) for rmse for missing only columns
    X_train = pd.DataFrame(X_train, columns=data.columns["original"])
    X_val = pd.DataFrame(X_val, columns=data.columns["original"])
    X_test = pd.DataFrame(X_test, columns=data.columns["original"])

    dump(imputer, get_serialized_model_path("knn"))

    return {"train": X_train, "val": X_val, "test": X_test}


def mice(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    """Uses sklearn instead of miceforest package."""
    imputer = IterativeImputer(
        max_iter=args.mice_num_iterations, random_state=args.seed
    )
    X_train = imputer.fit_transform(data.splits["data"]["train"])
    X_val = imputer.transform(data.splits["data"]["val"])
    X_test = imputer.transform(data.splits["data"]["test"])

    # Add columns back in (sklearn erases) for rmse for missing only columns
    X_train = pd.DataFrame(X_train, columns=data.columns["original"])
    X_val = pd.DataFrame(X_val, columns=data.columns["original"])
    X_test = pd.DataFrame(X_test, columns=data.columns["original"])

    # Serialize Model
    dump(imputer, get_serialized_model_path("mice"))

    return {"train": X_train, "val": X_val, "test": X_test}


def miceforest(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    ## IMPUTE ##
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
