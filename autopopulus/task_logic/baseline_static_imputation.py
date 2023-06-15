import re
from argparse import Namespace
from typing import Dict

import miceforest as mf
import pandas as pd
from cloudpickle import dump

## Sklearn
# Required for IterativeImputer, as it's experimental
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeClassifier

## Local Modules
from autopopulus.data import CommonDataModule
from autopopulus.models.sklearn_model_utils import MixedFeatureImputer
from autopopulus.utils.log_utils import get_serialized_model_path
from autopopulus.models.sklearn_model_utils import TransformScorer, TunableEstimator
from autopopulus.task_logic.utils import (
    ImputerT,
    get_tune_metric,
)


def none(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    return {split: data.splits["data"][split] for split in ["train", "val", "test"]}


def simple(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    # nothing to tune
    imputer = MixedFeatureImputer(
        ctn_cols=data.dataset_loader.continuous_cols,
        onehot_groupby=data.groupby["original"]["categorical_onehots"],
    )
    imputer.fit(data.splits["data"]["train"])
    with open(get_serialized_model_path("simple"), "wb") as f:
        dump(imputer, f)
    return {  # impute data
        split: imputer.transform(data.splits["data"][split])
        for split in ["train", "val", "test"]
    }


def knn(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    imputer = TunableEstimator(
        MixedFeatureImputer(
            data.dataset_loader.continuous_cols,
            onehot_groupby=data.groupby["original"]["categorical_onehots"],
            numeric_transformer=KNNImputer(),
            categorical_transformer=KNNImputer(
                n_neighbors=1,  # any more than that and it'll take the average
            ),
            categorical_ignore_params=["n_neighbors", "weights"],
        ),
        score_fn=TransformScorer(
            get_tune_metric(ImputerT.BASELINE, data, "original"),
            higher_is_better=False,
            missingonly=True,
        ),
    )
    imputer.fit(data.splits["data"], data.splits["ground_truth"])
    # need to pickle with cloudpickle bc score_fn is lambda
    with open(get_serialized_model_path("knn"), "wb") as f:
        dump(imputer, f)
    return {
        # TODO: this might not be necessary anymore bc of mixedfeatureimputer
        # Add columns back in (sklearn erases) for rmse for missing only columns
        split: pd.DataFrame(
            # impute data
            imputer.transform(data.splits["data"][split]),
            columns=data.columns["original"],
            # some pandas operations rely on the pandas indices (e.g. pd.where)
            index=data.splits["data"][split].index,
        )
        for split in ["train", "val", "test"]
    }


def mice(args: Namespace, data: CommonDataModule) -> Dict[str, pd.DataFrame]:
    """Uses sklearn instead of miceforest package."""
    imputer = TunableEstimator(
        MixedFeatureImputer(
            data.dataset_loader.continuous_cols,
            onehot_groupby=data.groupby["original"]["categorical_onehots"],
            numeric_transformer=IterativeImputer(random_state=args.seed),
            categorical_transformer=IterativeImputer(
                estimator=DecisionTreeClassifier(),  # works for bin and multicat
                initial_strategy="most_frequent",
                random_state=args.seed,
            ),
        ),
        score_fn=TransformScorer(
            get_tune_metric(ImputerT.BASELINE, data, "original"),
            higher_is_better=False,
            missingonly=True,
        ),
    )
    imputer.fit(data.splits["data"], data.splits["ground_truth"])
    with open(get_serialized_model_path("mice"), "wb") as f:
        dump(imputer, f)
    return {
        # Add columns back in (sklearn erases) for rmse for missing only columns
        split: pd.DataFrame(
            # impute data
            imputer.transform(data.splits["data"][split]),
            columns=data.columns["original"],
            # some pandas operations rely on the pandas indices (e.g. pd.where)
            index=data.splits["data"][split].index,
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
