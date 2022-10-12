import json
from argparse import Namespace
from typing import Dict, Optional, Tuple, Union, List, Callable

import numpy as np
import pandas as pd
import itertools

DATA_DIR = "/home/davina/Private/mimic3-benchmarks/data/decompensation"
CHANNEL_INFO_PATH = (
    "/home/davina/Private/mimic3-benchmarks/mimic3models/resources/channel_info.json"
)
MIMIC_CTN_COLS = [
    # "Glucose",
    "Systolic blood pressure",
    # "Temperature",
    # "Weight",
    "Diastolic blood pressure",
    # "Fraction inspired oxygen",
    "Mean blood pressure",
    "Heart Rate",
    "Oxygen saturation",
    # "pH",
    # "Height",
    "Respiratory rate",
]
MIMIC_CAT_COLS = [
    # "Glascow coma scale verbal response",
    # "Glascow coma scale total",
    # "Capillary refill rate",
    # "Glascow coma scale eye opening",
    # "Glascow coma scale motor response",
]

PERIODS = {
    "all": lambda x: pd.Series([True] * len(x)),
    "first4days": lambda x: x < 4 * 24,
    "first8days": lambda x: x < 8 * 24,
    "last12hours": lambda x: x > x.max() - 12,
    "first25percent": lambda x: x < first_p_percent(x.min(), x.max(), 25),
    "first50percent": lambda x: x < first_p_percent(x.min(), x.max(), 50),
}
# add fuzzy window of margin 1e-6. 7 sub-periods
SUBPERIODS = {
    "all": lambda x: pd.Series([True] * len(x)),
    "first10percent": lambda x: x < first_p_percent(x.min(), x.max(), 10) + 1e-6,
    "first25percent": lambda x: x < first_p_percent(x.min(), x.max(), 25) + 1e-6,
    "first50percent": lambda x: x < first_p_percent(x.min(), x.max(), 50) + 1e-6,
    "last10percent": lambda x: x > last_p_percent(x.min(), x.max(), 10) - 1e-6,
    "last25percent": lambda x: x > last_p_percent(x.min(), x.max(), 25) - 1e-6,
    "last50percent": lambda x: x > last_p_percent(x.min(), x.max(), 50) - 1e-6,
}


def count(coldata):
    """Returns count if column is not all nans."""
    return coldata.count() if coldata.notna().all() else np.nan


AGG_FUNCTIONS = ["min", "max", "mean", "std", "skew", count]


def load_mimic_data(
    args: Namespace, period: str = "all", limit: Optional[int] = None
) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]]:
    """Load mimic data, preprocessed by the YerevaNN package."""
    splits = ["train", "val", "test"]
    subdatadirs = ["train", "train", "test"]  # val lives in train dir

    # Load data: ground truth
    X_true, y_true = {}, {}
    for split, subdatadir in zip(splits, subdatadirs):
        listfile_path = f"{DATA_DIR}/{split}_listfile.csv"
        # stay (csv file name), period_length, y_true
        stay_data = pd.read_csv(listfile_path)
        cat_map = get_categorical_map()
        cat_map = {
            col: mapping for col, mapping in cat_map.items() if col in MIMIC_CAT_COLS
        }
        nrows = limit if limit else len(stay_data)

        # add each episode row by row
        X = pd.concat(
            [
                extract_episode_features(
                    stay_data, ep_index, subdatadir, cat_map, period
                )
                for ep_index in range(nrows)
            ],
            axis=1,
        ).T
        y = stay_data["y_true"][:nrows]

        if args.fully_observed:
            # keep rows NOT missing a value for any feature
            fully_observed_mask = X.notna().all(axis=1)
            X = X[fully_observed_mask]
            y = y[fully_observed_mask]

        X_true[split] = X
        y_true[split] = y

    # combines train, val, test
    # TODO: ampute on combined instead of individual?
    ground_truth = pd.concat([X_true[split] for split in splits], ignore_index=True)
    # but accounts for indices to retreive them again later
    # it's strange because this is how i set it up for ckd dataset
    indices = {}
    for i, split in enumerate(splits):
        # sum of len of all previous splits
        start = sum(len(X_true[splits[j]]) for j in range(i))
        # start + length of this split
        end = start + len(X_true[split])
        indices[split] = pd.Index(range(start, end))

    X = {}
    for split in splits:
        if "percent_missing" in args and "missingness_mechanism" in args:
            pass
        else:
            X[split] = X_true[split].copy()
        X[split].index = indices[split]

    return {
        "train": (X["train"], y_true["train"]),
        "val": (X["val"], y_true["val"]),
        "test": (X["test"], y_true["test"]),
        "ground_truth": ground_truth,
    }


def extract_episode_features(
    stay_data: pd.DataFrame,
    episode_index: int,
    subdatadir: str,
    cat_map: Dict[str, Dict[str, int]],
    period: str,
) -> pd.Series:
    """Loads an episode and aggregates the info to create a flattened sample.
    7 subperiods * 6 functions * 17 features (18 - 1, since we don't include time) = 714 features for 1 flattened sample.
    """
    # Grab stay data from csv
    episode = pd.read_csv(f"{DATA_DIR}/{subdatadir}/{stay_data['stay'][episode_index]}")
    # Filter for features to keep
    episode = episode[["Hours"] + MIMIC_CTN_COLS + MIMIC_CAT_COLS]
    # filter for proper period length with fuzzy 1e-6 margin
    episode = episode[
        episode["Hours"] <= stay_data["period_length"][episode_index] + 1e-6
    ]

    # Transform cat cols (Ref: https://stackoverflow.com/a/41678874/1888794)
    for col, colmap in cat_map.items():
        # for non-exhaustive maps, fillna with original
        episode[col] = episode[col].map(colmap).fillna(episode[col])
    # ensure all are floats
    episode = episode.astype(float)
    subperiods = get_subperiods(period, episode["Hours"])

    # calc each agg function per subperiod for all features
    agg_value_by_subperiod = (
        pd.concat(
            [
                episode[subperiod].agg(AGG_FUNCTIONS)
                for subperiod in subperiods.values()
            ],
            keys=subperiods.keys(),
        )
        .drop("Hours", axis=1)  # drop hours since it's not a feature
        .T  # Transpose so flattening works properly
    )

    ## Flatten ##
    # an episode = 1 example for training, so we need to flatten the df into 1 line
    # Flatten hierarchical columns
    agg_value_by_subperiod.columns = [
        "_".join(col).strip() for col in agg_value_by_subperiod.columns.values
    ]
    # TODO: deal with whitespace in feature names
    flattened_names = [
        f"{feature}_{subp_fn}"
        for feature in agg_value_by_subperiod.index
        for subp_fn in agg_value_by_subperiod.columns
    ]
    flattened = agg_value_by_subperiod.values.flatten()
    return pd.Series(flattened, index=flattened_names)


def get_subperiods(period: str, episode_hours: pd.Series) -> Dict[str, pd.Series]:
    """Returns the masks for each subperiod window.
    Takes the period window and the "Hours" for each episode.
    """
    # create the mask using the hours
    period_window = PERIODS[period](episode_hours)
    return {
        subperiodn: window(episode_hours[period_window])
        for subperiodn, window in SUBPERIODS.items()
    }


def get_categorical_map() -> Dict[str, Dict[str, int]]:
    """Returns dict to map categorical features to be numerically encoded."""
    with open(CHANNEL_INFO_PATH) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    return {
        coln: colmap["values"]
        for coln, colmap in channel_info.items()
        if "values" in colmap
    }


def first_p_percent(start: int, end: int, p: int) -> int:
    """Returns new endpoint for a given window of an episode."""
    return start + (end - start) * p / 100.0


def last_p_percent(start: int, end: int, p: int) -> int:
    """Returns new startpoint for a given window of an episode."""
    return end - (end - start) * p / 100.0


def generate_agg_col_name(cols: List[Union[str, Callable]]) -> List[str]:
    """Takes colname and then returns colname_subperiod_aggregatefxn."""
    newcols = []
    for colname, subperiod, aggfxn in itertools.product(
        cols, SUBPERIODS.keys(), AGG_FUNCTIONS
    ):
        # Get function name as string if it's a function and not string name of function
        if callable(aggfxn):
            aggfxn = aggfxn.__name__
        newcols.append(f"{colname}_{subperiod}_{aggfxn}")
    return newcols
