from typing import List, Optional, Tuple, Union

import re
import numpy as np
from numpy.lib.utils import deprecate
import pandas as pd
from torch import Tensor, LongTensor

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import torch

from utils.MDLDiscretization import MDLDiscretizer
from data.ckd import RACE_COLS

# from lib.MDLPC.MDLP import MDLP_Discretizer
from utils.utils import ChainedAssignent, nanmean


def scale(
    train: pd.DataFrame,
    to_transform: List[pd.DataFrame],
    ctn_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, ...]:
    # ) -> Tuple[np.ndarray, ...]:
    """Scale continuous features to [0, 1].
    This can produce negative numbers."""
    if not ctn_cols:
        ctn_cols = train.columns
    scaler = MinMaxScaler()
    scaler.fit(train[ctn_cols])

    def transform_and_merge(df: pd.DataFrame) -> pd.DataFrame:
        # Avoid set with copy warning
        with ChainedAssignent():
            df.loc[:, ctn_cols] = scaler.transform(df[ctn_cols])
        return df

    return (*(transform_and_merge(data) for data in [train] + to_transform),)


def simple_impute(
    train: pd.DataFrame,
    # columns: List[str],  # used to preserve order, and get cat_cols
    ctn_cols: List[str],
    to_transform: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, ...]:
    """Uses training data to impute train and everything in to_transform.
    Mean impute continuous data, mode impute categorical data."""
    columns = list(train.columns)
    # categorical columns is everything but continuous columns
    cat_cols = [col for col in columns if col not in ctn_cols]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    transformers = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ctn_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    transformers.fit(train)

    def impute(data: pd.DataFrame) -> pd.DataFrame:
        # columntransformer rearranges the columns
        transformed = transformers.transform(data)
        transformed = pd.DataFrame(transformed, columns=ctn_cols + cat_cols)
        # Rearrange to match original order
        return transformed[columns]

    return (
        impute(train),
        *(impute(data) for data in to_transform),
    )


def simple_impute_tensor(
    X: Tensor, ctn_col_idx: LongTensor, cat_col_idx: LongTensor
) -> Tensor:
    """Simple imputes on a tensor. Mean(ctn) mode(cat)."""
    # Aggregate across rows (column mean/mode)
    X[:, ctn_col_idx] = X[:, ctn_col_idx].where(
        ~torch.isnan(X[:, ctn_col_idx]), nanmean(X[:, ctn_col_idx], dim=0)
    )
    X[:, cat_col_idx] = X[:, cat_col_idx].where(
        ~torch.isnan(X[:, cat_col_idx]), torch.mode(X[:, cat_col_idx], dim=0).values
    )
    return X


def discretize(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    to_transform: List[pd.DataFrame],
    ctn_col: List[str],
) -> Tuple[pd.DataFrame, ...]:
    # ) -> Tuple[pd.DataFrame, pd.DataFrame, ..., Dict[str, Dict[int, str]]]:
    """Discretizes continuous vars (supervised).
    Trained on training and applied to validation and test sets.
    Will return one-hot encoded results along with the bins generated.
    """
    discretizer = Discretizer(X_train.columns, ctn_col)
    train = discretizer.fit_transform(X_train, y_train)
    transformed = (discretizer.transform(df) for df in to_transform)

    return (train, *transformed, discretizer.map_dict)


def uniform_prob_across_nan(
    onehot: pd.DataFrame, num_categories: pd.Series, ctn_cols: List[str]
) -> pd.DataFrame:
    """Imposes a uniform probability across all categories with nans."""
    # Do nothing if nothing is missing
    if onehot.notna().values.all():
        return onehot

    # Each var was discretized with a different number of bins,
    # so we fill in differently per continuous var.
    for ctn_col_name in ctn_cols:
        discretized_cols = onehot.columns[onehot.columns.str.startswith(ctn_col_name)]
        uniform_prob = 1 / num_categories[ctn_col_name]
        onehot.fillna({col: uniform_prob for col in discretized_cols}, inplace=True)

    for col in onehot.columns[onehot.isna().any()]:
        uniform_prob = 1 / num_categories[col.rpartition("_")[0]]
        onehot.fillna({col: uniform_prob}, inplace=True)

    return onehot


def undiscretize_tensor(
    X: Tensor,
    orig_columns: List[str],
    discrete_columns: List[str],
    ctn_columns: List[str],
) -> Tensor:
    """Convert discretized columns to continuous, ignoring categorical columns.

    If a continuous var was discretized into multiple bins,
    grab the most likely one (max score of the bins) and return the
    mean of the range of that bin.
    """
    # X is Tensor, want to convert to df
    # X_disc = pd.DataFrame(X.detach().numpy(), columns=discrete_columns)
    X_disc = pd.DataFrame(X.detach().cpu().numpy(), columns=discrete_columns)

    # We can group vars into one by the prefix of the name
    def orig_ctn_col_name(coln: str) -> str:
        prefix, delim, suffix = coln.rpartition("_")
        # get original name if match, else leave it alone (categorical/one hot)
        return prefix if prefix in ctn_columns else coln

    # Get the colname of the bin with the maximum score
    coln_maxes = X_disc.groupby(orig_ctn_col_name, axis=1).idxmax(axis=1)
    # parse the range of the bin and return the mean
    means_of_maxbin = coln_maxes.applymap(mean_of_maxbin)
    # consolidate the multiple bins into the 1 continuous column
    X_cont = X_disc.groupby(orig_ctn_col_name, axis=1).max()
    # keep value if categorical/one hot, else replace with mean of max bin
    X_cont = X_cont.where(means_of_maxbin.isna(), means_of_maxbin)
    # Reorder to match original order of columns that discretizing might have affected
    X_cont = X_cont[orig_columns].astype(float)

    return torch.tensor(X_cont.values, device=X.device).float()


def ampute(
    X: pd.DataFrame,
    seed: int,
    missing_cols: List[str],
    percent: float = 0.33,
    mech: str = "MCAR",
    observed_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Will simulate missing values according to the missingness mechanism.

    MCAR =  missingness is not related to any value.
    MAR = missingness is related to observed values.
    MNAR = missingness is dependant on unobserved values.

    observed_cols only for MAR
    """
    X_amputed = X.copy()
    if mech == "MCAR":
        # for each observation sample a random value from uniform dist
        # and keep it if it's less than percent.
        # X[uniform(size=X.shape) < percent] = np.nan
        np.random.seed(seed)
        X_amputed[missing_cols] = X_amputed[missing_cols].mask(
            np.random.uniform(size=X_amputed[missing_cols].shape) < percent,
            np.nan,
        )
    elif mech == "MAR":
        """
        Missing because of the value of some observed var.
        If your egfr is greater than some value that puts you in the
        (1-p) percentile, then your a1c and sbp will be missing.
        """
        # these match the original dataset
        for observed_col, missing_col in zip(observed_cols, missing_cols):
            # cutoff at (1-p) percentile, if you fall above, ampute
            p_quantile = X_amputed[observed_col].quantile(1 - percent)
            cutoff = X_amputed[observed_col] > p_quantile
            X_amputed[missing_col] = X_amputed[missing_col].mask(cutoff, np.nan)
    elif mech == "MNAR":
        """
        Missing because of its true value itself for each var.
        If it's in the middle/expected range then it will be missing.
        """
        # grab `percent` patients in the middle (%-ile)
        low, high = (0.5 - (percent / 2), 0.5 + (percent / 2))
        quantiles = X[missing_cols].quantile([low, high])
        for col in missing_cols:
            above_low = X[col] > quantiles.loc[low, col]
            below_high = X[col] < quantiles.loc[high, col]
            X_amputed[col] = X_amputed[col].mask(above_low & below_high, np.nan)
    elif mech == "MNAR1":
        """
        Missing because of the value of some hidden/unobserved var.
        If the value of the hidden var is in the tails, it will be missing.
        """
        # these match the original dataset
        # hidden normally distributed var
        np.random.seed(seed)
        hidden = pd.DataFrame(np.random.normal(0, 1, X.shape), columns=X.columns)
        # grab tails
        low, high = ((percent / 2), 1 - (percent / 2))
        quantiles = hidden.quantile([low, high])
        for col in missing_cols:
            low_tail = hidden[col] < quantiles.loc[low, col]
            high_tail = hidden[col] > quantiles.loc[high, col]
            X_amputed[col] = X_amputed[col].mask(low_tail | high_tail, np.nan)
    return X_amputed


def mean_of_maxbin(max_col: str) -> float:
    """For the maximum likely bin of a discretized variable, return the
    mean for that bin range."""
    # decimal number, space, - , space, decimal number
    range_regex = r"\d+\.?\d*-\d*\.?\d*"
    # the range comes after the name: name_low - high
    range_str = max_col.rpartition("_")[-1].replace(" ", "")
    if re.search(range_regex, range_str):
        # strip spaces and brackets, split by comma, convert to float
        low, high = tuple(map(float, range_str.split("-")))
        # return average
        return (low + high) / 2


# DEPRECATED: the preprocessed file already has this one hot encoded
def one_hot_race(X: pd.DataFrame):
    """Directly mutates X to one-hot encode the race columns."""
    X[RACE_COLS] = pd.get_dummies(X["demo_race_ethnicity_cat"])
    X.drop("demo_race_ethnicity_cat", axis=1, inplace=True)


class Discretizer(TransformerMixin, BaseEstimator):
    # have to use orange for now, the other version is still WIP
    def __init__(self, orig_cols: List[str], ctn_cols: List[str], useOrange=True):
        self.ctn_cols = ctn_cols
        self.orig_cols = orig_cols
        self.ctn_col_idx = [orig_cols.get_loc(col) for col in self.ctn_cols]
        self.orange = useOrange
        if useOrange:  # Minimum description length via Orange package.
            self.d = MDLDiscretizer()
        else:
            self.d = MDLP_Discretizer(self.ctn_col_idx)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Discretizer":
        if self.orange:
            self.d.fit(X, y, self.ctn_cols)
            self.map_dict = self.d.dicts
            return self
        self.d.fit(X.values, y.values)
        # replace indices with column names
        self.map_dict = {
            self.orig_cols[col_idx]: mapping
            for col_idx, mapping in self.d._bin_descriptions.items()
        }
        # self.map_dict = {
        #     self.orig_cols[col_idx]: {
        #         k: f"{binrange[0]} - {binrange[1]}" for k, binrange in binmap.items()
        #     }
        #     for col_idx, binmap in self.d._bin_descriptions.items()
        # }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.orange:
            return self.special_dummies(self.d.transform(X, self.ctn_cols))
        return self.special_dummies(self.nptopd(self.d.transform(X.values)))

    def special_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        # Columns: "colname_binrange0 - binrange1"
        df = pd.get_dummies(df, dummy_na=True)
        # put back in all nans
        for col in self.ctn_cols:
            df.loc[df[f"{col}_nan"] == 1, df.columns.str.startswith(col)] = np.nan
        # drop nan dummy
        df.drop(df.columns[df.columns.str.endswith("_nan")], axis=1, inplace=True)

        if not self.orange:
            # put back in columns that should be there but aren't in the split
            discretized_cols = list(self.orig_cols.difference(self.ctn_cols))[::-1] + [
                f"{col}_{range_str}"
                for col in self.ctn_cols
                for range_str in self.map_dict[col].values()
            ]
            # https://stackoverflow.com/questions/37425961/dummy-variables-when-not-all-categories-are-present
            df = df.T.reindex(discretized_cols).T
            missing_cols = df.columns[df.isna().all()]
            df.fillna({col: 0 for col in missing_cols}, inplace=True)
        return df

    def nptopd(self, arr: np.ndarray) -> pd.DataFrame:
        """Converts np array from transform into proper pandas dataframe with desired column."""
        df = pd.DataFrame(arr, columns=self.orig_cols)
        # df map from numerical encoding to str representing bin ranges
        df.replace(self.map_dict, inplace=True)
        return df
