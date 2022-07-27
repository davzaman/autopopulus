from typing import Dict, List, Optional, Tuple, Union

import re
import numpy as np

# from numpy.lib.utils import deprecate
import pandas as pd
from torch import Tensor, LongTensor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import torch

# from lib.MDLPC.MDLP import MDLP_Discretizer

from autopopulus.data.mdl_discretization import ColInfoDict, MDLDiscretizer
from autopopulus.utils.utils import nanmean


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    nparr = tensor.detach().cpu().numpy()
    if len(nparr.shape) == 3:  # If longitudinal reshape to 2d, keeping nfeatures
        nfeatures = nparr.shape[-1]
        series_len = nparr.shape[0]
        nparr = nparr.reshape(-1, nfeatures)
    return nparr


def sigmoid_cat_cols(data: Tensor, cat_col_idx: Optional[LongTensor]) -> Tensor:
    """Puts categorical columns of tensor through sigmoid.
    Uses list of continuous columns/ctn/cat_col_idx to put only categorical columns through sigmoid.
    """
    if cat_col_idx is not None:
        data[:, cat_col_idx] = torch.sigmoid(data[:, cat_col_idx])
    else:
        data = torch.sigmoid(data)
    return data


class SimpleImpute(TransformerMixin, BaseEstimator):
    """Mean impute continuous data, mode impute categorical data."""

    def __init__(self, ctn_cols: List[str]):
        self.ctn_cols = ctn_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SimpleImpute":
        self.columns = list(X.columns)
        # categorical columns is everything but continuous columns
        self.cat_cols = [col for col in self.columns if col not in self.ctn_cols]

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean"))]
        )
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
        )
        self.transformers = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.ctn_cols),
                ("cat", categorical_transformer, self.cat_cols),
            ]
        )
        self.transformers.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transformers.transform(X)
        transformed = pd.DataFrame(transformed, columns=self.ctn_cols + self.cat_cols)
        # Rearrange to match original order
        return transformed[self.columns]


def simple_impute_tensor(
    X: Tensor, ctn_col_idx: LongTensor, cat_col_idx: LongTensor
) -> Tensor:
    """Simple imputes on a tensor. Mean(ctn) mode(cat)."""
    # Aggregate across rows (column mean/mode)
    if len(X.shape) == 2:  # static
        X[:, ctn_col_idx] = X[:, ctn_col_idx].where(
            ~torch.isnan(X[:, ctn_col_idx]), nanmean(X[:, ctn_col_idx], dim=0)
        )
        X[:, cat_col_idx] = X[:, cat_col_idx].where(
            ~torch.isnan(X[:, cat_col_idx]), torch.mode(X[:, cat_col_idx], dim=0).values
        )
    elif len(X.shape) == 3:  # static
        # TODO: figure out how to do this right? LEFT OF HERE
        X[:, :, ctn_col_idx] = X[:, :, ctn_col_idx].where(
            ~torch.isnan(X[:, :, ctn_col_idx]), nanmean(X[:, :, ctn_col_idx], dim=0)
        )
        X[:, :, cat_col_idx] = X[:, :, cat_col_idx].where(
            ~torch.isnan(X[:, :, cat_col_idx]),
            torch.mode(X[:, :, cat_col_idx], dim=0).values,
        )

    return X


def undiscretize_tensor(
    X: Tensor,
    groupby: Dict[str, Dict[int, int]],
    discretizations: Dict[str, Union[List[Tuple[float, float]], List[int]]],
    orig_columns: List[str],
) -> Tensor:
    # TODO: Write tests
    """Convert discretized columns to continuous, ignoring categorical columns.

    If a continuous var was discretized into multiple bins,
    grab the most likely one (max score of the bins) and return the
    mean of the range of that bin.
    """
    # X is Tensor, want to convert to df
    X_disc = pd.DataFrame(tensor_to_numpy(X))

    # Get the index of the bin with the maximum score for each column group
    col_max_indices = X_disc.groupby(groupby, axis=1).idxmax(axis=1)
    #  offset the indices to 0 so we can directly index into that vars ["bins"]
    offset_coln_maxes = col_max_indices.apply(
        lambda var_data: var_data - discretizations[var_data.name]["indices"][0]
    )
    # get the bin range
    coln_max_bins = offset_coln_maxes.apply(
        lambda var: var.map(lambda idx: discretizations[var.name]["bins"][int(idx)])
    )
    # apply function to bin range for continuous estimate
    continuous_estimates = coln_max_bins.applymap(lambda range: np.mean(range))
    # drop the discretized columns and add their continuous estimates
    X_cont = pd.concat(
        [X_disc.drop(groupby.keys(), axis=1), continuous_estimates], axis=1
    )
    # add other column names back in
    X_cont = X_cont.rename(
        {
            int(idx): name
            for idx, name in enumerate(
                pd.Index(orig_columns).drop(col_max_indices.columns)
            )
        },
        axis=1,
    )
    # Reorder to match original order of columns that discretizing might have affected
    X_cont = X_cont[orig_columns].astype(float)

    # reshape back to 3d if longitudinal
    if len(X.shape) == 3:
        n_features = X_cont.shape[-1]
        n_sequence = X.shape[1]
        X_cont = X_cont.values.reshape((-1, n_sequence, n_features))
    else:
        X_cont = X_cont.values

    return torch.tensor(X_cont, device=X.device).float()


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
        TODO: this could possibly produce a whole column of errors.
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
    range_str = max_col.rpartition("_")[-1]
    if re.search(range_regex, range_str):
        # strip spaces and brackets, split by comma, convert to float
        low, high = tuple(map(float, range_str.split(" - ")))
        # return average
        return (low + high) / 2


class UniformProbabilityAcrossNans(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        groupby_categorical_only: Dict[str, Dict[int, str]],
        ground_truth_pipeline: bool = False,
    ):
        self.groupby_categorical_only = groupby_categorical_only
        self.ground_truth_pipeline = ground_truth_pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UniformProbabilityAcrossNans":
        return self

    def transform(
        self, discretizer_output: Tuple[pd.DataFrame, ColInfoDict]
    ) -> pd.DataFrame:
        """Imposes a uniform probability across all categories with nans."""
        X, col_info_dict = discretizer_output
        # Do nothing if nothing is missing
        if X.notna().values.all():
            return X

        # TODO: this might not be necessary the groupby seems to be a pointer so after fit this might reflect the changes properly?
        # deal with dicretized vars, I cannot add it to groupby until after fit is done anyway
        for col_info in col_info_dict.values():
            col_names = X.columns[col_info["indices"]]
            X[col_names] = X[col_names].fillna(1 / len(col_names))

        # deal with categorical vars (multicat and binary)
        invert_groupby = {}
        for indices_to_groupid in self.groupby_categorical_only.values():
            # TODO: pass the correct indicator of data/gt
            if indices_to_groupid:
                # discretized ctn_cols is the only one that separates data and
                if "data" in indices_to_groupid:
                    key = "ground_truth" if self.ground_truth_pipeline else "data"
                    indices_to_groupid = indices_to_groupid[key]
                for index, group_id in indices_to_groupid.items():
                    invert_groupby.setdefault(group_id, []).append(X.columns[index])
        # TODO: move this comment to set_groupby in commondatautils
        # need to have indices for column labels for groupby to work
        # we keep the indices bc later when we dont have column names we still want groupby to work
        for expanded_cols in invert_groupby.values():
            X[expanded_cols] = X[expanded_cols].fillna(1 / len(expanded_cols))

        return X


class Discretizer(TransformerMixin, BaseEstimator):
    # have to use orange for now, the other version is still WIP
    def __init__(
        self,
        orig_cols: List[str],
        ctn_cols_idx: List[int],
        method="mdl",
        return_info_dict: bool = False,
    ):
        self.ctn_cols_idx = ctn_cols_idx
        self.orig_cols = orig_cols
        self.method = method
        self.return_info_dict = return_info_dict

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Discretizer":
        if self.method == "mdl":  # Minimum description length via Orange package.
            self.d = MDLDiscretizer()
            self.d.fit(X, y, X.columns[self.ctn_cols_idx])
            self.map_dict = self.d.ctn_to_cat_map
            return self
        raise NotImplementedError
        self.d.fit(X.values, y.values)
        # replace indices with column names
        self.map_dict = {
            self.orig_cols[col_idx]: mapping
            for col_idx, mapping in self.d._bin_descriptions.items()
        }
        return self

    def transform(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ColInfoDict]]:
        if self.method == "mdl":
            transformed_df = self.d.transform(X)
            # TODO: document what map dict is
        else:
            raise NotImplementedError
            transformed_df = (
                self.special_dummies(self.nptopd(self.d.transform(X.values))),
            )
        return (
            (transformed_df, self.map_dict) if self.return_info_dict else transformed_df
        )

    def special_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        # put back in columns that should be there but aren't in the split
        # TODO: this doesn't work because it fills the whole missing column with 0s but the row should propagate nans if the value is missing...
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
