from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import re
from category_encoders import TargetEncoder
import numpy as np
import pandas as pd
from torch import Tensor, nan_to_num

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import torch
from torch.nn.utils.rnn import unpad_sequence, pad_sequence

# from lib.MDLPC.MDLP import MDLP_Discretizer

from autopopulus.data.mdl_discretization import ColInfoDict, MDLDiscretizer
from autopopulus.data.utils import (
    explode_nans,
    onehot_multicategorical_column,
    enforce_numpy,
)
from autopopulus.data.constants import PAD_VALUE

DEFAULT_DEVICE = torch.device("cpu")


###############
#   HELPERS   #
###############


def identity(x: Any) -> Any:
    # Useful as I can compare for testing
    return x


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    tensor_cpu = tensor.detach().cpu()
    if tensor_cpu.dtype == torch.bfloat16:
        tensor_cpu = tensor_cpu.to(torch.float16)
    nparr = tensor_cpu.numpy()
    if len(nparr.shape) == 3:  # If longitudinal reshape to 2d, keeping nfeatures
        nfeatures = nparr.shape[-1]
        series_len = nparr.shape[0]
        nparr = nparr.reshape(-1, nfeatures)
    return nparr


def list_to_tensor(
    idxs: Union[List[int], List[List[int]]],
    device: Union[str, torch.device] = DEFAULT_DEVICE,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """
    Converts col_idxs_by_type indices into a tensor.
    If List[List[int]] there may uneven length of group indices,
        which produces ValueError, so this tensor needs to be padded.
    dtype must be long for indices, but must be floats if you want to do computations like mean.
    """
    try:
        # Creating tensor from list of arrays is slow
        if len(idxs) > 1 and isinstance(idxs[0], list):
            idxs = torch.stack(idxs, axis=0)
        return torch.tensor(idxs, dtype=dtype, device=device)
    except (ValueError, TypeError):  # pad when onehot list of group indices
        return pad_sequence(
            [
                torch.tensor(group_idxs, dtype=dtype, device=device)
                for group_idxs in idxs
            ],
            batch_first=True,  # so they function as rows
            padding_value=PAD_VALUE,
        )


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


###################
# MAIN TRANSFORMS #
###################
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
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.transformers.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_of_order_cols = self.transformers.get_feature_names_out(self.columns)
        return pd.DataFrame(
            self.transformers.transform(X), columns=out_of_order_cols, index=X.index
        )[self.columns]


def simple_impute_tensor(
    data: Tensor,
    non_missing_mask: Tensor,
    ctn_col_idxs: Tensor,
    bin_col_idxs: Tensor,
    onehot_group_idxs: Tensor,
    return_learned_stats: bool = False,  # for testing
) -> Tensor:
    # TODO: what to do if whole column is nan?
    # TODO: what to do if none missing
    X = data.clone()  # if we dont copy, batch["original"]["data"] will change too
    means = []
    for ctn_col in ctn_col_idxs:
        mean = X[:, ctn_col].nanmean()
        X[:, ctn_col] = nan_to_num(X[:, ctn_col], mean)
        means.append(mean)

    modes = []
    for bin_col in bin_col_idxs:
        # filter the rows with missing bin_col values and compute the mode
        mode = X[non_missing_mask[:, bin_col]][:, bin_col].mode().values
        X[:, bin_col] = nan_to_num(X[:, bin_col], mode)
        modes.append(mode)
    # WARNING: https://github.com/pytorch/pytorch/issues/46225
    # this code works despite that behavior
    for onehot_group_idx in onehot_group_idxs:
        # ignore pads of -1
        onehot_group_idx = onehot_group_idx[onehot_group_idx != PAD_VALUE]
        # all the onehot categories should be nan
        non_missing = non_missing_mask[:, onehot_group_idx]

        # mode but numerically encoded
        # get the "index/bin/category" for each sample, then the mode of that
        numerical_enc_mode = (
            torch.argmax(X[non_missing.all(axis=1), :][:, onehot_group_idx], dim=1)
            .mode()
            .values
        )
        # cannot combine mask and idxs bc not same size
        X[:, onehot_group_idx] = X[:, onehot_group_idx].where(
            non_missing,  # keep wehre not missing else impute with onehotted mode
            # explode the mode back into a onehot vec
            torch.nn.functional.one_hot(
                numerical_enc_mode,
                num_classes=len(onehot_group_idx),
            ).to(X.dtype),
        )
        modes.append(numerical_enc_mode)

    if return_learned_stats:
        return (X, means, modes)
    return X


class CombineOnehots(TransformerMixin, BaseEstimator):
    """
    Inverts onehot columns by combining them into a single column.
    Relies on a groupby, and will add the combined column at the end.
    Assumed onehots are prefixed with its name: prefix_value.
    """

    def __init__(self, onehot_groupby: Dict[int, str], columns: List[str]):
        # need the column names not indices since we're working with pd
        self.onehot_groupby = {
            columns[idx]: prefix for idx, prefix in onehot_groupby.items()
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CombineOnehots":
        """Set things we'll need for CommonDataModule so we can test separately."""
        groups = X.groupby(self.onehot_groupby, axis=1).groups
        onehot_prefixes = groups.keys()
        # drop old cols
        shifted_down_cols = X.drop(self.onehot_groupby.keys(), axis=1).columns
        N = len(shifted_down_cols)
        # the combined col will be added at the end so we can guess the indices
        # map index -> prefix name by tacking onto the end of the shifted down cols after dropping the onehots
        self.combined_onehot_groupby = {
            N + i: prefix for i, prefix in enumerate(onehot_prefixes)
        }
        # we subtract all the exploded onehots and add the singular combined ones
        self.nfeatures = len(X.columns) - sum(map(len, groups.values())) + len(groups)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """."""
        if self.onehot_groupby:
            combined_onehots = (
                X.groupby(self.onehot_groupby, axis=1)
                .idxmax(
                    1, numeric_only=True
                )  # replace value with the column name of the max
                .apply(lambda col: col.str.replace(f"{col.name}_", ""))  # strip prefix
            )
            # combine with the rest of the data and then drop old onehot cols
            new_df = pd.concat([X, combined_onehots], axis=1).drop(
                self.onehot_groupby.keys(), axis=1
            )
            return new_df
        return X


class UniformProbabilityAcrossNans(TransformerMixin, BaseEstimator):
    """Make sure to also check "uniform_probability_across_nans" to update tests."""

    def __init__(
        self,
        groupby_categorical_only: Dict[str, Dict[int, str]],
        columns: List[str],
        ground_truth_pipeline: bool = False,
    ):
        self.groupby_categorical_only = {
            groupby_name: {columns[idx]: prefix for idx, prefix in groupby_idxs.items()}
            for groupby_name, groupby_idxs in groupby_categorical_only.items()
        }
        self.ground_truth_pipeline = ground_truth_pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UniformProbabilityAcrossNans":
        return self

    def transform(
        self, discretizer_output: Tuple[pd.DataFrame, ColInfoDict]
    ) -> pd.DataFrame:
        """Imposes a uniform probability across all categories with nans."""
        df, col_info_dict = discretizer_output
        X = df.copy()
        # Do nothing if nothing is missing
        if X.notna().values.all():
            return X

        # deal with dicretized vars, I cannot add it to groupby until after fit is done anyway
        for col_info in col_info_dict.values():
            col_names = X.columns[col_info["indices"]]
            X[col_names] = X[col_names].fillna(1 / len(col_names))

        # deal with multi-categorical onehot vars
        if "categorical_onehots" in self.groupby_categorical_only:
            for onehot_idxs in X.groupby(
                self.groupby_categorical_only["categorical_onehots"], axis=1
            ).groups.values():
                X.loc[:, onehot_idxs] = X.loc[:, onehot_idxs].fillna(
                    1 / len(onehot_idxs)
                )

        if "binary_vars" in self.groupby_categorical_only:
            # binary should all be 0.5
            for bin_idxs in X.groupby(
                self.groupby_categorical_only["binary_vars"], axis=1
            ).groups.values():
                X.loc[:, bin_idxs] = X.loc[:, bin_idxs].fillna(0.5)

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
        # Set stuff I'll need for CommonDataModule so I can test it here.
        if self.method == "mdl":  # Minimum description length via Orange package.
            self.d = MDLDiscretizer()
            self.d.fit(X, y, X.columns[self.ctn_cols_idx])
            self.map_dict: Dict[
                str, List[Union[Tuple[float, float], str, int]]
            ] = self.d.ctn_to_cat_map  # bins, labels, indices

            self.discretized_groupby = {
                index: col
                for col, col_info in self.map_dict.items()
                for index in col_info["indices"]
            }
            num_added_bins = reduce(
                lambda added_num_cols, k: added_num_cols
                + len(self.map_dict[k]["indices"]),
                self.map_dict,
                0,
            )
            self.nfeatures = len(X.columns) + num_added_bins - len(self.map_dict)
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


class ColTransformPandas(TransformerMixin, BaseEstimator):
    """
    Wrapper class to ColumnTransformer to scale only continuous columns.
    Allows us to also return to pandas and restore original column order.
    TODO: write tests
    """

    def __init__(
        self, orig_cols: List[str], enforce_numpy: bool = False, *args, **kwargs
    ) -> None:
        super().__init__()
        self.orig_cols = orig_cols
        self.enforce_numpy = enforce_numpy  # TODO: might not be necessary
        self.column_tsfm = ColumnTransformer(
            *args, remainder="passthrough", verbose_feature_names_out=False, **kwargs
        )

    def fit(self, X: np.ndarray, y: pd.Series):
        # Enforce numpy so we don't get "UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names"
        if self.enforce_numpy:
            self.column_tsfm.fit(enforce_numpy(X), y)
        else:
            self.column_tsfm.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        if self.enforce_numpy:  # Need to retain access to X as df so don't overwrite
            Xt = self.column_tsfm.transform(enforce_numpy(X))
        else:
            Xt = self.column_tsfm.transform(X)
        # without original cols it's going to have meaningless feature names like x7
        out_of_order_cols = self.column_tsfm.get_feature_names_out(self.orig_cols)
        df = pd.DataFrame(Xt, columns=out_of_order_cols, index=X.index)
        return df[self.orig_cols]


##############################
# FEATURE MAPPING INVERSIONS #
##############################
def get_invert_discretize_tensor_args(
    discretizations: Dict,
    original_columns: pd.Index,
    to_device: torch.device = DEFAULT_DEVICE,
) -> Dict[str, Union[Tensor, List[Tensor]]]:
    # TODO: do i need the groupby anymore?
    # self.groupby["mapped"]["discretized_ctn_cols"]["data"],
    args = {
        "disc_groupby_idxs": [],  # list of tensor
        "bin_ranges": [],  # list of tensor
        # extra 0 is the "starting" column for stitching everythign together at the end
        "original_idxs": [],  # tensor
    }
    for coln, disc_info in discretizations.items():
        args["disc_groupby_idxs"].append(
            list_to_tensor(disc_info["indices"], device=to_device)
        )
        args["bin_ranges"].append(
            list_to_tensor(disc_info["bins"], dtype=torch.float, device=to_device)
        )
        args["original_idxs"].append(original_columns.get_loc(coln))
    args["original_idxs"] = list_to_tensor(args["original_idxs"], device=to_device)
    return args


def invert_discretize_tensor_gpu(
    encoded_data: Tensor,
    disc_groupby_idxs: List[Tensor],
    bin_ranges: List[Tensor],
    original_idxs: Tensor,
) -> Tensor:
    continuous_estimates = [bin_range.mean(dim=1) for bin_range in bin_ranges]
    # Get the index of the bin with the maximum score for each column group
    #  offset the indices to 0 so we can directly index into that var's ["bins"]
    offset_coln_max_idxs = torch.stack(
        [  # to not offset to 0 add + l[0]
            torch.index_select(encoded_data, dim=1, index=l).argmax(dim=1)
            for l in disc_groupby_idxs
        ],
        dim=1,
    )
    # get the mean of each category for each feature
    mapped_to_continuous = [
        mapping.index_select(0, offset_coln_max_idxs[:, i])
        for i, mapping in enumerate(continuous_estimates)
    ]
    # drop indices of discretized columns
    keep_columns = torch_set_diff(
        torch.arange(0, encoded_data.shape[1]), torch.cat(disc_groupby_idxs)
    )
    # drop discretized_columns and add the continuous estimates back in
    encoded_data = encoded_data.index_select(1, keep_columns)
    # put everything back in order and add in mapped columns
    cols = []
    # these act as indexors into encoded data and mapped data respectively.
    n_ctn_cols = 0
    n_cat_cols = 0
    for i, idx in enumerate(original_idxs):
        # since i starts@ 0  while idx starts@ 1, original_idxs[i] is the previous idx
        cols.append(encoded_data[:, n_cat_cols : idx - n_ctn_cols])
        cols.append(mapped_to_continuous[i].unsqueeze(1))  # adds only 1 column
        n_cat_cols = idx - n_ctn_cols
        n_ctn_cols += 1
    # add any remaining encoded data, will add nothing if idx is out of bounds
    cols.append(encoded_data[:, n_cat_cols:])
    return torch.cat(cols, 1)


def torch_set_diff(t1: Tensor, t2: Tensor) -> Tensor:
    """Ref: https://stackoverflow.com/a/62407582/1888794"""
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]
    return difference


def invert_discretize_tensor(
    encoded_data: Tensor,
    disc_groupby: Dict[int, str],
    discretizations: Dict[str, Union[List[Tuple[float, float]], List[int]]],
    orig_columns: List[str],
) -> Tensor:
    """Convert discretized columns to continuous, ignoring categorical columns.

    If a continuous var was discretized into multiple bins,
    grab the most likely one (max score of the bins) and return the
    mean of the range of that bin.
    """
    # X is Tensor, want to convert to df
    X_disc = pd.DataFrame(tensor_to_numpy(encoded_data))

    # Get the index of the bin with the maximum score for each column group
    col_max_indices = X_disc.groupby(disc_groupby, axis=1).idxmax(
        axis=1, numeric_only=True
    )
    #  offset the indices to 0 so we can directly index into that var's ["bins"]
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
        [X_disc.drop(disc_groupby.keys(), axis=1), continuous_estimates], axis=1
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
    if len(encoded_data.shape) == 3:
        nfeatures = X_cont.shape[-1]
        n_sequence = encoded_data.shape[1]
        X_cont = X_cont.values.reshape((-1, n_sequence, nfeatures))
    else:
        X_cont = X_cont.values

    return torch.tensor(X_cont, device=encoded_data.device).float()


def _nanargmin(arr: np.ndarray, axis: int = 0) -> int:
    try:
        return np.nanargmin(arr, axis)
    except ValueError:
        return np.nan


def get_invert_target_encode_tensor_args(
    encoder_mapping: Dict,
    ordinal_encoder_mapping: Dict,
    mapped_cols: pd.Index,
    orig_cols: pd.Index,
    to_device: torch.device = DEFAULT_DEVICE,
) -> Dict[str, Union[Tensor, List[Tensor]]]:
    args = {
        "mean_encoded_values": [],  # List of tensors
        "col_idxs": [],  # tensor
        "orig_col_idxs": [],  # tensor
        "num_classes": [],  # list
    }
    for coln, col_mapping in encoder_mapping.items():
        args["col_idxs"].append(mapped_cols.get_loc(coln))
        # get first instance of "True" if multicategorical
        matching_cols = orig_cols.str.startswith(coln)
        args["orig_col_idxs"].append(matching_cols.argmax())
        args["num_classes"].append(matching_cols.sum())
        args["mean_encoded_values"].append(
            list_to_tensor(col_mapping.values, dtype=torch.float, device=to_device)
        )
    args["col_idxs"] = list_to_tensor(args["col_idxs"], device=to_device)
    args["orig_col_idxs"] = list_to_tensor(args["orig_col_idxs"], device=to_device)
    # ordinal_values = [
    #     list_to_tensor(list(mapping.keys()), dtype=torch.float)
    #     for mapping in mean_to_ordinal_map["mapping"].values()
    # ]
    args["original_categorical_values"] = [  # list of tensors
        list_to_tensor(info["mapping"].index, dtype=torch.float, device=to_device)
        for info in ordinal_encoder_mapping
    ]
    return args


def invert_target_encoding_tensor_gpu(
    encoded_data: Tensor,
    col_idxs: Tensor,
    orig_col_idxs: Tensor,
    num_classes: List[int],
    mean_encoded_values: List[Tensor],
    original_categorical_values: List[Tensor],
) -> Tensor:
    ordinal_idxs = torch.stack(
        [
            # TODO make this nanargmin
            # get the index of the encoding value that' closest (smallest diff)
            torch.argmin(
                # abs difference
                torch.abs(
                    # difference between each value in the column and the possible encoded values
                    # this will broadcast all the possible encoding values over each value in that column
                    encoded_data[:, col_idx]
                    - mean_encoded_values[i].unsqueeze(1)
                ),
                dim=0,
            )
            for i, col_idx in enumerate(col_idxs)
        ]
    ).T  # returns 1 row per column but we want to keep column format (so transpose)
    # replace the continuous values with the categorcal numerical encoded number
    mapped_to_categorical = []
    for i, cat_vals in enumerate(original_categorical_values):
        # get category num encoding using the index in ordinal_idxs
        col_mapped_to_categorical = cat_vals.index_select(0, ordinal_idxs[:, i])
        # onehot encode multicategorical columns, preserving where there are nans
        if num_classes[i] > 2:
            col_mapped_to_categorical = torch.where(
                # for the column get the row entry where it's nan, and expand that for all expanded columns (len(cat_vals)) since torch.where is elwise
                ~col_mapped_to_categorical.isnan().tile((num_classes[i], 1)).T,
                torch.nn.functional.one_hot(
                    # fill nan with 0 to not create an extra category, we know where it's originally nan so this is ok
                    # we need long for one_hot to work
                    col_mapped_to_categorical.nan_to_num(0).long(),
                    num_classes=num_classes[i],  # enforce the correct number of cols
                ),
                # put nans back into all the cols for the row that was originally nan
                torch.tensor(torch.nan),
            )
        else:  # make bin column be same dimensions as multicat so i can concat them later
            col_mapped_to_categorical = col_mapped_to_categorical.unsqueeze(1)

        mapped_to_categorical.append(col_mapped_to_categorical)
    # drop indices of discretized columns
    keep_columns = torch_set_diff(torch.arange(0, encoded_data.shape[1]), col_idxs)
    # drop discretized_columns and add the (potentially onehot) categoricals back in
    encoded_data = encoded_data.index_select(1, keep_columns)
    # put everything back in order and add in mapped columns
    cols = []
    # these act as indexors into encoded data and mapped data respectively.
    n_ctn_cols = 0
    n_cat_cols = 0
    for i, idx in enumerate(orig_col_idxs):
        # if starts with the categorical itll be [0:0] and nothing will be added
        """
        e.g. i've added 1 ctn col and 3 cat ones, but the next idx of a cat col is 6
            that means that I need to add 6-3=3 ctn cols, starting after the 1 ctn col
        """
        cols.append(encoded_data[:, n_ctn_cols : idx - n_cat_cols])
        cols.append(mapped_to_categorical[i])
        n_ctn_cols = idx - n_cat_cols
        n_cat_cols += num_classes[i]
    cols.append(
        encoded_data[:, n_ctn_cols:]
    )  # this will add nothing if there isn't anything left (n_ctn_cols out of bounds)
    return torch.cat(cols, 1)


def invert_target_encoding_tensor(
    encoded_data: Tensor,
    mean_to_ordinal_map: Dict[str, Union[Callable, Dict[int, Dict[float, int]]]],
    combined_onehot_columns: List[str],
    original_columns: List[str],
    combined_onehots_groupby: Optional[Dict[int, int]] = None,
) -> Tensor:
    encoded_data = tensor_to_numpy(encoded_data)  # needed for inverse transform sklearn

    # this is in collapsed-onehot space
    for col_idx, mapping in mean_to_ordinal_map["mapping"].items():
        mean_encoded_values = np.array(list(mapping.values()))
        ordinal_value = np.array(list(mapping.keys()))
        # get nearest since after autoencoder i wont' have the exact continuous value
        # if there's multiple mins prefers the 1st one (multiple cats with same enc)
        # preserves nans in place
        encoded_data[:, col_idx] = np.array(
            [
                ordinal_value[mapping_idx] if not np.isnan(mapping_idx) else mapping_idx
                for mapping_idx in [
                    _nanargmin(np.abs(mean_encoded_values - v))
                    for v in encoded_data[:, col_idx]
                ]
            ]
        )

    # Need pd if category was string
    encoded_data = pd.DataFrame(encoded_data, columns=combined_onehot_columns)
    # can only un-ordinal encode the ordinally encoded columns
    encoded_data = mean_to_ordinal_map["inverse_transform"](encoded_data)

    # explode columns if onehots were flattened
    if combined_onehots_groupby is not None:
        # reorder to match original
        onehot_groups_idxs = [
            np.nonzero(original_columns.str.startswith(prefix))[0]
            for prefix in combined_onehots_groupby.values()
        ]
        return torch.tensor(
            explode_nans(
                onehot_multicategorical_column(combined_onehots_groupby.values())(
                    encoded_data
                ).reindex(
                    columns=original_columns, fill_value=0
                ),  # will fill 0s for missing cats
                onehot_groups_idxs,
            ).values
        ).float()

    # reorder to match original
    return torch.tensor(encoded_data[original_columns].values).float()
