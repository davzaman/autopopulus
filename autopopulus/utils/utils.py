from argparse import Namespace
import os
from functools import reduce
from inspect import getmembers, isfunction
import operator
from typing import Dict, List, Optional, Union
import re

import pandas as pd
import numpy as np

import torch
from torch import Tensor
import pytorch_lightning as pl


def should_ampute(args: Namespace) -> bool:
    return (
        "percent_missing" in args
        and "missingness_mechanism" in args
        and args.method != "none"
    )


def seed_everything(seed: int):
    """Sets seeds and also makes cuda deterministic for pytorch."""
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """
    pl.seed_everything(seed)
    # RNN/LSTM determininsm error with cuda 10.1/10.2
    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def get_module_function_names(module) -> List[str]:
    return [
        name
        for name, fn in getmembers(module, isfunction)
        # ignore imported methods
        if fn.__module__ == module.__name__
    ]


class ChainedAssignent:
    """For pandas: to silence the settingwithcopywarning.
    Use as context:
    >>> with ChainedAssignment():
    >>>     df["col"] = value
    Ref: https://stackoverflow.com/a/53954986/1888794
    """

    def __init__(self, chained=None):
        acceptable = [None, "warn", "raise"]
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw


# TODO: do i need this anymore?
def col_prefix(coln: str) -> str:
    """Group features by prefix of varname. Ignores binary vars."""
    prefix, delim, suffix = coln.rpartition("_")
    if suffix == "onehot":  # If onehot: var_category_onehot: get the prefix ("var")
        return prefix

    def originally_ctn_col(suffix: str) -> bool:
        # regex matching col names with ranges in them (discretized)
        return re.search(r"\s-\s", suffix)

    return (
        prefix if originally_ctn_col(suffix) else coln
    )  # don't split single binary vars


def nanmean(v: Union[np.ndarray, Tensor], *args, inplace=False, **kwargs) -> float:
    """Equivalent of np.nanmean which ignores nans to compute the mean.
    Note: this will fail if the whole column is NaN.
    Ref: https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    """
    if isinstance(v, Tensor):
        if not inplace:
            v = v.clone()
        denom = (~torch.isnan(v)).float().sum(*args, **kwargs)
        return v.nan_to_num(nan=0).sum(*args, **kwargs) / denom
    else:
        denom = (~np.isnan(v)).sum(*args, **kwargs)
        return np.nan_to_num(v, nan=0).sum(*args, **kwargs) / denom


def div0(
    a: Union[np.ndarray, Tensor], b: Union[np.ndarray, Tensor]
) -> Union[np.ndarray, Tensor]:
    """Division, but drop the values where the denominator was 0 on purpose (to not be included in subsequent mean).
    Ref: https://stackoverflow.com/a/35696047/1888794"""
    valid_denom = b != 0
    if isinstance(a, Tensor):
        # put a 1 where denom is 0, will be ignored later
        res = a / b.where(valid_denom, torch.ones_like(b))
    else:
        # numpy equivalent
        res = a / np.where(valid_denom, b, np.ones_like(b))
    # remove where denom was 0
    return res[valid_denom]


def masked_mean(
    data: Union[np.ndarray, Tensor],
    mask: Optional[Tensor] = None,
) -> Tensor:
    if mask is not None:
        nelements = mask.sum(axis=1)
        return div0(data.sum(axis=1), nelements)
    return data.mean()


def maskfill_0(
    data: Union[np.ndarray, Tensor], mask: Optional[Tensor]
) -> Union[np.ndarray, Tensor]:
    """Sets values at the mask to 0. Does nothing if mask is None."""
    if mask is None:
        return data
    # TODO add checks that the dimensions match or are broadcastable
    return (
        np.where(mask, data, 0)
        if isinstance(data, np.ndarray)
        # IF this complains remain dtype arg
        else torch.where(
            mask, data, torch.tensor(0.0, device=data.device, dtype=data.dtype)
        )
    )


def flatten_groupby(groupby: Dict[str, Dict[int, str]]) -> Dict[str, Dict[int, str]]:
    """
    Groupby has a dict of indices to original column names for different groups:
    - multicategorical
    - binary
    - discretized continuous columns: {data, ground_truth}

    Here we want to flatten this to have one flattened version to just list them all instead of separated.
    One flattened version for data and one for ground truth (if GT is discretized differently from data).
    """

    def get_nested_index(group_name: str, data_name: str):
        # Drill into ["data"] or ["ground_truth"] if unpacking the discretized groupby subdict
        return (
            [group_name, data_name]
            if group_name == "discretized_ctn_cols"
            else [group_name]
        )

    return {
        data_name: {
            index: group_id
            # https://stackoverflow.com/a/14692747/1888794
            # This weird way is so that we can index nested keys at once
            for group_name in groupby.keys()
            for index, group_id in reduce(
                operator.getitem,
                get_nested_index(group_name, data_name),
                groupby,
            ).items()
        }
        for data_name in ["data", "ground_truth"]
    }
