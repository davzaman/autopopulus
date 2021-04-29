from argparse import ArgumentParser, Namespace, Action
import re
from warnings import warn
from collections import ChainMap
from typing import List, Optional, Union
import os
import pandas as pd
import numpy as np
import torch
from torch import Tensor


def should_ampute(args: Namespace) -> bool:
    return (
        "percent_missing" in args
        and "missingness_mechanism" in args
        and args.method != "none"
    )


def seed_everything(seed: int):
    """Sets seeds and also makes cuda deterministic for pytorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v, default: bool = False):
    """Convert argparse boolean to true boolean.
    Ref: https://stackoverflow.com/a/43357954/1888794"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        warn(
            f"Boolean value expected. Passed {v}, which is of type {type(v)} and wasn't recognized. Using default value {default}."
        )
        return default


def YAMLStringListToList(convert: type = str, choices: Optional[List[str]] = None):
    class ConvertToList(Action):
        """Takes a comma separated list (no spaces) from command line and parses into list of some type (Default str)."""

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: str,
            option_string: Optional[str] = None,
        ):
            if choices:
                values = [convert(x) for x in values.split(",") if x in choices]
            else:
                values = [convert(x) for x in values.split(",")]
            setattr(namespace, self.dest, values)

    return ConvertToList


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


def col_prefix(coln: str) -> str:
    """Group features by prefix of varname. Ignores binary vars."""
    prefix, delim, suffix = coln.rpartition("_")
    if suffix == "onehot":  # If onehot: var_category_onehot: get the prefix ("var")
        return coln.rpartition("_")[0]

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
    return (
        np.where(mask, data, 0)
        if isinstance(data, np.ndarray)
        # mask, data, torch.tensor(0.0, device=data.device, dtype=torch.double)
        else torch.where(mask, data, torch.tensor(0.0, device=data.device))
    )


# For debugging
def parse_yaml_args(obj, base_prefix=""):
    """
    The results of loading the guild yml file is a list of dictionaries.
    Each item in the list is an object in the file represented as a dict.
        -config: common-flags will become {"config": "common-flags", ...}
        -flags: k: v, k: v, ... will be {"flags": {k: v, k: v}}
        This all goes into the same dict object.
    For multiple config groupings we'll have different objects.
    """
    # Grabs the config objects from the yaml file and then merges them.
    # the if: grab only config objects.
    # the chainmap will merge all the flag dictionaries from each group.
    #   if it encounters the same name later, it keeps the first one
    flags = dict(
        ChainMap(*[flag_group["flags"] for flag_group in obj if "config" in flag_group])
    )

    for k, v in flags.items():
        # ingore the $includes, because bash will think it's a var
        if k != "$include":
            # if the yaml has a list, just pick the first one for testing purposes
            if isinstance(v, list):
                v = v[0]
            # print('{}="{}"'.format(k.replace("-", "_"), v))
    return {k: v[0] if isinstance(v, list) else v for k, v in flags.items()}
