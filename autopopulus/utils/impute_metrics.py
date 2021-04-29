from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from utils.utils import col_prefix, maskfill_0, masked_mean

PdNpOrTensor = Union[pd.DataFrame, np.ndarray, Tensor]

EPSILON = 1e-10


def force_np_if_pd(dfs: List[PdNpOrTensor]) -> Tuple[Union[np.ndarray, Tensor], ...]:
    # Force to np is pandas, else keep tensor/numpy
    return (*(df.values if isinstance(df, pd.DataFrame) else df for df in dfs),)


def MAAPE(
    predicted: PdNpOrTensor,
    target: PdNpOrTensor,
    mask: Optional[Tensor] = None,
    epsilon: int = EPSILON,
    reduction: str = "mean",
) -> Tensor:
    """
    Mean Arctangent Absolute Percentage Error
    MAPE but works around the divide by 0 issue.
    Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
    Note: result is NOT multiplied by 100
    """
    assert predicted.shape == target.shape
    predicted, target, mask = force_np_if_pd([predicted, target, mask])

    predicted = maskfill_0(predicted, mask)
    target = maskfill_0(target, mask)

    if isinstance(predicted, np.ndarray):
        row_maape = np.arctan(np.abs((target - predicted) / (target + epsilon)))
    else:
        row_maape = torch.atan(torch.abs((target - predicted) / (target + epsilon)))

    if reduction == "none":
        return masked_mean(row_maape, mask)
    if reduction == "sum":
        return masked_mean(row_maape, mask).sum()
    return masked_mean(row_maape, mask).mean()


def RMSE(
    predicted: PdNpOrTensor,
    target: PdNpOrTensor,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Should work both for Tensor and np.ndarray."""
    assert predicted.shape == target.shape
    predicted, target, mask = force_np_if_pd([predicted, target, mask])

    predicted = maskfill_0(predicted, mask)
    target = maskfill_0(target, mask)

    if reduction == "none":
        return masked_mean((predicted - target) ** 2, mask) ** 0.5
    if reduction == "sum":
        return masked_mean((predicted - target) ** 2, mask).sum() ** 0.5
    return masked_mean((predicted - target) ** 2, mask).mean() ** 0.5


def AccuracyPerBin(
    predicted: Tensor,
    target: Tensor,
    discretized_col_names: List[str],
    orig_col_names: List[str],
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Dict[str, np.ndarray]:
    """For APnew only: log accuracy over number of bins correctly inferred."""
    predicted = pd.DataFrame(
        # torch.sigmoid(predicted).cpu().numpy(), columns=discretized_col_names
        predicted.cpu().numpy(),
        columns=discretized_col_names,
    )
    target = pd.DataFrame(target.cpu().numpy(), columns=discretized_col_names)
    # Get colname of the bin with the maximum score
    coln_maxes = get_feature_max_bins(predicted)
    true_coln_maxes = get_feature_max_bins(target)
    # accuracy_per_bin = coln_maxes.eq(true_coln_maxes).sum() / coln_maxes.shape[0]

    if mask is not None:
        # if the mask is for discretized, otherwise assume mask is undiscretized
        if mask.shape[1] == len(discretized_col_names):
            mask = pd.DataFrame(mask.cpu().numpy(), columns=discretized_col_names)
            # gets the mask for the feature: just pick the first
            # This assumes the mask will be the same for all bins under the same var
            mask = mask.groupby(col_prefix, axis=1).first()
        else:
            mask = pd.DataFrame(mask.cpu().numpy(), columns=orig_col_names)
            # still need to do this for one-hot
            mask = mask.groupby(col_prefix, axis=1).first()
        # only compute on columns where data was missing
        if reduction == "none":
            return masked_mean(coln_maxes.eq(true_coln_maxes) & mask, mask)
        if reduction == "sum":
            return masked_mean(coln_maxes.eq(true_coln_maxes) & mask, mask).sum()
        return masked_mean(coln_maxes.eq(true_coln_maxes) & mask, mask).mean()
    if reduction == "none":
        return coln_maxes.eq(true_coln_maxes).mean()
    if reduction == "sum":
        return coln_maxes.eq(true_coln_maxes).mean().sum()
    return coln_maxes.eq(true_coln_maxes).mean().mean()


def get_feature_max_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Gets the name of the bin that's most likely. Works for onehot, binary, and originally continuous (now discretized/onehot) vars.
    Discretized/onehot vars are grouped by the column name prefix.
    Binary vars are just thresholded.
    """
    # group by column name prefixes.
    return df.groupby(col_prefix, axis=1).apply(
        # idxmax: get the column with the max bin value
        lambda group: group.idxmax(axis=1)
        if group.shape[1] > 1
        # binary var if it's just one column: just threshold for the values, squeeze into series
        else (group > 0.5).astype(int).squeeze()
    )
