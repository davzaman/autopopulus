from re import I
from deprecated import deprecated
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from lightning_utilities import apply_to_collection
import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric
from math import pi

from autopopulus.data.constants import PAD_VALUE
from autopopulus.data.utils import enforce_numpy, enforce_tensor


EPSILON = 1e-10

"""
NOTE: for all metrics, I have checked with my test suite that
setting `full_state_update` to False and True both gives the same result,
therefore I am setting them all to False for optimization.
https://torchmetrics.readthedocs.io/en/latest/pages/implement.html#internal-implementation-details
If the metrics change they might have to be checked again.
"""


class MAAPEMetric(Metric):
    """
    MAAPE.
    EW: sum(sum(arctan(abs((true - pred) / true))))/(nrows * ncols)
    CW: mean(mean(arctan(abs((true - pred) / true))))
    Not the same when there's missingness indicators.
    This might be useless as the feature-map inversion cannot be done on the gpu especially since we need to reorder stuff as pandas df.
    https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    https://github.com/Lightning-AI/metrics/tree/master/src/torchmetrics/regression.mape.py
    https://github.com/allenai/allennlp/tree/main/allennlp/training/metrics
    https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        epsilon: float = EPSILON,
        scale_to_01: bool = True,
        columnwise: bool = False,
        nfeatures: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert (
            not columnwise or nfeatures is not None
        ), "If columnwise, nfeatures cannot be None."
        self.columnwise = columnwise
        shape = (nfeatures,) if columnwise else (1,)
        self.epsilon = epsilon
        self.scale_to_01 = scale_to_01
        self.add_state(
            "sum_errors",
            default=torch.full(shape, torch.tensor(0.0)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.full(shape, torch.tensor(0)), dist_reduce_fx="sum"
        )

    def sum_fn(self, data: torch.Tensor) -> torch.Tensor:
        # 0 -> sum across rows for a per-column sum, none -> all dims reduced
        return torch.sum(data, dim=0) if self.columnwise else torch.sum(data)

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        missing_indicators: Optional[torch.BoolTensor] = None,
    ):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        assert not (
            any(
                [
                    torch.isnan(data).any()
                    for data in [preds, target, missing_indicators]
                    if data is not None
                ]
            )
        )

        row_maape = torch.arctan(torch.abs((target - preds) / (target + self.epsilon)))
        if missing_indicators is not None:
            row_maape *= missing_indicators
            count = self.sum_fn(missing_indicators)
        else:
            count = torch.tensor(  # num rows if columnwise
                target.size()[0] if self.columnwise else target.numel(),
                device=self.device,
            )

        self.sum_errors += self.sum_fn(row_maape)
        self.total += count

    def compute(self) -> float:
        maape = self.sum_errors / self.total
        """
        Fill with 0 where numerator and denominator are 0.
            With missing_indicator, the total (denominator) might be 0.
            When there's no missing values in a column, we get div by 0 -> nan
            Don't fill nan to not confound where nans are leaking in the data.
            (vs nan caused by 0/0 AKA where there's no missing values).
        This also works for EW when no data is missing.
        """
        maape = maape.where(
            ~((self.sum_errors == 0) & (self.total == 0)),
            torch.tensor(0.0, device=self.sum_errors.device),
        )
        if self.scale_to_01:  # range [0, pi/2] scale to [0, 1]
            maape *= 2 / pi
        if self.columnwise:
            return torch.mean(maape)
        return maape


class RMSEMetric(Metric):
    """
    RMSE.
    Element-wise is computationally different from column-wise (order of sqrt and sum matters).
        EW: sqrt(sum(sum((true - pred)**2)) / (nrows * ncols))
        CW: mean(sqrt(mean((true - pred)**2)))
    This might be useless as the feature-map inversion cannot be done on the gpu especially since we need to reorder stuff as pandas df.
    https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/regression/mse.py
    https://github.com/allenai/allennlp/tree/main/allennlp/training/metrics
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self, columnwise: bool = False, nfeatures: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert (
            not columnwise or nfeatures is not None
        ), "If columnwise, nfeatures cannot be None."
        shape = (nfeatures,) if columnwise else (1,)
        self.columnwise = columnwise
        self.add_state(
            "sum_errors",
            default=torch.full(shape, torch.tensor(0.0)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.full(shape, torch.tensor(0)), dist_reduce_fx="sum"
        )

    def sum_fn(self, data: torch.Tensor) -> torch.Tensor:
        # 0 -> sum across rows for a per-column sum, none -> all dims reduced
        return torch.sum(data, dim=0) if self.columnwise else torch.sum(data)

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        missing_indicators: Optional[torch.BoolTensor] = None,
    ):
        assert preds.shape == target.shape
        assert not (
            any(
                [
                    torch.isnan(data).any()
                    for data in [preds, target, missing_indicators]
                    if data is not None
                ]
            )
        )

        squared_error = torch.pow(preds - target, 2)
        if missing_indicators is not None:
            squared_error *= missing_indicators
            count = self.sum_fn(missing_indicators)
        else:
            count = torch.tensor(  # num rows if columnwise
                target.size()[0] if self.columnwise else target.numel(),
                device=self.device,
            )

        self.sum_errors += self.sum_fn(squared_error)
        self.total += count

    def compute(self) -> float:
        rmse = torch.sqrt(self.sum_errors / self.total)
        """
        Fill with 0 where numerator and denominator are 0.
            With missing_indicator, the total (denominator) might be 0.
            When there's no missing values in a column, we get div by 0 -> nan
            Don't fill nan to not confound where nans are leaking in the data.
            (vs nan caused by 0/0 AKA where there's no missing values).
        This also works for EW when no data is missing.
        """
        rmse = rmse.where(
            ~((self.sum_errors == 0) & (self.total == 0)),
            torch.tensor(0.0, device=self.sum_errors.device),
        )
        if self.columnwise:
            return torch.mean(rmse)
        return rmse


class AccuracyMetric(Metric):
    """
    Categorical accuracy.
    CW != EW when there's a mask.
    https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/classification/accuracy.py
    https://github.com/allenai/allennlp/blob/main/allennlp/training/metrics/categorical_accuracy.py
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        bin_cols_idx: torch.Tensor,
        onehot_cols_idx: torch.Tensor,
        columnwise: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # number of cat cols is the # of bin cols, and the # of onehot_col groups
        nfeatures = len(bin_cols_idx) + len(onehot_cols_idx)
        assert (
            not columnwise or nfeatures is not None
        ), "If columnwise, nfeatures cannot be None."
        self.columnwise = columnwise
        shape = (nfeatures,) if columnwise else (1,)
        self.bin_cols_idx = bin_cols_idx
        self.onehot_cols_idx = onehot_cols_idx
        self.add_state(
            "num_correct", torch.full(shape, torch.tensor(0)), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.full(shape, torch.tensor(0)), dist_reduce_fx="sum"
        )

    def sum_fn(self, data: torch.Tensor) -> torch.Tensor:
        # 0 -> sum across rows for a per-column sum, none -> all dims reduced
        return torch.sum(data, dim=0) if self.columnwise else torch.sum(data)

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        missing_indicators: Optional[torch.BoolTensor] = None,
    ):
        assert preds.shape == target.shape
        assert not (
            any(
                [
                    torch.isnan(data).any()
                    for data in [preds, target, missing_indicators]
                    if data is not None
                ]
            )
        )

        predicted_cats = get_categories(preds, self.bin_cols_idx, self.onehot_cols_idx)
        target_cats = get_categories(target, self.bin_cols_idx, self.onehot_cols_idx)

        correct = predicted_cats.eq(target_cats).to(int)
        if missing_indicators is not None:
            missing_indicators = get_categories(
                missing_indicators, self.bin_cols_idx, self.onehot_cols_idx
            )
            correct *= missing_indicators
            count = self.sum_fn(missing_indicators)
        else:
            count = torch.tensor(
                # num rows if columnwise
                target_cats.size()[0] if self.columnwise else target_cats.numel(),
                device=self.device,
            )
        self.num_correct += self.sum_fn(correct)
        self.total += count

    def compute(self) -> float:
        acc = self.num_correct / self.total
        """
        Fill with 1 (no error) where numerator and denominator are 0.
            With missing_indicator, the total (denominator) might be 0.
            When there's no missing values in a column, we get div by 0 -> nan
            Don't fill nan to not confound where nans are leaking in the data.
            (vs nan caused by 0/0 AKA where there's no missing values).
        This also works for EW when no data is missing.
        """
        acc = acc.where(
            ~((self.num_correct == 0) & (self.total == 0)),
            torch.tensor(1.0, device=self.num_correct.device),
        )
        if self.columnwise:
            return torch.mean(acc)
        return acc


def universal_metric(metric: Metric) -> Callable:
    """Functional version of the torchmetric so I can use it on sklearn models."""

    def apply_metric(
        predicted: Union[pd.DataFrame, torch.Tensor],
        target: Union[pd.DataFrame, torch.Tensor],
        missing_indictors: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
    ) -> float:
        metric.update(*enforce_tensor(predicted, target, missing_indictors))
        value = metric.compute()
        metric.reset()
        return value

    return apply_metric


def get_categories(
    data: torch.Tensor, bin_col_idxs: torch.Tensor, onehot_cols_idx: torch.Tensor
) -> torch.Tensor:
    """
    For binary and onehot groups get the column name of the bin with the maximum score.
    """
    to_stack = [data[:, bin_col_idxs].T]
    if len(onehot_cols_idx) > 0:
        if data.dtype == torch.bool or data.dtype == bool:  # if data is the mask
            # This assumes the mask will be the same for all bins under the same var
            fn = lambda idxs: data[:, idxs].all(axis=1)
        else:  # data is actual data not a mask
            fn = lambda idxs: torch.argmax(data[:, idxs], axis=1)

        # if this is empty the stack will complain
        to_stack.append(
            torch.stack(
                [
                    fn(onehot_group[onehot_group != PAD_VALUE])
                    for onehot_group in onehot_cols_idx
                ]
            )
        )
    # everything will concatenate on rows
    return torch.cat(to_stack, axis=0).T  # flip back so it's N x F


@deprecated
def force_np(*tensors: torch.Tensor) -> np.ndarray:
    """
    Force to np and force on cpu if computing metrics
    (after loss so doesn't need to be on GPU for backward).
    All post-training metrics should do this.
    """
    np_data = []
    for tensor in tensors:
        if isinstance(tensor, pd.DataFrame):
            np_data.append(tensor.values)
        elif isinstance(tensor, torch.Tensor):
            if tensor.device != torch.device("cpu"):
                tensor = tensor.cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            np_data.append(tensor.numpy())
        else:
            np_data.append(tensor)
    return tuple(np_data)


@deprecated
def CWRMSE(
    predicted: torch.Tensor,
    target: torch.Tensor,
    missing_indicators: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    normalize: Optional[str] = None,
) -> torch.Tensor:
    """
    DEPRECATED: use RMSEMetric with columnwise set.
    Keeping this for reference of how to do it with normalization if wanted.
    Column-wise reduction. Should work both for torch.Tensor and np.ndarray.
    """
    assert predicted.shape == target.shape
    predicted, target, missing_indicators = force_np(
        predicted, target, missing_indicators
    )
    mask_out_observed = (
        ~missing_indicators if missing_indicators is not None else missing_indicators
    )
    predicted = np.ma.masked_array(predicted, mask_out_observed)
    target = np.ma.masked_array(target, mask_out_observed)

    if normalize == "mean":
        denom = np.ma.mean(target, axis=1)
    elif normalize == "std":
        denom = np.ma.std(target, axis=1)
    elif normalize == "range":
        denom = np.ma.max(target, axis=1) - np.ma.min(target, axis=1)
    else:
        denom = 1
    # RMSE per column: square root of the mean of squared diffs in that column
    no_reduce = np.ma.mean((predicted - target) ** 2, axis=0) ** 0.5
    if reduction == "none":
        return no_reduce / denom
    if reduction == "sum":
        return no_reduce.sum() / denom
    return no_reduce.mean() / denom
