from re import I
from typing import Any, Callable, Iterable, List, Optional
import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric
from math import pi

from autopopulus.data.constants import PAD_VALUE


EPSILON = 1e-10


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
            np_data.append(tensor.numpy())
        else:
            np_data.append(tensor)
    return tuple(np_data)


def format_tensor(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
    """
    If you actually passed gradient-tracking Tensors to a Metric, there will be
    a huge memory leak, because it will prevent garbage collection for the computation
    graph. This method ensures the tensors are detached.
    Ref: https://github.com/allenai/allennlp/blob/9f879b0964e035db711e018e8099863128b4a46f/allennlp/training/metrics/metric.py#L45

    Also enforce float.
    """
    # Check if it's actually a tensor in case something else was passed.
    return (x.detach().long() if isinstance(x, torch.Tensor) else x for x in tensors)


class MAAPEMetric(Metric):
    """
    Element-wise MAAPE.
    This might be useless as the feature-map inversion cannot be done on the gpu especially since we need to reorder stuff as pandas df.
    https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    https://github.com/Lightning-AI/metrics/tree/master/src/torchmetrics/regression.mape.py
    https://github.com/allenai/allennlp/tree/main/allennlp/training/metrics
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self, epsilon: float = EPSILON, scale_to_01: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.scale_to_01 = scale_to_01
        self.add_state("sum_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        row_maape = torch.arctan(torch.abs((target - preds) / (target + self.epsilon)))
        if mask is not None:
            row_maape *= ~mask
            count = torch.sum(~mask)
        else:
            count = target.numel()
        self.sum_errors += torch.sum(row_maape)
        self.total += count

    def compute(self) -> float:
        maape = self.sum_errors / self.total
        if self.scale_to_01:  # range [0, pi/2] scale to [0, 1]
            maape *= 2 / pi
        return maape


class RMSEMetric(Metric):
    """
    Element-wise RMSE.
    This might be useless as the feature-map inversion cannot be done on the gpu especially since we need to reorder stuff as pandas df.
    https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
    https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/regression/mse.py
    https://github.com/allenai/allennlp/tree/main/allennlp/training/metrics
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        preds, target = format_tensor(preds, target)
        assert preds.shape == target.shape

        squared_error = torch.pow(preds - target, 2)
        if mask is not None:
            squared_error *= ~mask
            count = torch.sum(~mask)
        else:
            count = torch.tensor(target.numel(), device=self.device)
        self.sum_errors += torch.sum(squared_error)
        self.total += count

    def compute(self) -> float:
        return torch.sqrt(self.sum_errors / self.total)


class AccuracyMetric(Metric):
    """
    Element-wise accuracy.
    https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/classification/accuracy.py
    https://github.com/allenai/allennlp/blob/main/allennlp/training/metrics/categorical_accuracy.py
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self, bin_cols_idx: torch.Tensor, onehot_cols_idx: torch.Tensor, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.bin_cols_idx = bin_cols_idx
        self.onehot_cols_idx = onehot_cols_idx
        self.add_state("num_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        # preds, target, bin_cols_idx, onehot_cols_idx, mask = self._input_format(
        #     preds, target, bin_cols_idx, onehot_cols_idx, mask
        # )
        assert preds.shape == target.shape

        predicted_cats = get_categories(preds, self.bin_cols_idx, self.onehot_cols_idx)
        target_cats = get_categories(target, self.bin_cols_idx, self.onehot_cols_idx)

        correct = predicted_cats.eq(target_cats).to(int)
        if mask is not None:
            mask = get_categories(mask, self.bin_cols_idx, self.onehot_cols_idx)
            correct *= ~mask
            count = torch.sum(~mask)
        else:
            count = target_cats.numel()
        self.num_correct += torch.sum(correct)
        self.total += count

    def compute(self) -> float:
        return self.num_correct / self.total


def CWMAAPE(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = EPSILON,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean Arctangent Absolute Percentage Error reduced column-wise.
    MAPE but works around the divide by 0 issue.
    Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
    Note: result is NOT multiplied by 100
    """
    assert predicted.shape == target.shape
    predicted, target, mask = force_np(predicted, target, mask)

    predicted = np.ma.masked_array(predicted, mask)
    target = np.ma.masked_array(target, mask)

    if isinstance(predicted, np.ndarray):
        row_maape = np.arctan(np.abs((target - predicted) / (target + epsilon)))

    no_reduce = np.ma.mean(row_maape, axis=0)
    if reduction == "none":
        return no_reduce
    if reduction == "sum":
        return no_reduce.sum()
    return no_reduce.mean()


def CWRMSE(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    normalize: Optional[str] = None,
) -> torch.Tensor:
    """Column-wise reduction. Should work both for torch.Tensor and np.ndarray."""
    assert predicted.shape == target.shape
    predicted, target, mask = force_np(predicted, target, mask)

    predicted = np.ma.masked_array(predicted, mask)
    target = np.ma.masked_array(target, mask)

    if normalize == "mean":
        denom = np.ma.mean(target, axis=1)
    elif normalize == "std":
        denom = np.ma.std(target, axis=1)
    elif normalize == "range":
        denom = np.ma.max(target, axis=1) - np.ma.min(target, axis=1)
    else:
        denom = 1
    no_reduce = np.ma.mean((predicted - target) ** 2, axis=0)
    if reduction == "none":
        return (no_reduce**0.5) / denom
    if reduction == "sum":
        return (no_reduce.sum() ** 0.5) / denom
    return (no_reduce.mean() ** 0.5) / denom


def categorical_accuracy(
    bin_col_idxs: torch.Tensor, onehot_cols_idx: torch.Tensor
) -> Callable:
    def accuracy_fn(
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Log accuracy over all categorical variables."""
        # TODO: accuracy per bin per time point for longitudinal?
        # Get colname of the bin with the maximum score
        predicted_cats = get_categories(predicted, bin_col_idxs, onehot_cols_idx)
        target_cats = get_categories(target, bin_col_idxs, onehot_cols_idx)

        if mask is not None:
            mask = get_categories(mask, bin_col_idxs, onehot_cols_idx)
            predicted_cats, target_cats, mask = force_np(
                predicted_cats, target_cats, mask
            )
            predicted_cats = np.ma.masked_array(predicted_cats, mask)
            target_cats = np.ma.masked_array(target_cats, mask)
            accuracy_per_bin = np.ma.mean((predicted_cats == target_cats), axis=0)
        else:  # no mask
            accuracy_per_bin = predicted_cats.eq(target_cats).to(float).mean(axis=0)

        if reduction == "none":
            return accuracy_per_bin
        if reduction == "sum":
            return accuracy_per_bin.sum()
        return accuracy_per_bin.mean()

    return accuracy_fn


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
