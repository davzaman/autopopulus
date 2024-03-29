from typing import Callable, List, Union
from pandas import DataFrame, MultiIndex, Series, Index, get_dummies
from numpy import ndarray, nan
from torch import Tensor
import torch
from torch.utils.data import DataLoader, TensorDataset

from autopopulus.data.types import DataT
from autopopulus.data.constants import TIME_LEVEL


def explode_nans(X: DataFrame, onehot_groups_idxs: List[List[int]]) -> DataFrame:
    """
    For onehot groups if 1 of the values is nan, all of the category values for that group should be nan for that sample.
    """
    for onehot_group in onehot_groups_idxs:
        # only .loc doesn't return a copy so I can set the value
        # but that requires the col names, not indices
        X.loc[X.iloc[:, onehot_group].isna().any(axis=1), X.columns[onehot_group]] = nan
    return X


def enforce_tensor(*df: DataT, enforce_numeric: bool = True) -> Tensor:
    def cases(data: DataT):
        if data is None:
            return None
        elif isinstance(data, Tensor):
            return data
        else:
            return torch.from_numpy(enforce_numpy(data, enforce_numeric))

    return (cases(data) for data in df)


def enforce_numpy(df: DataT, enforce_numeric: bool = True) -> ndarray:
    # enforce numpy is numeric with df*1 (fixes bools)
    if enforce_numeric:
        df = df * 1
    return df.values if isinstance(df, DataFrame) else df


def get_samples_from_index(index: Index) -> ndarray:
    # ignore time dimension if longitudinal portion of data
    is_multiindex = isinstance(index, MultiIndex)
    # level = PATIENT_ID if is_multiindex else None
    # sample_ids = X.index.unique(level).values
    if is_multiindex and TIME_LEVEL in index.names:
        return index.droplevel(TIME_LEVEL).values
    else:
        return index


def regex_safe_colname(coln: str) -> str:
    """
    When looking up pd.str.contains(str) the passed str has to be regex safe.
    If a column has regex keywords in it like `+` then there's a problem.
    e.g., 'IONIZED CA++,CORRECTED_min' from crrt dataset.
    Similar with parenthesis.
    Chaining is fastest: https://stackoverflow.com/a/27086669/1888794
    """
    return coln.replace("+", "\+").replace("(", "\(").replace(")", "\)")


def onehot_multicategorical_column(
    prefixes: Union[str, List[str]],
) -> Callable[[DataFrame], DataFrame]:
    """
    Onehot encodes columns with prefix_value.
    e.g. fries [l, s, m, m, l] -> [fries_l, fries_s, fries_m]
    Can use with hypothesis testing and in the code.
    Will retain nans.
    ORDER NOTE: get_dummies adds the onehot cols at the end.
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]

    def integrate_onehots(df: DataFrame) -> DataFrame:
        if df[prefixes].empty:
            return df
        dummies = get_dummies(df, columns=prefixes, prefix=prefixes, dummy_na=True)
        # Retain nans
        nan_cols = [f"{prefix}_nan" for prefix in prefixes]
        row_mask = dummies[nan_cols].astype(bool)
        for i, prefix in enumerate(prefixes):
            col_mask = dummies.columns.str.startswith(prefix)
            dummies.loc[row_mask.iloc[:, i], col_mask] = nan
        dummies = dummies.drop(nan_cols, axis=1)
        return dummies

    return integrate_onehots


def get_dataloader(
    X: Union[DataFrame, ndarray],
    y: Union[DataFrame, ndarray],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Pytorch modules require DataLoaders for train/val/test,
    but we start with a df or ndarray.
    Used for passing data to autoencoders and classifiers alike (pytorch)."""
    if isinstance(X, DataFrame):
        X = X.values
    if isinstance(y, DataFrame) or isinstance(y, Series):
        y = y.values
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


class concat_dataloaders:
    """Class to concatenate multiple dataloaders.
    Ref: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/35
    """

    def __init__(self, dataloaders: List[DataLoader]):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter))  # may raise StopIteration
        return tuple(out)
