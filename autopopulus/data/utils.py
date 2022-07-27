from typing import List, Union
from pandas import DataFrame, Series
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(
    X: Union[DataFrame, ndarray],
    y: Union[DataFrame, ndarray],
    batch_size: int,
    num_gpus: int,
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
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_gpus * 4
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
