import inspect
from typing import List, Optional, Tuple, Dict, Union
from argparse import ArgumentParser, Namespace
from collections import Counter
from warnings import warn

import pandas as pd
import numpy as np
from pytorch_lightning.core.datamodule import LightningDataModule

#### Pytorch ####
import torch
from torch.utils.data import DataLoader, TensorDataset

#### Scitkit Learn ####
from sklearn.model_selection import train_test_split

#### Local module ####
from data.ckd import CTN_ENTRY_COLS, DATA_PATH, ONEHOT_PREFIXES, TIME_ZERO_COLS
from data.covid_ckd import (
    COVID_CTN_COLS,
    COVID_DATA_PATH,
    COVID_ONEHOT_PREFIXES,
    DEFAULT_COVID_TARGET,
)
from data.covid_ckd import (
    load_features_and_labels as covid_load_features_and_labels,
)
from data.ckd import load_features_and_labels
from data.transforms import (
    ampute,
    discretize,
    scale,
    uniform_prob_across_nan,
)
from data.ckd import get_subgroup
from utils.utils import str2bool


def load_data(
    args: Namespace,
) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]]:
    """Master data loading method.
    Create subfunction per dataset to handle each one's special cases separately.
    Mimic is deprecated.
    """
    if args.dataset == "cure_ckd":
        return load_ckd_data(args)
    elif args.dataset == "covid_ckd":
        return load_covid_ckd_data(args)


def get_ctn_cols(dataset: str) -> List[str]:
    if dataset == "cure_ckd":
        ctn_cols = CTN_ENTRY_COLS + TIME_ZERO_COLS
    if dataset == "covid_ckd":
        ctn_cols = COVID_CTN_COLS
    elif dataset == "mimic3":
        warn(DeprecationWarning)
        return None
        # ctn_cols = generate_agg_col_name(MIMIC_CTN_COLS)
    return ctn_cols


def get_onehot_prefix_names(dataset: str) -> List[str]:
    if dataset == "cure_ckd":
        prefixes = ONEHOT_PREFIXES
    if dataset == "covid_ckd":
        prefixes = COVID_ONEHOT_PREFIXES
    elif dataset == "mimic3":
        warn(DeprecationWarning)
        return None
        # prefixes = generate_agg_col_name(MIMIC_CTN_COLS)
    return prefixes


def load_ckd_data(
    args: Namespace,
) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]]:
    """Load the ckd data as train, val, test, and ground truth."""
    subgroup = get_subgroup(args.cohort, args.site_source)
    X, y = load_features_and_labels(DATA_PATH, target=args.target, subgroup=subgroup)
    # return common_ckd_data_load(args, X, y)
    return X, y


def load_covid_ckd_data(
    args: Namespace,
) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]]:
    """Load the covid ckd data."""
    X, y = covid_load_features_and_labels(
        COVID_DATA_PATH, DEFAULT_COVID_TARGET, args.covid_site_source
    )
    # return common_ckd_data_load(args, X, y)
    return X, y


def get_dataloader(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, np.ndarray],
    batch_size: int,
    num_gpus: int,
) -> DataLoader:
    """Pytorch modules require DataLoaders for train/val/test,
    but we start with a df or ndarray.
    Used for passing data to autoencoders and classifiers alike (pytorch)."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_gpus * 4
    )
    return loader


class CommonDataModule(LightningDataModule):
    """Data loader for AEImputer Works with pandas/numpy data.
    Assumes data has been one-hot encoded.
    IMPORTANT: Assumes (one-hot encoded) categorical vars all share the same unique prefix.
    Also assumes that if categorical var is missing then the nan is propagated for all of the one-hot features.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seed: int,
        val_test_size: float,
        test_size: float,
        batch_size: int,
        num_gpus: int,
        fully_observed: bool = False,
        scale: bool = False,
        ampute: bool = False,
        discretize: bool = False,
        uniform_prob: bool = False,
        ctn_columns: Optional[List[str]] = None,
        onehot_prefix_names: Optional[List[str]] = None,
        percent_missing: Optional[float] = None,
        missingness_mechanism: Optional[str] = None,
        missing_cols: Optional[List[str]] = None,
        observed_cols: Optional[List[str]] = None,
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.seed = seed
        self.val_test_size = val_test_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        # options
        self.fully_observed = fully_observed
        self.scale = scale
        self.ampute = ampute
        self.discretize = discretize
        self.uniform_prob = uniform_prob
        self.onehot_prefix_names = onehot_prefix_names

        # necessary if amputing
        self.percent_missing = percent_missing
        self.missingness_mechanism = missingness_mechanism
        # In case the missing/obs cols are onehot encoded  cat vars, add indicators
        self.missing_cols = [self.add_onehot_indicator(coln) for coln in missing_cols]
        self.observed_cols = [self.add_onehot_indicator(coln) for coln in observed_cols]
        # necessary if discretizing, don't  need to onehot indicate bc they're continous
        self.ctn_columns = ctn_columns

        # Assertions
        # need continuous cols for discretization
        assert not discretize or (
            ctn_columns is not None
        ), "Failed to provide which continous columns to discretize."
        # need to discretize if imposing uniform dist
        assert (
            not uniform_prob or discretize
        ), "Did not indicate to discretize but indicated uniform probability. You need discretization to impose a uniform probability."
        # need auxiliarry info for amputation, (more if mar)
        if ampute:
            assert (
                percent_missing is not None
                and missingness_mechanism is not None
                and missing_cols is not None
            ), "Failed to provide settings for amputation."
            if missingness_mechanism == "MAR":
                assert (
                    observed_cols is not None
                ), "Failed to provide observed columns for MAR mechanism."

    def setup(self, stage: Optional[str] = None):
        # add in onehot indicators
        self.X = self.X.rename(self.add_onehot_indicator, axis=1)

        # get the columns before sklearn/other preprocessing steps strip them away
        self.columns = self.X.columns
        X = self.X
        y = self.y
        if self.fully_observed:
            # keep rows NOT missing a value for any feature
            fully_observed_mask = self.X.notna().all(axis=1)
            X = self.X[fully_observed_mask]
            y = self.y[fully_observed_mask]

        ground_truth = X.copy()

        # Don't ampute if we're doing a purely F.O. experiment.
        if self.ampute:
            X = ampute(
                X,
                self.seed,
                self.missing_cols,
                self.percent_missing,
                self.missingness_mechanism,
                self.observed_cols,
            )

        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X,
            y,
            test_size=self.val_test_size,
            stratify=y,
            random_state=self.seed,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=self.test_size,
            stratify=y_val_test,
            random_state=self.seed,
        )

        X_true_train = ground_truth.loc[X_train.index]
        X_true_val = ground_truth.loc[X_val.index]
        X_true_test = ground_truth.loc[X_test.index]

        #### Post-split transforms ####
        if self.scale:
            X_train, X_val, X_test = scale(X_train, [X_val, X_test], self.ctn_columns)
            X_true_train, X_true_val, X_true_test = scale(
                X_true_train, [X_true_val, X_true_test], self.ctn_columns
            )

        if self.discretize:
            self.X_train_undisc = X_train
            self.X_val_undisc = X_val
            self.X_test_undisc = X_test
            self.X_true_train_undisc = X_true_train
            self.X_true_val_undisc = X_true_val
            self.X_true_test_undisc = X_true_test
            # use the discretizer trained on potentially amputed train on the truth as well because we need unified discretization
            (
                X_train,
                X_val,
                X_test,
                X_true_train,
                X_true_val,
                X_true_test,
                discretizer_dicts,
            ) = discretize(
                X_train,
                y_train,
                [X_val, X_test, X_true_train, X_true_val, X_true_test],
                self.ctn_columns,
            )
            self.columns_disc = X_train.columns
            self.discretizations = discretizer_dicts
            # self.num_categories = {k: len(v) for k, v in discretizer_dicts.items()}
            self.num_categories = self.get_num_categories(self.columns_disc)

        if self.uniform_prob:
            # Does nothing if nothign is missing
            for df in [X_train, X_val, X_test, X_true_train, X_true_val, X_true_test]:
                df.update(
                    uniform_prob_across_nan(df, self.num_categories, self.ctn_columns)
                )

        if stage == "fit" or stage is None:
            self.X_train, self.X_true_train, self.y_train = (
                X_train,
                X_true_train,
                y_train,
            )
            self.X_val, self.X_true_val, self.y_val = X_val, X_true_val, y_val

        if stage == "test" or stage is None:
            self.X_test, self.X_true_test, self.y_test = X_test, X_true_test, y_test

    def train_dataloader(self):
        if self.discretize and self.uniform_prob:
            return self.create_dataloader(
                [self.X_train, self.X_true_train],
                [self.X_train_undisc, self.X_true_train_undisc],
            )
        return self.create_dataloader([self.X_train, self.X_true_train])

    def val_dataloader(self):
        if self.discretize and self.uniform_prob:
            return self.create_dataloader(
                [self.X_val, self.X_true_val],
                [self.X_val_undisc, self.X_true_val_undisc],
            )
        return self.create_dataloader([self.X_val, self.X_true_val])

    def test_dataloader(self):
        if self.discretize and self.uniform_prob:
            return self.create_dataloader(
                [self.X_test, self.X_true_test],
                [self.X_test_undisc, self.X_true_test_undisc],
            )
        return self.create_dataloader([self.X_test, self.X_true_test])

    def create_dataloader(
        self,
        dfs: List[Union[pd.DataFrame, np.ndarray]],
        dfs_undisc: Optional[List[Union[pd.DataFrame, np.ndarray]]] = None,
    ) -> DataLoader:
        """Takes any number of pandas dfs or numpy arrays and packages them for pytorch."""

        def enforce_numpy(df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            return df.values if isinstance(df, pd.DataFrame) else df

        dataset = TensorDataset(*[torch.Tensor(enforce_numpy(df)) for df in dfs])
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 * self.num_gpus,
            pin_memory=True,
        )
        # the undiscretized version will be a tensor of different length
        # dataloader does not support tensors of different sizes in the same dataloader, so we load them side by side like this
        if dfs_undisc:
            dataset_undisc = TensorDataset(
                *[torch.Tensor(enforce_numpy(df)) for df in dfs_undisc]
            )
            loader_undisc = DataLoader(
                dataset_undisc,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4 * self.num_gpus,
                pin_memory=True,
            )
            loader = cat_dataloaders([loader, loader_undisc])
        return loader

    def get_num_categories(self, columns) -> Counter:
        """Assuming everything is one_hot (share prefixes)."""
        assert (
            self.discretize
        ), "Tried to get one-hot num_categories for uniform probability across nan but discretization was not selected."
        list_of_col_prefixes = columns.str.rpartition("_").get_level_values(0).to_list()

        num_categories = Counter(list_of_col_prefixes)
        # if missing (prefix=="") that means it was just a binary variable before, so count is 2
        num_categories[""] += 1
        return num_categories

    def add_onehot_indicator(self, coln: str) -> str:
        """Takes the prefixes passed to the constructor and adds a suffix of "_onehot" to each column name so that we can later group these together."""
        if self.onehot_prefix_names is None:
            return coln
        matches_any_prefix = any(
            [coln.startswith(prefix) for prefix in self.onehot_prefix_names]
        )
        return coln + "_onehot" if matches_any_prefix else coln

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "CommonDataModule":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="When training the autoencoder, set the batch size.",
        )
        p.add_argument(
            "--num-gpus",
            type=int,
            default=4,
            help="Number of workers for the pytorch dataset used in passing batches to the autoencoder.",
        )
        p.add_argument(
            "--batch-log-interval",
            type=int,
            default=500,
            help="When training the autoencoder and verbosity is on, set the interval for printing progress in training on a batch.",
        )
        p.add_argument(
            "--scale",
            type=str2bool,
            default=False,
            help="When training the autoencoder, whether or not to scale the data before passing the data to the network.",
        )
        return p


class cat_dataloaders:
    """Class to concatenate multiple dataloaders.
    Ref: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/35
    """

    def __init__(self, dataloaders):
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


## DEPRECATED: use discretize from data_transforms instead
def manual_discretize(
    X: pd.DataFrame, config=None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Discretize continuous vars manually.

    Sources are medical or, with age just intuitive.
    Includes num_categories since sometimes we end up
    with empty bins and the nunique fails.

    It takes a config dict from ray.tune if tuning the bins.
    """
    data = X.copy()
    if config:
        manual_bins = {
            "age": config["age"],
            "egfr": config["egfr"],
            "a1c": config["a1c"],
            "hba1c": config["a1c"],
            "sbp": config["sbp"],
        }
    else:
        manual_bins = {
            # 'age': [0, 20, 30, 40, 50, 60, 70, 80, 90, 200],
            # # ref: https://www.kidney.org/atoz/content/gfr
            # 'egfr': [0, 15, 30, 45, 60, 90, 1000],
            # # ref: https://www.diabetes.org/a1c/diagnosis
            # 'a1c': [0, 5.7, 6.5, 100],
            # 'hba1c': [0, 5.7, 6.5, 100],  # same thing named twice
            # # ref : heart.org/en/health-topics/high-blood-pressure/
            # #                   understanding-blood-pressure-readings
            # 'sbp': [0, 120, 130, 140, 180, 500]
            "age": [0, 15, 30, 45, 60, 75, 90, 105, 120, 200],
            "egfr": [0, 40, 80, 1000],
            "a1c": [0.0, 3.3, 6.6, 9.899999999999999, 100],
            "hba1c": [0.0, 3.3, 6.6, 9.899999999999999, 100],
            "sbp": [0, 80, 160, 500],
        }

    # map to correct nunique if no values fall into a bin
    num_manual_categories = {}
    for col in data.columns:
        # split and grab thing which will be 'age' for example
        if "entry" in col:
            # in study_entry it's the last position
            name = col.split("_")[-1]
        else:
            # in time_zero/beyond it's <name>_mean
            name = col.split("_")[-2]
        if name in manual_bins:
            data.loc[:, col] = pd.cut(data[col], manual_bins[name])
            num_manual_categories[col] = len(manual_bins[name]) - 1
        else:  # if not medically advised bin, just split into quartiles
            data.loc[:, col] = pd.qcut(data[col], 4, duplicates="drop")

    return (data, num_manual_categories)
