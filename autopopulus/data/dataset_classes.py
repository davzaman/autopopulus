import inspect
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

from numpy import array, full_like, ndarray, where
from pandas import DataFrame, MultiIndex, Index, Series
from pytorch_lightning import LightningDataModule

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


#### Local module ####
from autopopulus.data.constants import (
    PAD_VALUE,
    PATIENT_ID,
)
from autopopulus.data.types import (
    DataColumn,
    DataT,
    DataTypeTimeDim,
    LabelT,
    LongitudinalFeatureAndLabel,
    StaticFeatureAndLabel,
)
from autopopulus.data.transforms import (
    Discretizer,
    UniformProbabilityAcrossNans,
    ampute,
)
from autopopulus.utils.cli_arg_utils import StringToEnum, str2bool

# Ref for abstract class attributes wihtout @property
# https://stackoverflow.com/a/50381071/1888794
R = TypeVar("R")


class DummyAttribute:  # Helper for abstract attribute for abstract class
    pass


def abstract_attribute(obj: Callable[[Any], R] = None) -> R:
    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(R, _obj)


class AbstractDatasetLoader(ABC):
    """Dataset loading classes to be used by CommonDataModule (which is separate because it's a LightningDataModule.
    Not to be confused with pytorch DataLoader.
    """

    # These must be initialized sometime during or before `load_features_and_labels()`
    continuous_cols: DataColumn = abstract_attribute()
    categorical_cols: DataColumn = abstract_attribute()
    # Need to support longitudinal/static.
    # Can add an extra attribute that's a dict pointing to the static/long versions.
    missing_cols: DataColumn = abstract_attribute()
    observed_cols: DataColumn = abstract_attribute()
    onehot_prefixes: List[str] = abstract_attribute()
    subgroup_filter: Optional[Dict[str, str]] = None

    @abstractmethod
    def load_features_and_labels(
        self, data_type_time_dim: Optional[DataTypeTimeDim]
    ) -> Union[StaticFeatureAndLabel, LongitudinalFeatureAndLabel]:
        """Method to load data and labels for training  and prediction.
        Optionally allow the longitudinal form of the data.
        When longitudinal is not none, this will return the:
        longitudinal portion (longitudinal_split == "longitudinal")
        vs
        static portion (longitudinal_split == "static").
        """
        pass

    def filter_subgroup(
        self, df: DataFrame, encoding: Dict[str, Dict[str, int]]
    ) -> DataFrame:
        """Filters the (binary) features/label for the given subgroup by name.
        Since data is numerically encoded, we have ENCODINGS for bin vars of interest (to filter).
        That way you can filter site source to "ucla" instead of "0".

        If you filter, we remove that var from the df / list of categorical columns (since there's no variation for that var anymore).
        """
        if self.subgroup_filter is None:  # do nothing
            return df

        filters = {
            varname: df[varname] == encoding[varname][group_name]
            for varname, group_name in self.subgroup_filter.items()
            if varname in encoding and group_name in encoding[varname]
        }

        # drop var if filtering down
        df.drop(filters.keys(), axis=1, errors="ignore", inplace=True)
        for filtered_var in filters.keys():
            self.categorical_cols.remove(filtered_var)

        combined_subgroup_filters = reduce(
            lambda filter1, filter2: filter1 & filter2, filters.values()
        )
        return df[combined_subgroup_filters]

    @staticmethod
    @abstractmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Method that can append the corresponding CLI args for that dataset to the ArgParser."""
        pass

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "DataLoader":
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


class SimpleDatasetLoader(AbstractDatasetLoader):
    """Convenience class to pass a basic dataset and necessary auxilliary data.
    Functions like TensorDataset where you pass the basic data and some necessary extras and it will convert it to a class for you easily.
    If the `label` is a string or int it assumes that the `label` is part of `data` and will split it accordingly if the name or index exists.
    """

    def __init__(
        self,
        data: DataT,
        label: Union[str, int, LabelT],
        continuous_cols: DataColumn,
        categorical_cols: DataColumn,
        missing_cols: DataColumn,
        observed_cols: DataColumn,
        onehot_prefixes: List[str],
        subgroup_filter: Optional[Dict[str, str]] = None,
    ) -> None:
        self.data = data
        self.labels = label
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.missing_cols = missing_cols
        self.observed_cols = observed_cols
        self.onehot_prefixes = onehot_prefixes
        self.subgroup_filter = subgroup_filter

    def load_features_and_labels(
        self, data_type_time_dim: Optional[DataTypeTimeDim] = None
    ) -> Union[StaticFeatureAndLabel, LongitudinalFeatureAndLabel]:
        """Takes the basic dataset and splits out the label if desired, otherwise just returns it for CommonDatamodule."""
        # Split data and given label name or index
        if isinstance(self.labels, str) or isinstance(self.labels, int):
            if isinstance(self.labels, str):  # Label given by name (pandas only)
                if isinstance(self.data, DataFrame):
                    mask = self.data.columns != self.labels
                else:
                    raise Exception(
                        "You cannot provide a string label name when the provided data is a numpy array."
                    )
            elif isinstance(self.labels, int):  # label given by index
                mask = full_like(self.data.columns, fill_value=True, dtype=bool)
                mask[self.labels] = False

            label_idx = (~mask).nonzero()[0][0]
            if isinstance(self.data, DataFrame):
                return (
                    self.data[self.data.columns[mask]],
                    # no mask bc i'll get a DF back instead of Series
                    self.data[self.data.columns[label_idx]],
                )
            return (self.data[mask], self.data[~mask])

        # directly given data and labels, just return them
        return (self.data, self.labels)

    @staticmethod
    def add_data_args():
        pass


class CommonDataset(Dataset):
    """Extension of Pytorch Dataset for easy training/dataloading and applying transforms on the fly as the data is loaded into the model."""
    def __init__(
        self,
        split: Dict[str, DataT],
        transforms: Optional[Dict[str, Callable]] = None,
        longitudinal: bool = False,
    ) -> None:
        self.split = split
        self.longitudinal = longitudinal

        def get_unique(index: Index) -> ndarray:
            try:
                return index.unique(PATIENT_ID)
            except Exception:
                return index.unique()

        self.split_ids = {
            k: {k2: get_unique(split_data.index) for k2, split_data in v.items()}
            for k, v in split.items()
        }
        self.transforms = transforms

    def __len__(self):
        return (
            len(self.split_ids["data"]["discretized"])
            if "discretized" in self.split["data"]
            else len(self.split_ids["data"]["normal"])
        )

    def __getitem__(self, index: int):
        data_list_names = [("data", "normal"), ("ground_truth", "normal")]
        # prepend the discretized data (reverse order so that it's in the correct final order)
        if "discretized" in self.split["ground_truth"]:
            data_list_names.insert(0, ("ground_truth", "discretized"))
        if "discretized" in self.split["data"]:
            data_list_names.insert(0, ("data", "discretized"))

        transformed_data_list = []
        for names in data_list_names:
            # get the 1st split_id, then 2nd, etc.
            df = self.split[names[0]][names[1]].loc[
                self.split_ids[names[0]][names[1]][index]
            ]
            if self.transforms:
                transform = self.transforms[names[0]][names[1]]
                # May not exist if no non-discretization steps for example
                if transform:
                    if self.longitudinal:
                        df = transform(df.values)
                    else:
                        # minmaxscaler needs a 2d array even if its just 1 row (for static)
                        df = transform(df.values.reshape(1, -1))
            # go back to 1D so we can accumulate a batch properly
            # df = self.enforce_numpy(df).squeeze()
            df = self.enforce_numpy(df)
            df = Tensor(df).squeeze()
            transformed_data_list.append(df)

        return transformed_data_list

    @staticmethod
    def enforce_numpy(df: DataT) -> ndarray:
        # enforce numpy is numeric with df*1 (fixes bools)
        return (df * 1).values if isinstance(df, DataFrame) else (df * 1)


class CommonDataModule(LightningDataModule):
    """Data loader for AEImputer Works with pandas/numpy data with data wrangling.
    Assumes data has been one-hot encoded.
    IMPORTANT: Assumes (one-hot encoded) categorical vars all share the same unique prefix.
    Also assumes that if categorical var is missing then the nan is propagated for all of the one-hot features.

    Also has a dictionary available `discretizations` after calling setup():
    For both the normal "data" and "ground_truth", it maps a column name (str) to its categorical "bins" (a list of value ranges, List[Tuple[float, float]]), and to the "indices" in the discretized df (List[int]).
    """

    def __init__(
        self,
        dataset_loader: AbstractDatasetLoader,
        seed: int,
        val_test_size: float,
        test_size: float,
        batch_size: int,
        num_gpus: int,
        fully_observed: bool = False,
        data_type_time_dim=DataTypeTimeDim.STATIC,
        scale: bool = False,
        ampute: bool = False,
        discretize: bool = False,
        uniform_prob: bool = False,
        separate_ground_truth_transform: bool = False,
        percent_missing: Optional[float] = None,
        missingness_mechanism: Optional[str] = None,
    ):
        super().__init__()
        self.seed = seed
        self.dataset_loader = dataset_loader

        self.val_test_size = val_test_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.fully_observed = fully_observed
        self.data_type_time_dim = data_type_time_dim
        self.scale = scale
        self.ampute = ampute
        self.discretize = discretize
        self.uniform_prob = uniform_prob
        self.separate_ground_truth_transform = separate_ground_truth_transform
        self.percent_missing = percent_missing
        self.missingness_mechanism = missingness_mechanism
        self._validate_inputs()

    def setup(self, stage: Optional[str] = None):
        """
        Pytorch-Lightning overloaded function, called automatically by Trainer.
        Grabs:
            - longitudinal portion of longitudinal data, or static portion of longitudinal data
            - static data
        Amputation occurs before splitting the data, transforms are fitted after splitting the data.
        Transforms are fitted but not applied to the whole dataset immediately here.
        Transform functions are passed to torch Dataset obj and applied on the fly as data is loaded in batches for training/val/test.
        """
        if stage == "fit" or stage == "train":
            X, y = self.dataset_loader.load_features_and_labels(
                data_type_time_dim=self.data_type_time_dim
            )

            # get the columns before sklearn/other preprocessing steps strip them away
            self.columns = X.columns

            if self.fully_observed:
                # keep rows NOT missing a value for any feature
                fully_observed_mask = X.notna().all(axis=1)
                X = X[fully_observed_mask]
                y = y[fully_observed_mask]

            ground_truth = X.copy()

            # Don't ampute if we're doing a purely F.O. experiment.
            if self.ampute:
                X = ampute(
                    X,
                    seed=self.seed,
                    missing_cols=self.dataset_loader.missing_cols,
                    percent=self.percent_missing,
                    mech=self.missingness_mechanism,
                    # since load features is called before this, we should expect the correct static/longitudinal version
                    observed_cols=self.dataset_loader.observed_cols,
                )

            # cat indices and ctn indices
            self._set_col_indices_by_type()
            # group onehots together, binary vars together, etc.
            self._set_groupby()
            # split by pt id
            self._split_dataset(ground_truth, X, y)
            self._set_post_split_transforms()
            self._set_nfeatures()

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")

    ###############
    #   HELPERS   #
    ###############
    def _validate_inputs(self):
        assert not self.discretize or (
            self.dataset_loader.continuous_cols is not None
        ), "Failed to provide which continous columns to discretize."
        # need to discretize if imposing uniform dist
        assert (
            not self.uniform_prob or self.discretize
        ), "Did not indicate to discretize but indicated uniform probability. You need discretization to impose a uniform probability."
        # need auxiliarry info for amputation, (more if mar)
        if self.ampute:
            assert (
                self.percent_missing is not None
                and self.missingness_mechanism is not None
                and self.dataset_loader.missing_cols is not None
            ), "Failed to provide settings for amputation."
            if self.missingness_mechanism == "MAR":
                assert (
                    self.dataset_loader.observed_cols is not None
                ), "Failed to provide observed columns for MAR mechanism."

    def _set_col_indices_by_type(self):
        """
        Dictionary with list of indices as ndarray (which can be used as indexer) of continuous cols and categorical cols in the tensor.
        If processing longitudinal data, it will choose the continuous/categorical columns from the longitudinal/static features accordingly for each portion of the data.
        """
        # If discretizing continuous col indices are required, otherwise this is not needed
        if self.dataset_loader.continuous_cols is None:
            self.col_indices_by_type = {"continuous": None, "categorical": None}
            return

        # keep longitudinal columns that are continuous
        # But continuous columns are in the flattened df name format so it will contain the longitudinal col name
        ctn_cols = Index(self.dataset_loader.continuous_cols)
        cat_cols = Index(self.dataset_loader.categorical_cols)
        self.col_indices_by_type = {
            "continuous": array(
                [
                    self.columns.get_loc(col)
                    for col in self.columns
                    if ctn_cols.str.contains(col).any()
                ]
            ),
            "categorical": array(
                [
                    self.columns.get_loc(col)
                    for col in self.columns
                    if cat_cols.str.contains(col).any()
                ]
            ),
        }

    def _set_nfeatures(self):
        """
        Set number of features in the dataset.
        MUST come after setting the transforms.
        This will help dynamically set input dimension for AE layers.
        """
        n_features = len(self.columns)
        if self.discretize:
            # add all the added bins
            n_features += reduce(
                lambda added_num_cols, k: added_num_cols
                + len(self.discretizations["data"][k]["indices"]),
                self.discretizations["data"],
                0,
            )
            # subtract duplicate of the original column
            n_features -= len(self.discretizations["data"])
        self.n_features = n_features

    def _set_groupby(self):
        """
        Creates a mapping from index to which group name (column name in common) they belong to so that it can directly be passed to a groupby.

        Since this is done in setup before we actually discretize, we have to figure out how to adjust indices of everything (onehots, binary vars, discretized vars) ahead of time.
        NOTE: We assume that all bins will be used in pd.get_dummies, and self.groupby will be updated when we fit the discretizer (if discretizing).
        """
        # TODO: test this # A B X0 X2 C D Y => A1 A2 B1 B2 B3 X1 X2 C1 C2 D1 D2 D3 Y
        cols = self.columns
        # TODO: use the coltype by indices here instead?
        if self.data_type_time_dim.is_longitudinal():
            # keep longitudinal columns that are continuous
            # But continuous columns are in the flattened df name format so it will contain the longitudinal col name
            ctn_cols = Index(self.dataset_loader.continuous_cols)
            ctn_cols = [col for col in cols if ctn_cols.str.contains(col).any()]
            # repeat with categorical
            cat_cols = Index(self.dataset_loader.categorical_cols)
            cat_cols = [col for col in cols if cat_cols.str.contains(col).any()]
        else:
            ctn_cols = self.dataset_loader.continuous_cols
            cat_cols = self.dataset_loader.categorical_cols

        # dummies will put the new discretized columns at the end (sliding everything else down), reference cols should adjust for that
        # sort=False so symmetric difference doesn't mess up the index
        reference_cols = Index(
            cols.symmetric_difference(ctn_cols, sort=False) if self.discretize else cols
        )
        high_level_groups = {
            "categorical_onehots": {
                index: col
                for col in self.dataset_loader.onehot_prefixes
                for index in where(reference_cols.str.contains(col))[0]
            }
        }

        # bin vars = categorical columns that are not one-hot encoded
        binary_var_indices = set(
            [reference_cols.get_loc(col) for col in cat_cols]
        ) ^ set(high_level_groups["categorical_onehots"].keys())
        high_level_groups["binary_vars"] = {
            index: reference_cols[index] for index in binary_var_indices
        }

        self.groupby: Dict[str, Dict[int, str]] = high_level_groups

    def _split_dataset(
        self,
        ground_truth: DataFrame,
        X: DataFrame,
        y: Union[Series, ndarray],
    ):
        """
        Splits dataset into train/test/val via data index (pt id).
        Also into the different componenets needed for training:
            - data {discretized, normal}
            - ground_truth {discretized, normal}
            - labels
        """
        splits = self._get_dataset_splits(X, y)
        self.splits = {"ground_truth": {}, "data": {}}
        self.splits["ground_truth"]["normal"] = {
            split_name: ground_truth.loc[split_ids]
            for split_name, split_ids in splits.items()
        }
        self.splits["data"]["normal"] = {
            split_name: X.loc[split_ids] for split_name, split_ids in splits.items()
        }

        if self.discretize:
            self.splits["ground_truth"]["discretized"] = {
                split_name: ground_truth.loc[split_ids].copy()
                for split_name, split_ids in splits.items()
            }
            self.splits["data"]["discretized"] = {
                split_name: X.loc[split_ids].copy()
                for split_name, split_ids in splits.items()
            }

        self.splits["label"] = {
            split_name: y[split_ids] for split_name, split_ids in splits.items()
        }

    def _get_dataset_splits(
        self,
        X: DataFrame,
        y: Union[Series, ndarray],
    ) -> Dict[str, ndarray]:
        """
        Splitting via df index with label stratification using sklearn.
        """
        # TODO: enforce pt id is the same name for all dataset
        # pick pt id if longitudinal portion of data (multiindex)
        level = PATIENT_ID if isinstance(X.index, MultiIndex) else None
        sample_ids = X.index.unique(level).values

        train_val_ids, test_ids = train_test_split(
            sample_ids,
            test_size=self.val_test_size,
            stratify=y,
            random_state=self.seed,
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=self.test_size,
            stratify=y[train_val_ids],
            random_state=self.seed,
        )

        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

    def _set_post_split_transforms(self):
        """
        Setup sklearn pipeline for transforms to run after splitting data.
        Assumes groupby is set for categorical onehots and binary vars.
        There are separate pipelines for data and ground_truth.
        If discretizing, we will keep a separate transform function that only applies the non-discretizing steps.
        If discretizing update the groupby (so all bins for the same var can be grouped later), and save the bins fitted/learned by discretizer.
        """

        def get_steps(
            scale: bool = self.scale,
            discretize: bool = self.discretize,
            uniform_prob: bool = self.uniform_prob,
        ) -> List[TransformerMixin]:
            steps = []
            if scale:
                # Scale continuous features to [0, 1].  Can produce negative numbers.
                steps.append(
                    (
                        "continuous-scale",
                        ColumnTransformer(
                            [  # (name, transformer, columns) tuples
                                (
                                    "scale",
                                    MinMaxScaler(),
                                    self.col_indices_by_type["continuous"],
                                )
                            ],
                            remainder="passthrough",
                        ),
                    ),
                )
                steps.append(
                    (
                        "enforce-pandas",
                        FunctionTransformer(
                            lambda np_array: DataFrame(np_array, columns=self.columns)
                        ),
                    )
                )

            if discretize:
                # discretizes continuous vars (supervised).
                steps.append(
                    (
                        "discretize",
                        Discretizer(
                            self.splits["data"]["discretized"]["train"].columns,
                            self.col_indices_by_type["continuous"],
                            # only return it in transform() if uniform prob is toggled
                            return_info_dict=self.uniform_prob,
                        ),
                    )
                )

            if uniform_prob:
                steps.append(
                    (
                        "uniform_probability_across_nans",
                        UniformProbabilityAcrossNans(self.groupby),
                    )
                )
            return steps

        steps = get_steps()
        if steps:
            data_pipeline = Pipeline(steps)
            # train on train, apply to all
            # at this point if discretizing, "normal" and "discretized" are the same, they're copies of the same data.
            # .values so the transformers don't complain about being fitted with feature names and then transforming data with no feature names
            data_pipeline.fit(
                self.splits["data"]["normal"]["train"],  # .values,
                self.splits["label"]["train"],
            )
            if self.separate_ground_truth_transform:
                # Create new instance of all steps, or else they'll point to the same transformer objects
                ground_truth_pipeline = Pipeline(get_steps())
                if "uniform_probability_across_nans" in ground_truth_pipeline:
                    ground_truth_pipeline[
                        "uniform_probability_across_nans"
                    ].ground_truth_pipeline = True
                ground_truth_pipeline.fit(
                    self.splits["ground_truth"]["normal"]["train"],  # .values,
                    self.splits["label"]["train"],
                )
            else:
                ground_truth_pipeline = data_pipeline

            if self.discretize:
                # keep steps that are not discretization related
                def non_discretization_transform(
                    pipeline: Pipeline,
                ) -> Optional[Callable[[ndarray], ndarray]]:
                    # do not create new instance, we want to keep the info it learned from fit!
                    # It is possible there are no non-discretization steps, in that case, do nothing
                    # Decided to not do identity function so it's clearer when we do nothing
                    steps = [
                        (stepname, step)
                        for stepname, step in pipeline.steps
                        if stepname != "discretize"
                        and stepname != "uniform_probability_across_nans"
                    ]
                    return Pipeline(steps).transform if steps else None

                self.transforms = {
                    "data": {
                        # Leave out discretize and uniform prob across nan (last two steps) for regular, not discretized data
                        "normal": non_discretization_transform(data_pipeline),
                        "discretized": data_pipeline.transform,
                    },
                    "ground_truth": {
                        "normal": non_discretization_transform(ground_truth_pipeline),
                        "discretized": ground_truth_pipeline.transform,
                    },
                }
            else:  # no discretization pipeline
                self.transforms = {
                    "data": {
                        "normal": data_pipeline.transform,
                    },
                    "ground_truth": {
                        "normal": ground_truth_pipeline.transform,
                    },
                }
        else:  # do nothing, no preprocessing requested
            self.transforms = {
                "data": {"normal": lambda data: data},
                "ground_truth": {"normal": lambda data: data},
            }

        if self.discretize:  # save discretization dict after running fit.
            self.discretizations = {
                "data": data_pipeline.named_steps["discretize"].d.ctn_to_cat_map,
                "ground_truth": ground_truth_pipeline.named_steps[
                    "discretize"
                ].d.ctn_to_cat_map,
            }

            # update groupby with discretized cols
            # NOTE: requires groupby to be set before post_split_transforms
            self.groupby["discretized_ctn_cols"] = {
                data_name: {
                    index: col
                    for col, col_info in self.discretizations[data_name].items()
                    for index in col_info["indices"]
                }
                for data_name in ["data", "ground_truth"]
            }
        else:
            self.discretizations = None

    def _get_dataloader(self, split: str) -> DataLoader:
        split_data = {
            "data": {"normal": self.splits["data"]["normal"][split]},
            "ground_truth": {"normal": self.splits["ground_truth"]["normal"][split]},
        }
        if self.discretize and self.uniform_prob:
            split_data["data"]["discretized"] = self.splits["data"]["discretized"][
                split
            ]
            split_data["ground_truth"]["discretized"] = self.splits["ground_truth"][
                "discretized"
            ][split]

        return self._create_dataloader(split_data)

    def _create_dataloader(
        self, split_data: Dict[str, Dict[str, DataFrame]]
    ) -> DataLoader:
        """Packages data for pytorch Dataset/DataLoader."""
        is_longitudinal = self.data_type_time_dim in (
            DataTypeTimeDim.LONGITUDINAL,
            DataTypeTimeDim.LONGITUDINAL_SUBSET,
        )
        dataset = CommonDataset(
            split_data,
            self.transforms,
            is_longitudinal,
        )
        loader = DataLoader(
            dataset,
            collate_fn=self._batch_collate if is_longitudinal else None,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 * self.num_gpus,
            pin_memory=True,
        )
        return loader

    def _batch_collate(
        self,
        batch: List[
            Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]
        ],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Batch is a list of tuples with (example, label).
        Pad the variable length sequences, add seq lens, and enforce tensor.
        """
        inputs = list(zip(*batch))
        data_indices = [0, 2] if self.discretize else [0]

        seq_lens = [
            Tensor([len(pt_seq) for pt_seq in inputs[idx]]) for idx in data_indices
        ]
        #  TODO: is that necessary: pad all the data going in ?
        inputs = [
            pad_sequence(X, batch_first=True, padding_value=PAD_VALUE) for X in inputs
        ]
        if self.discretize:
            return (
                (inputs[0], inputs[1], seq_lens[0]),
                (inputs[2], inputs[3], seq_lens[1]),
            )
        return (inputs[0], inputs[1], seq_lens[0])

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "CommonDataModule":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/-1.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        valid_kwargs.update(
            inspect.signature(CommonDataModule.__init__).parameters.copy()
        )
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--data-type-time-dim",
            default=DataTypeTimeDim.STATIC,
            action=StringToEnum(DataTypeTimeDim),
            help="Specify if the dataset should be considered purely static, purely longitudinal, or a subset of a mix.",
        )
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
        p.add_argument(
            "--separate_ground_truth_transform",
            type=str2bool,
            default=False,
            help="Specify whether or not to fit the ground_truth transforms separately on the ground_truth data. When false, the ground_truth transforms are the same as the data transforms (fit to the data).",
        )

        return p
