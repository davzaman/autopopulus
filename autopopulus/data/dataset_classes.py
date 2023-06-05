from abc import ABC, abstractmethod
from argparse import ArgumentParser
from copy import deepcopy
from functools import reduce
from itertools import chain
from os import cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import regex as re

from scipy.stats import norm, yulesimon

from category_encoders.target_encoder import TargetEncoder
from numpy import array, full_like, ndarray, where, nan
from pandas import DataFrame, MultiIndex, Index, Series
from pytorch_lightning import LightningDataModule
from pyampute import MultivariateAmputation

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
    TIME_LEVEL,
)
from autopopulus.data.utils import (
    enforce_numpy,
    explode_nans,
    get_samples_from_index,
    regex_safe_colname,
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
    CombineOnehots,
    Discretizer,
    ColTransformPandas,
    UniformProbabilityAcrossNans,
    identity,
)
from autopopulus.utils.cli_arg_utils import StringToEnum, YAMLStringListToList, str2bool
from autopopulus.utils.utils import CLIInitialized

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


class AbstractDatasetLoader(ABC, CLIInitialized):
    """Dataset loading classes to be used by CommonDataModule (which is separate because it's a LightningDataModule.
    Not to be confused with pytorch DataLoader.
    """

    # These must be initialized sometime during or before `load_features_and_labels()`
    continuous_cols: DataColumn = abstract_attribute()
    categorical_cols: DataColumn = abstract_attribute()
    # Need to support longitudinal/static.
    onehot_prefixes: Optional[List[str]] = None
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
        onehot_prefixes: List[str] = None,
        split_id_column: Optional[str] = None,
        subgroup_filter: Optional[Dict[str, str]] = None,
    ) -> None:
        self.data = data
        self.labels = label
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.onehot_prefixes = onehot_prefixes
        self.subgroup_filter = subgroup_filter
        if split_id_column:
            self.split_ids = self._get_split_ids_from_column(split_id_column)

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

    def _get_split_ids_from_column(
        self, split_column: Optional[str]
    ) -> Dict[str, Index]:
        """If your df has a column indicating splits we grab it here."""
        if split_column is None:
            return

        split_ids = {
            split_name: self.data[self.data[split_column] == split_name].index
            for split_name in ["train", "val", "test"]
        }
        # drop column because we don't need it anymore
        self.data = self.data.drop(split_column, axis=1)

        return split_ids

    @staticmethod
    def add_data_args():
        pass


class CommonTransformedDataset(Dataset):
    """
    Extension of Pytorch Dataset for easy training/dataloading
    does NOT transform data on the fly and assumes data is already transformed.
    Assumes the data is already sliced for a particular split.
    transformed_data_train = {
        "original": {"data": ..., "ground_truth": ...},
        "mapped": {"data": ..., "ground_truth": ...}
    }
    """

    def __init__(
        self,
        transformed_data: Dict[str, Dict[str, DataT]],
        longitudinal: bool = False,
    ) -> None:
        self.transformed_data = transformed_data
        self.longitudinal = longitudinal

        # "original" should always exist and "data" should match "ground_truth" index.
        self.split_ids = get_samples_from_index(
            self.transformed_data["original"]["data"].index
        )

    def __len__(self) -> int:
        return len(self.split_ids)

    def __getitem__(self, index: int) -> Dict[str, Dict[str, Tensor]]:
        """Slices a single sample, converts to Tensor, and returns dictionary."""
        return_tensors = {}
        for (
            data_version,
            data_dict,
        ) in self.transformed_data.items():  # ["original", "mapped"]
            return_tensors[data_version] = {}

            for data_role, data in data_dict.items():  # ["data", "ground_truth"]
                # a list of indices will ensure I get back a DF (2d array) whether im asking for just 1 index
                sample = data.loc[[self.split_ids[index]]]
                return_tensors[data_version][data_role] = Tensor(
                    enforce_numpy(sample)
                ).squeeze()

        return return_tensors


class CommonDatasetWithTransform(Dataset):
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

        # TODO[LOW]: Technically I only need split ids for data, it should match ground_truth.
        self.split_ids = {
            k: get_unique(split_data.index)
            for k, split_data in split.items()
            if k != "label"
        }
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.split_ids["data"])

    def __getitem__(self, index: int) -> Dict[str, Dict[str, Tensor]]:
        """
        Returns (transformed_data) dictionary instead of list of tuples.
        {
            "original": {"data": ..., "ground_truth": ...},
            "mapped": {"data": ..., "ground_truth": ...}
        }
        The structure matches `split`, `split_ids`, but NOT `transforms`.
        Transforms has "original" transforms e.g. scaling that should be applied whether or not we are feature mapping.
        """
        if self.transforms:
            transformed_data = {
                data_version: {} for data_version in self.transforms.keys()
            }
        else:
            transformed_data = {"original": {}}

        for data_version in transformed_data.keys():
            for (
                data_role,
                split_ids,
            ) in self.split_ids.items():  # ["data", "ground_truth"]
                # a list of indices will ensure I get back a DF (2d array) whether im asking for just 1 index
                df = self.split[data_role].loc[[split_ids[index]]]
                if self.transforms:
                    transform = self.transforms[data_version][data_role]
                    # May not exist if no non-discretization steps for example
                    if transform:
                        if self.longitudinal:
                            # df = transform(df.values)
                            df = transform(df)
                        else:
                            # minmaxscaler needs a 2d array even if its just 1 row (for static)
                            # df = transform(df.values.reshape(1, -1))
                            df = transform(df)
                # go back to 1D so we can accumulate a batch properly
                # df = self.enforce_numpy(df).squeeze()
                transformed_data[data_version][data_role] = Tensor(
                    enforce_numpy(df)
                ).squeeze()

        return transformed_data


class CommonDataModule(LightningDataModule, CLIInitialized):
    """Data loader for AEImputer Works with pandas/numpy data with data wrangling.
    Splits can be automatically taken care of or be specified as "split_ids" in the DatasetLoader.
    Pass the data in the way you'd like it to be passed to your downstream model.
    That is, if you need onehot features, then pass a dataset with onehot features.

    FEATURE-MAP:
    All mappings are invertible, where the inverted mapping will be applied after imputation loss is computed and are only used for metrics, not training.
    Your multi-categorical features must already be one-hot encoded (onehot_prefixes should be set in DatasetLoader) if you choose [None, discretize_continuous]. You are allowed to set onehot_prefixes if you choose target_encode_categorical, the will be automatically combined (and then divided later) for you.
    If you do not not specify onehot_prefixes for [None, discretize_continuous] then it is assumed that the whole of the categorical variables passed in are all binary.
    IMPORTANT: If you pass data with one-hot encoded categorical vars, we assume each shares the same unique prefix.
    Also assumes that if categorical var is missing then the nan is propagated for all of the one-hot features.
        - discretize_continuous: Dictionary available `discretizations` after calling setup().  For both the normal "data" and "ground_truth", it maps a column name (str) to its categorical "bins" (a list of value ranges, List[Tuple[float, float]]), and to the "indices" in the discretized df (List[int]).
        - target_encode_categorical:  Dictionary available `inverse_target_encode_map` after calling setup(). It contains the inverse transform function for the ordinal encoder, and a dictionary that maps a column name to the inverse float mean target to ordinal encoded number.
    """

    def __init__(
        self,
        dataset_loader: AbstractDatasetLoader,
        batch_size: int = 32,
        num_workers: Union[str, int] = 4,
        fully_observed: bool = False,
        data_type_time_dim=DataTypeTimeDim.STATIC,
        scale: bool = False,
        ampute: bool = False,
        feature_map: Optional[str] = "onehot_categorical",
        uniform_prob: bool = False,
        separate_ground_truth_transform: bool = False,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        percent_missing: Optional[float] = None,
        amputation_patterns: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        limit_data: Optional[int] = None,  # Debugging purposes
    ):
        super().__init__()
        self.seed = seed
        self.dataset_loader = dataset_loader

        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        if isinstance(num_workers, int):
            self.num_workers = num_workers
        elif num_workers == "auto":
            self.num_workers = cpu_count()
        else:  # no multiprocessing
            self.num_workers = 0
        self.fully_observed = fully_observed
        self.data_type_time_dim = data_type_time_dim
        self.scale = scale
        self.ampute = ampute
        self.feature_map = feature_map
        self.uniform_prob = uniform_prob
        self.separate_ground_truth_transform = separate_ground_truth_transform
        self.percent_missing = percent_missing
        self.amputation_patterns = amputation_patterns
        self._validate_inputs()

        self.limit_data = limit_data

    def setup(self, stage: Optional[str] = None):
        """
        Pytorch-Lightning overloaded function, called automatically by Trainer.
        Grabs:
            - longitudinal portion of longitudinal data, or static portion of longitudinal data
            - static data
        Amputation occurs before splitting the data, transforms are fitted after splitting the data.
        If amputing and you select a already-onehot encoded column, we will explode the nans across the bins for you.
        Splits are specified either by the DatasetLoader, if not then by relative split sizes. If both are specified it will default to the splits from DatasetLoader.
        Transforms are fitted but not applied to the whole dataset immediately here.
        Transform functions are passed to torch Dataset obj and applied on the fly as data is loaded in batches for training/val/test.
        """
        if stage == "fit" or stage == "train":
            X, y = self.dataset_loader.load_features_and_labels(
                data_type_time_dim=self.data_type_time_dim
            )
            if self.limit_data:
                # half data be True the other False
                idx = y.index[y == True][: self.limit_data // 2]
                idx = idx.append(y.index[y == False][: self.limit_data // 2])
                X = X.loc[idx]
                y = y.loc[idx]

            # get the columns info before sklearn/other preprocessing steps strip them away
            self._set_auxilliary_column_info(X)

            if self.fully_observed:
                # keep rows NOT missing a value for any feature
                fully_observed_mask = X.notna().all(axis=1)
                X = X[fully_observed_mask]
                y = y[fully_observed_mask]

            ground_truth = X.copy()
            self.ground_truth_has_nans = ground_truth.isna().any().any()

            # Don't ampute if we're doing a purely F.O. experiment.
            if self.ampute:
                # TODO: add tests for _add_latent_features, refact into function to test?
                X, pyampute_patterns = self._add_latent_features(X)
                amputer = MultivariateAmputation(
                    prop=self.percent_missing,
                    patterns=pyampute_patterns,
                    seed=self.seed,
                )
                X = amputer.fit_transform(X)
                # drop the added latent features.
                X = X.drop(X.filter(regex=r"latent_p\d+_.+").columns, axis=1)

            # expand nans where its onehot to make sure the whole group is nan
            # before assigned to splits
            X = explode_nans(X, self.col_idxs_by_type["original"].get("onehot", []))

            # split by pt id
            self._split_dataset(ground_truth, X, y)
            self._set_post_split_transforms()

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
        assert (self.num_workers >= 0) and (
            self.num_workers <= cpu_count()
        ), "Number of CPUs has to be a non-negative integer, and not exceed the number of cores."
        assert (self.test_size is not None and self.val_size is not None) or (
            hasattr(self.dataset_loader, "split_ids")
        ), "Need to either specify split percentages or provide custom splits to the dataset loader."
        if self.feature_map == "discretize_continuous":
            assert (
                self.dataset_loader.continuous_cols is not None
            ), "Failed to provide which continous columns to discretize."
        elif self.feature_map == "target_encode_categorical":
            assert (
                self.dataset_loader.categorical_cols is not None
            ), "Failed to provide which categorical columns to encode."
        if self.uniform_prob:  # need to discretize if imposing uniform dist
            assert (
                self.feature_map == "discretize_continuous"
            ), "Did not indicate to discretize but indicated uniform probability. You need discretization to impose a uniform probability."
        # need auxiliary info for amputation, (more if mar)
        if self.ampute:
            assert (
                self.percent_missing is not None
                and self.amputation_patterns is not None
            ), "Failed to provide settings for amputation."

    def _add_latent_features(
        self, data: DataFrame
    ) -> Tuple[DataFrame, List[Dict[str, Any]]]:
        """
        For amputation, if MNAR is recoverable we need to create and add our own latent features.
        If we want MNAR (recoverable) specify the distribution name(s) in parentheses.
        On MNAR (non-recoverable) the default behavior of pyampute is correct.
        Mutates self.amputation_patterns.
        """
        X = data.copy()
        # we might modify the patterns passed in so we create a new object
        pyampute_patterns = deepcopy(self.amputation_patterns)
        for i, pattern in enumerate(pyampute_patterns):
            # match MNAR(*) or MNAR(*, *, ...) but not MNAR alone
            if re.search(r"MNAR\(\w+(?:,\s*\w+)*\)", pattern["mechanism"]):
                # Grab everything in between () and split by comma
                distributions = (
                    re.findall(r"\((.*?)\)", pattern["mechanism"])[0]
                    .replace(" ", "")
                    .split(",")
                )
                # Add feature F_i to data X
                # enumerate in case you want 2 of the same distribution
                # we need to have unique feature/col name
                for j, distribution in enumerate(distributions):
                    distribution = distribution.lower()
                    feature_name = f"latent_p{i}_{distribution}{j}"
                    # Create feature following distribution *
                    if re.search(r"(g(auss(ian)?)?)|(n(orm(al)?)?)", distribution):
                        X[feature_name] = norm.rvs(
                            0, 1, size=len(X), random_state=self.seed
                        )
                    elif re.search(r"y(ule)?|s(imon)?", distribution):
                        X[feature_name] = yulesimon.rvs(
                            1.5, size=len(X), random_state=self.seed
                        )
                    # Add weights
                    if "weights" in pattern:
                        if isinstance(pattern["weights"], dict):
                            pattern["weights"][feature_name] = 1
                        else:  # list, where the feature was added to the end
                            pattern["weights"].append(1)
                    else:
                        pattern["weights"] = {feature_name: 1}

                    # Change name to be pyampute compatible
                    pattern["mechanism"] = "MNAR"
        return (X, pyampute_patterns)

    def _set_auxilliary_column_info(self, X: DataFrame):
        self.columns = {"original": X.columns}
        # cat indices and ctn indices
        self._set_col_idxs_by_type()
        # group categorical (bin/multicat/onehot) vars together, etc.
        self._set_groupby()
        self._set_nfeatures()

    def _set_col_idxs_by_type(self):
        """
        Dictionary with list of indices as ndarray (which can be used as indexer) of continuous cols and categorical cols in the tensor.
        If processing longitudinal data, it will choose the continuous/categorical columns from the longitudinal/static features accordingly for each portion of the data.

        There's a "original" and "mapped" version, each should have keys: ["continuous", "categorical", "binary", "onehot"].
        All except "onehot" are List[int], where "onehot" is a List[List[int]] (for each group).
        """
        # If discretizing continuous col indices are required, otherwise this is not needed
        if self.dataset_loader.continuous_cols is None:
            self.col_idxs_by_type: Dict[
                str, Dict[str, Union[List[int], List[List[int]]]]
            ] = {"original": {}}
            return

        # keep longitudinal columns that are continuous
        # But continuous columns are in the flattened df name format so it will contain the longitudinal col name
        ctn_cols = Index(self.dataset_loader.continuous_cols)
        cat_cols = Index(self.dataset_loader.categorical_cols)
        self.col_idxs_by_type = {
            "original": {
                "continuous": array(
                    [
                        self.columns["original"].get_loc(col)
                        for col in self.columns["original"]
                        if ctn_cols.str.contains(regex_safe_colname(col)).any()
                    ]
                ),
                "categorical": array(
                    [
                        self.columns["original"].get_loc(col)
                        for col in self.columns["original"]
                        if cat_cols.str.contains(regex_safe_colname(col)).any()
                    ]
                ),
            }
        }

        # useful for target encoding
        if self.dataset_loader.onehot_prefixes is not None:
            self.col_idxs_by_type["original"]["onehot"]: List[List[int]] = [
                where(
                    self.columns["original"].str.contains(f"^{regex_safe_colname(col)}")
                )[0].tolist()
                for col in self.dataset_loader.onehot_prefixes
            ]
            # If its missing argwhere() returns an empty array that i don't want.
            self.col_idxs_by_type["original"]["onehot"] = [
                item
                for item in self.col_idxs_by_type["original"]["onehot"]
                if len(item) > 0
            ]
            # bin vars = categorical columns that are not one-hot encoded
            self.col_idxs_by_type["original"]["binary"] = list(
                set(self.col_idxs_by_type["original"]["categorical"])
                ^ set(chain.from_iterable(self.col_idxs_by_type["original"]["onehot"]))
            )
        elif (
            self.feature_map != "target_encode_categorical"
        ):  # assume every cat var is binary
            self.col_idxs_by_type["original"]["binary"] = self.col_idxs_by_type[
                "original"
            ]["categorical"]

    def _set_nfeatures(self):
        """
        Set number of features in the dataset.
        This will help dynamically set input dimension for AE layers.
        Assumed to be set before set_post_split_transforms, as may be changed by feature_maps
        """
        self.nfeatures = {"original": len(self.columns["original"])}

    def _set_groupby(self):
        """
        Creates a mapping from index to which group name (column name in common) they belong to so that it can directly be passed to a groupby.
        We require the indices for the AE model (e.g., binary column thresholding), but the column names for the post_split_transforms.
        We keep just the indices and convert them to column names just for the post_split_transforms.

        - ["categorical_onehots"] => if onehot_prefixes is set
        - ["combined_onehots"] => if onehot_prefixes and feature_map "target_encode_categorical"
        - ["discretized_ctn_cols"] => if feature_map is "discretize_continuous"
        - ["binary_vars"] => if categorical features set and they're not onehot or converted to continuous via target encoding.

        NOTE: We assume that all bins will be used in pd.get_dummies, and self.groupby will be updated when we fit the discretizer (if discretizing).
        """
        # keep longitudinal columns that are continuous
        # But continuous columns are in the flattened df name format so it will contain the longitudinal col name
        # This should work either case.
        high_level_groups = {}
        if "onehot" in self.col_idxs_by_type["original"]:
            high_level_groups["categorical_onehots"] = {
                # get prefix of each column
                index: self.dataset_loader.onehot_prefixes[group]
                for group, group_idxs in enumerate(
                    self.col_idxs_by_type["original"]["onehot"]
                )
                for index in group_idxs
            }

        if "binary" in self.col_idxs_by_type["original"]:
            # TODO[LOW]: since i have binaries in col_idxs_by_type do I need it in the groupby?
            high_level_groups["binary_vars"] = {
                index: self.columns["original"][index]
                for index in self.col_idxs_by_type["original"]["binary"]
            }

        # original/mapped -> [groupname -> indices]
        self.groupby: Dict[str, Dict[str, Dict[int, str]]] = {
            "original": high_level_groups
        }

    def _split_dataset(
        self,
        ground_truth: DataFrame,
        X: DataFrame,
        y: Union[Series, ndarray],
    ):
        """
        Splits dataset into train/test/val via data index (pt id).
        Also into the different componenets needed for training:
            - data {mapped, normal}
            - ground_truth {mapped, normal}
            - labels
        Where the mapping could be discretized, target encoded, or onehot encoded.
        The self.splits dictionary will then look like:
        {
            "data": {"train": ..., "val": ..., "test": ...},
            "ground_truth": {"train": ..., "val": ..., "test": ...},
            # "non_missing_mask": {"train": ..., "val": ..., "test": ...},
            "label": {"train": ..., "val": ..., "test": ...},
        }
        """
        splits = self._get_dataset_splits(X, y)
        self.splits: Dict[str, Tensor] = {
            "data": {},
            "ground_truth": {},
            # "non_missing_mask": {},  # I don't need this if I'm warm-starting in AEDitto shared_step
            "label": {},
        }
        for split_name, split_ids in splits.items():
            self.splits["ground_truth"][split_name] = ground_truth.loc[split_ids]
            self.splits["data"][split_name] = X.loc[split_ids]
            # self.splits["non_missing_mask"][split_name] = (
            #     ~X.loc[split_ids].isna().astype(bool)
            # )
            self.splits["label"][split_name] = y[split_ids]

    def _get_dataset_splits(
        self,
        X: DataFrame,
        y: Union[Series, ndarray],
    ) -> Dict[str, ndarray]:
        """
        Splitting via df index with label stratification using sklearn.
        """
        # Use pre-specified splits if user has them
        if hasattr(self.dataset_loader, "split_ids"):
            return self.dataset_loader.split_ids
        # TODO[LOW]: enforce pt id is the same name for all dataset
        sample_ids = get_samples_from_index(X.index)
        labels = y

        train_val_ids, test_ids = train_test_split(
            sample_ids,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.seed,
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=self.val_size,
            stratify=labels[train_val_ids],
            random_state=self.seed,
        )

        return {"train": train_ids, "val": val_ids, "test": test_ids}

    def _set_post_split_transforms(self):
        """
        Setup sklearn pipeline for transforms to run after splitting data.
        Assumes groupby is set for categorical onehots and binary vars.
        There are separate pipelines for data and ground_truth.
        If feature_map, we will keep a separate transform function that only applies the non-mapping steps.
        If discretizing update the groupby (so all bins for the same var can be grouped later), and save the bins fitted/learned by discretizer.

        # TODO write tests and asserts for these?
        NOTE: FEATURE MAPPINGS SHOULD
            1. preserve nans.
            2. not introduce any new nans.
        """

        def get_steps(
            scale: bool = self.scale,
            feature_map: str = self.feature_map,
            uniform_prob: bool = self.uniform_prob,
        ) -> List[TransformerMixin]:
            steps = []
            if scale:
                # Scale continuous features. Can produce negative numbers.
                steps.append(
                    (
                        "scale_continuous",
                        ColTransformPandas(
                            self.columns["original"],
                            transformers=[  # (name, transformer, columns) tuples
                                (
                                    "scale",
                                    # https://stats.stackexchange.com/a/328988/273369
                                    # MinMaxScaler(feature_range=(-1, 1)),
                                    MinMaxScaler(feature_range=(0, 1)),
                                    self.col_idxs_by_type["original"]["continuous"],
                                )
                            ],
                        ),
                    ),
                )

            if feature_map == "discretize_continuous":
                # discretizes continuous vars (supervised).
                steps.append(
                    (
                        "discretize",
                        Discretizer(
                            self.columns["original"],
                            self.col_idxs_by_type["original"]["continuous"],
                            # only return it in transform() if uniform prob is toggled
                            return_info_dict=self.uniform_prob,
                        ),
                    )
                )
                if uniform_prob:
                    steps.append(
                        (
                            "uniform_probability_across_nans",
                            # I don't need groupby in mapped space since i'm not using the indices, but the names.
                            UniformProbabilityAcrossNans(
                                self.groupby["original"], self.columns["original"]
                            ),
                        )
                    )
            elif feature_map == "target_encode_categorical":
                intermediate_categorical_cols = self.dataset_loader.categorical_cols
                if "onehot" in self.col_idxs_by_type["original"]:
                    # undo the onehot
                    steps.append(
                        (
                            "combine_onehots",
                            CombineOnehots(
                                self.groupby["original"]["categorical_onehots"],
                                self.columns["original"],
                            ),
                        )
                    )
                    # the columns will change between comebine and target_enc, we need the columns here now (intermediate)
                    self.columns["mapped"] = (
                        self.columns["original"]
                        # drop onehots
                        .drop(
                            self.columns["original"][
                                list(
                                    chain.from_iterable(  # flatten to List[int]
                                        self.col_idxs_by_type["original"]["onehot"]
                                    )
                                )
                            ]
                        ).union(  # add back the onehots but only their prefixes
                            self.dataset_loader.onehot_prefixes, sort=False
                        )
                    )
                    # a mix of binary and multicategorical (not one hot/now combined) features
                    intermediate_categorical_cols = self.columns["mapped"].drop(
                        self.columns["original"][
                            self.col_idxs_by_type["original"]["continuous"]
                        ]
                    )
                else:
                    self.columns["mapped"] = self.columns["original"]

                steps.append(
                    (
                        "target_encode_categorical",
                        TargetEncoder(
                            cols=intermediate_categorical_cols,
                            handle_missing="return_nan",
                        ),
                    ),
                )

            return steps

        #### POST FIT LOGIC ####
        steps = get_steps()
        if steps:
            # train on train, apply to all
            data_pipeline = Pipeline(steps)
            ground_truth_pipeline = data_pipeline
            # at this point if discretizing, "original" and "mapped" are the same, they're copies of the same data.
            data_pipeline.fit(
                self.splits["data"]["train"],
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
                    self.splits["ground_truth"]["train"],
                    self.splits["label"]["train"],
                )

            self._set_auxilliary_info_post_mapping(data_pipeline, ground_truth_pipeline)

            self.transforms = {
                "original": {  # keep steps that are not feature mapping related
                    "data": self._get_original_transform(data_pipeline),
                    "ground_truth": self._get_original_transform(ground_truth_pipeline),
                },
            }
            # TODO: currently we assume onehot_categorical, so nothing happens when you pick that.
            if self.feature_map in [
                "target_encode_categorical",
                "discretize_continuous",
            ]:
                self.transforms["mapped"] = {
                    "data": data_pipeline.transform,
                    "ground_truth": ground_truth_pipeline.transform,
                }
        else:  # If no steps asked for, no transforms
            self.transforms = None
            self.discretizations = None
            self.inverse_target_encode_map = None

    @staticmethod
    def _get_original_transform(
        pipeline: Pipeline,
    ) -> Optional[Callable[[ndarray], ndarray]]:
        """
        Do not create new instance, we want to keep the info it learned from fit!
        It is possible there are no non-mapping steps, in that case, do nothing.
        """
        steps = [
            (stepname, step)
            for stepname, step in pipeline.steps
            if stepname
            not in {
                "discretize",
                "uniform_probability_across_nans",
                "combine_onehots",
                "target_encode_categorical",
            }
        ]
        if steps:
            return Pipeline(steps).transform
        # do nothing
        return identity

    def _set_mapped_groupby_discretize(self):
        """
        Instead of dropping and adding the feauters and recalculating the indices we just iterate once through the categorical column indices.
        This must be indices since we use it to invert the tensor after passing through to the model.
        """
        if self.feature_map == "discretize_continuous":
            ## Calculate Shift
            # for each categorical feature, count how many continuous features were before it and shift it down
            def new_cat_indices(cat, ctn):
                n_smaller = 0
                new_idx = []
                for i in range(len(cat)):  # go through each categorical feature
                    # the index into continuous also counts # smaller
                    while n_smaller < len(ctn) and ctn[n_smaller] < cat[i]:
                        n_smaller += 1
                    # shift cat index down
                    new_idx.append(cat[i] - n_smaller)
                return new_idx

            # update existing groupby indices: map old idx -> new shifted idx
            mapping = dict(
                zip(
                    self.col_idxs_by_type["original"]["categorical"],
                    new_cat_indices(
                        self.col_idxs_by_type["original"]["categorical"],
                        self.col_idxs_by_type["original"]["continuous"],
                    ),
                )
            )
            self.groupby["mapped"] = {
                groupname: {mapping[idx]: col for idx, col in group_idxs.items()}
                for groupname, group_idxs in self.groupby["original"].items()
            }

    def _set_auxilliary_info_post_mapping(
        self, data_pipeline: Pipeline, ground_truth_pipeline: Pipeline
    ):
        """
        Will set any new attributes for that mapping.
        Also updates existing auxilliary info for any mapping that changes
        the feature cardinality/the order of features:
            - self.groupby
            - self.nfeatures
            - self.col_idxs_by_type
            - self.columns
        Each of these has "original" version and "mapped" version.
        The "mapped" version will be in mapped space.
        Columns also has "post-invert" self.columns which will be in original space but out of order.
        """
        self.discretizations = None
        self.inverse_target_encode_map = None
        # save discretization dict after running fit, this also changes cardinality of the dataset
        if self.feature_map == "discretize_continuous":
            self.discretizations = {
                "data": data_pipeline.named_steps["discretize"].map_dict,
                "ground_truth": ground_truth_pipeline.named_steps[
                    "discretize"
                ].map_dict,
            }

            ### UPDATE GROUPBY ###
            self._set_mapped_groupby_discretize()

            # Add new group
            # NOTE: requires groupby to be set before post_split_transforms
            self.groupby["mapped"]["discretized_ctn_cols"] = {
                data_name: pipeline.named_steps["discretize"].discretized_groupby
                for data_name, pipeline in [
                    ("data", data_pipeline),
                    ("ground_truth", ground_truth_pipeline),
                ]
            }

            ## UPDATE nfeatures ##
            self.nfeatures["mapped"] = data_pipeline.named_steps["discretize"].nfeatures

            ## UPDATE columns ##
            self.columns["mapped"] = (
                self.columns["original"]
                .drop(
                    self.columns["original"][
                        self.col_idxs_by_type["original"]["continuous"]
                    ]  # drop cotinuous cols
                )
                .union(
                    [
                        f"{col_name}_{label}"
                        for col_name, col_info in self.discretizations["data"].items()
                        for label in col_info["labels"]
                    ],
                    sort=False,
                )
            )  # add back dicretized ctn cols

            ### UPDATE col indices ###
            # everything is now categorical
            self.col_idxs_by_type["mapped"] = {
                # might not need categorical sicne im just using for cemseloss?
                "categorical": list(range(self.nfeatures["mapped"])),
                "binary": list(self.groupby["mapped"].get("binary_vars", {}).keys()),
                # we don't just flip the ["original"]["onehot"] indices since we have new ones.
                "onehot": self._get_onehot_indices_from_groupby(self.groupby["mapped"]),
                "continuous": [],
            }

            ### Update columns after inversion ###
            # All continuous columns at the end
            self.columns["map-inverted"] = self.columns["original"][
                self.col_idxs_by_type["original"]["categorical"]
            ].union(
                self.columns["original"][
                    self.col_idxs_by_type["original"]["continuous"]
                ],
                sort=False,
            )

        elif self.feature_map == "target_encode_categorical":
            target_encoder = data_pipeline.named_steps["target_encode_categorical"]
            # TODO: should there be data and ground_truth?
            """
            We drop the "unknown" and "nan" handling mappings.
            When we're inverting there should be no nans, so we don't need nan inversion.
                1. the model output should not have any nans
                2. the ground truth should not have any nans. When we're semi_observed_training there will be no feature inversion. We only compute the loss in mapped space.
            When we're inverting we don't care about the unknown mapping, we will opt for the closes known category and fill with that.
            """
            # If this changes test_transforms needs to change
            self.inverse_target_encode_map = {
                "mapping": {
                    k: v.drop([-1, -2], axis=0, errors="ignore").dropna()
                    for k, v in target_encoder.mapping.items()
                },  # Dict[str, DataFrame]
                "ordinal_mapping": [
                    info["mapping"].drop([nan], axis=0, errors="ignore")
                    for info in target_encoder.ordinal_encoder.mapping
                ],  # List[Dict[str, Union[str, DataFrame, dtype]]]
            }

            # This changes the cardinality of the dataset
            if "combine_onehots" in data_pipeline.named_steps:
                ### UPDATE GROUPBY ###
                self.groupby["mapped"] = {
                    "combined_onehots": {
                        data_name: pipeline.named_steps[
                            "combine_onehots"
                        ].combined_onehot_groupby
                        for data_name, pipeline in [
                            ("data", data_pipeline),
                            ("ground_truth", ground_truth_pipeline),
                        ]
                    }
                }

                ### UPDATE nfeatures ###
                self.nfeatures["mapped"] = data_pipeline.named_steps[
                    "combine_onehots"
                ].nfeatures

                ### UPDATE col indices ###
                # everything is now continuous
                self.col_idxs_by_type["mapped"] = {
                    "continuous": list(range(self.nfeatures["mapped"])),
                    "binary": [],
                    "onehot": [],
                    "categorical": [],
                }

                ### UPDATE columns ###
                flat_onehot_indices = list(
                    chain.from_iterable(self.col_idxs_by_type["original"]["onehot"])
                )
                # drop them and tack them on at the end, if no combine-onehot its the same thing
                self.columns["map-inverted"] = (
                    self.columns["original"]
                    .drop(self.columns["original"][flat_onehot_indices])
                    .union(self.columns["original"][flat_onehot_indices], sort=False)
                )

    @staticmethod
    def _get_onehot_indices_from_groupby(
        groupby: Dict[str, Dict[int, str]]
    ) -> List[List[int]]:
        """
        groupby is map: idx -> group name. Want a list of onehot indices groups
        onehots includes onehot_prefixes and the discretized ctn cols
        If none exist this will return an empty list.
        """
        groups = {}
        groups.update(groupby.get("categorical_onehots", {}))
        groups.update(groupby.get("discretized_ctn_cols", {}).get("data", {}))

        onehot_group_indices = {}
        for idx, group_name in groups.items():
            if group_name in onehot_group_indices:
                onehot_group_indices[group_name].append(idx)
            else:
                onehot_group_indices[group_name] = [idx]
        return list(onehot_group_indices.values())

    ##################
    #  Data Loading  #
    ##################
    def _get_dataloader(
        self, split: str, apply_transform_adhoc: bool = False
    ) -> DataLoader:
        """
        Grab the split from all the Tensors in the self.splits dictionary.
        If there is no mapped data, "mapped": {} will be empty or not exist.
        """
        if apply_transform_adhoc:  # Apply transform 1-by-1 in Dataset __getitem__
            split_data = {k: v[split] for k, v in self.splits.items()}
        else:  # apply transform beforehand on the entire split
            split_data = {}
            data_feature_spaces = ["original"]
            if self.feature_map != "onehot_categorical":
                data_feature_spaces.append("mapped")
            for data_feature_space in data_feature_spaces:  # ["original", "mapped"]
                split_data[data_feature_space] = {}
                for (
                    data_role,
                    split_dfs,
                ) in self.splits.items():  # ["data", "ground_truth"]
                    if data_role != "label":  # apply transform to particular split
                        if self.transforms is not None:
                            split_data[data_feature_space][data_role] = self.transforms[
                                data_feature_space
                            ][data_role](split_dfs[split])
                        else:
                            split_data[data_feature_space][data_role] = split_dfs[split]

        return self._create_dataloader(split_data, apply_transform_adhoc)

    def _create_dataloader(
        self, split_data: Dict[str, Dict[str, DataFrame]], apply_transform_adhoc: bool
    ) -> DataLoader:
        """Packages data for pytorch Dataset/DataLoader."""
        is_longitudinal = self.data_type_time_dim in (
            DataTypeTimeDim.LONGITUDINAL,
            DataTypeTimeDim.LONGITUDINAL_SUBSET,
        )
        if apply_transform_adhoc:
            dataset = CommonDatasetWithTransform(
                split_data,
                self.transforms,
                is_longitudinal,
            )
        else:
            dataset = CommonTransformedDataset(split_data, is_longitudinal)
        loader = DataLoader(
            dataset,
            collate_fn=self._batch_collate if is_longitudinal else None,
            batch_size=self.batch_size,
            shuffle=False,
            prefetch_factor=256,
            persistent_workers=True,
            num_workers=self.num_workers,
            # don't use pin memory: https://discuss.pytorch.org/t/dataloader-method-acquire-of-thread-lock-objects/52943
            pin_memory=True,
        )
        return loader

    def _batch_collate(
        self, batch: List[Dict[str, Union[Dict[str, Tensor], Tensor]]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Batch is list of nested dictionaries: {
            "original": {"data": ..., "ground_truth": ...},
            "mapped": {"data": ..., "ground_truth": ...},
            "label": ...
        }
        Pad the variable length sequences, add seq lens, and enforce tensor.
        """
        # group all the data together:
        # convert list of nested dictionaries to a nested dictionary with lists inside
        collated_batch = {
            data_version: {
                data_role: [dic[data_version][data_role] for dic in batch]
                for data_role in data
            }
            for data_version, data in batch[0].items()
            if data_version != "label"  # ignore label for now
        }

        for data_version, nested_dict in collated_batch.items():
            # Get sequence lengths to store to pack/unpack later.
            collated_batch[data_version]["seq_len"] = Tensor(
                [len(pt_seq) for pt_seq in collated_batch[data_version]["data"]]
            )
            # Pad the data after storing seq lens
            for data_role in nested_dict:
                #  TODO: necessary to pad all the data going in ?
                collated_batch[data_version][data_role] = pad_sequence(
                    collated_batch[data_version][data_role],
                    batch_first=True,
                    padding_value=PAD_VALUE,
                )
        # restore label if we want to use it later
        collated_batch["label"] = [dic["label"] for dic in batch]

        return collated_batch

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
            "--feature-map",
            type=str,
            default=None,
            choices=[
                "onehot_categorical",
                "target_encode_categorical",
                "discretize_continuous",
            ],
            help="Specify how to map the features (e.g. continuous features to categorical space. Do nothing by default if you don't want to do anything to the features.",
        )
        p.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="When training the autoencoder, set the batch size.",
        )

        p.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of workers for the pytorch dataloader used in passing batches to the autoencoder.",
        )
        p.add_argument(
            "--scale",
            type=str2bool,
            default=False,
            help="When training the autoencoder, whether or not to scale the data before passing the data to the network.",
        )
        p.add_argument(
            "--uniform-prob",
            type=str2bool,
            default=False,
            help="When training an autoencoder with feature_map=discretize_continuous, whether or not to impose uniform distribution on missing values of onehot features.",
        )
        p.add_argument(
            "--separate_ground_truth_transform",
            type=str2bool,
            default=False,
            help="Specify whether or not to fit the ground_truth transforms separately on the ground_truth data. When false, the ground_truth transforms are the same as the data transforms (fit to the data).",
        )
        p.add_argument(
            "--limit-data",
            type=int,
            default=None,
            help="Debugging: limits the dataset to a certain size. Must be >= batch-size.",
        )

        #### AMPUTE ####
        p.add_argument(
            "--fully-observed",
            type=str2bool,
            default="no",
            help="Filter down to fully observed dataset flag.",
        )
        p.add_argument(
            "--percent-missing",
            type=float,
            default=None,
            help="When filtering down to fully observed and amputing (imputer is not none), what percent of data should be missing.",
        )
        p.add_argument(
            "--amputation-patterns",
            action=YAMLStringListToList(convert=dict),
            default=None,
            help="Patterns to pass to pyampute for amputation.",
        )
        return p
