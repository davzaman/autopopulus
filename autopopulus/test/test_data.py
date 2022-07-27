from typing import Any, List, Tuple
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal


# Local imports
from autopopulus.data import CommonDataModule
from test.common_mock_data import (
    columns,
    X,
    y,
    splits,
    seed,
    groupby,
    discretization,
    col_indices_by_type,
)
from autopopulus.data.dataset_classes import SimpleDatasetLoader


def get_dataset_loader(data, label) -> SimpleDatasetLoader:
    return SimpleDatasetLoader(
        data,
        label,
        **{
            "continuous_cols": columns["ctn_cols"],
            "categorical_cols": list(
                set(columns["columns"]) - set(columns["ctn_cols"])
            ),
            "missing_cols": columns["columns"],
            "observed_cols": columns["columns"],
            "onehot_prefixes": columns["onehot_prefix_names"],
        },
    )


standard = {
    "dataset_loader": get_dataset_loader(X["X"], y),
    "seed": seed,
    "val_test_size": 0.5,
    "test_size": 0.5,
    "batch_size": 2,
    "num_gpus": 0,
    "percent_missing": 0.33,
    "missingness_mechanism": "MCAR",
}
"""
https://docs.python.org/3/library/unittest.mock.html#where-to-patch
patch where an object is looked up, not where it is defined
"""


class TestSimpleDatasetLoader(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_load_feature_and_labels(self):
        combined = pd.concat([X["X"], y], axis=1)
        true_return = (X["X"], y)
        with self.subTest("String Label"):
            dataset_loader = get_dataset_loader(combined, "outcome")
            fn_return = dataset_loader.load_features_and_labels()
            assert_frame_equal(fn_return[0], true_return[0])
            assert_series_equal(fn_return[1], true_return[1])

        with self.subTest("Int Label"):
            # concat at the end
            label_idx = X["X"].shape[1]
            dataset_loader = get_dataset_loader(combined, label_idx)
            fn_return = dataset_loader.load_features_and_labels()
            assert_frame_equal(fn_return[0], true_return[0])
            assert_series_equal(fn_return[1], true_return[1])

        with self.subTest("Direct Labels"):
            dataset_loader = get_dataset_loader(X["X"], y)
            fn_return = dataset_loader.load_features_and_labels()
            assert_frame_equal(fn_return[0], true_return[0])
            assert_series_equal(fn_return[1], true_return[1])

    def test_filter_subgroup(self):
        with self.subTest("No Filter"):
            encoding = {"ismale": {"male": 1, "female": 0}}
            filtered_df = standard["dataset_loader"].filter_subgroup(
                X["nomissing"], encoding
            )
            self.assertTrue(filtered_df.equals(X["nomissing"]))

        with self.subTest("Filter"):
            dataset_loader = get_dataset_loader(X["X"], y)
            dataset_loader.subgroup_filter = {"ismale": "male"}
            dataset_loader.categorical_cols = list(
                set(columns["columns"]) - set(columns["ctn_cols"])
            )
            encoding = {"ismale": {"male": 1, "female": 0}}
            filtered_df = dataset_loader.filter_subgroup(X["nomissing"], encoding)
            correct_rows = [1, 3, 4]
            self.assertTrue(filtered_df.equals(X["nomissing"].iloc[correct_rows]))


class TestCommonDataModule(unittest.TestCase):
    # TODO: write tests for longitudinal data
    def setUp(self) -> None:
        super().setUp()
        self.standard = standard

    def mock_sklearn_split(self, mock, fully_observed=False):
        trainidx = splits["train_FO"] if fully_observed else splits["train"]
        validx = splits["val_FO"] if fully_observed else splits["val"]
        testidx = splits["test_FO"] if fully_observed else splits["test"]
        # mock will return the next value from the iterable on each call
        mock.side_effect = [
            # first split: trainval, test
            (trainidx + validx, testidx),
            # second split: train, val
            (trainidx, validx),
        ]

    def test_bad_args(self):
        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["dataset_loader"] = get_dataset_loader(X["X"], y)
            new_settings["dataset_loader"].continuous_cols = None

            CommonDataModule(**new_settings, discretize=True)

        with self.assertRaises(AssertionError):
            # no discretize but uniform prob
            CommonDataModule(
                **self.standard,
                fully_observed=False,
                scale=False,
                ampute=False,
                discretize=False,
                uniform_prob=True,
            )

        # ampute with missing necessary components
        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["percent_missing"] = None
            CommonDataModule(**new_settings, ampute=True)

        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["missingness_mechanism"] = None
            CommonDataModule(**new_settings, ampute=True)

        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["dataset_loader"] = get_dataset_loader(X["X"], y)
            new_settings["dataset_loader"].missing_cols = None
            CommonDataModule(**new_settings, ampute=True)

    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_set_groupby(self, mock_split, mock_cuts):
        mock_cuts.return_value = discretization["cuts"]

        with self.subTest("No Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(**self.standard)
            data.columns = X["X"].columns
            data._set_groupby()
            np.testing.assert_equal(
                data.groupby, groupby["categorical_only"]["no_discretize"]
            )

            # Test updated after fit
            data.setup("fit")
            np.testing.assert_equal(data.groupby, groupby["after_fit"]["no_discretize"])

        with self.subTest("With Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(**self.standard, discretize=True)
            data.columns = X["X"].columns
            data._set_groupby()
            # Test before fit
            np.testing.assert_equal(
                data.groupby, groupby["categorical_only"]["discretize"]
            )

            # Test updated with discretizations after fit
            data.setup("fit")
            np.testing.assert_equal(data.groupby, groupby["after_fit"]["discretize"])

    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_set_nfeatures(self, mock_split, mock_cuts):
        mock_cuts.return_value = discretization["cuts"]

        with self.subTest("No Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(**self.standard)
            data.setup("fit")
            np.testing.assert_equal(data.n_features, len(columns["columns"]))

        with self.subTest("With Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(**self.standard, discretize=True)
            data.columns = X["X"].columns
            data.setup("fit")
            np.testing.assert_equal(
                data.n_features, len(columns["discretized_columns"])
            )

    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_set_col_indices_by_type(self, mock_split, mock_cuts):
        mock_cuts.return_value = discretization["cuts"]

        with self.subTest("No Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(**self.standard)
            data.setup("fit")
            np.testing.assert_equal(
                data.col_indices_by_type["continuous"],
                col_indices_by_type["continuous"],
            )
            np.testing.assert_equal(
                data.col_indices_by_type["categorical"],
                col_indices_by_type["categorical"],
            )

    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_set_post_split_transforms(self, mock_split, mock_cuts):
        mock_cuts.return_value = discretization["cuts"]

        with self.subTest("No Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(**self.standard)
            data.setup("fit")
            for transform_dict in data.transforms.values():
                self.assertEqual(list(transform_dict.keys()), ["normal"])
            # no separate ground truth transform
            # np.testing.assert_equal(
            np.testing.assert_array_equal(
                data.transforms["data"]["normal"](X["X"]),
                data.transforms["ground_truth"]["normal"](X["X"]),
            )

            with self.subTest("With Separate Ground Truth Transform"):
                self.mock_sklearn_split(mock_split)
                data = CommonDataModule(
                    **self.standard, separate_ground_truth_transform=True, scale=True
                )
                data.setup("fit")
                # Make ground truth something very different
                data.splits["ground_truth"]["normal"]["train"] = pd.DataFrame(
                    np.full_like(data.splits["ground_truth"]["normal"]["train"], 600),
                    columns=data.splits["ground_truth"]["normal"]["train"].columns,
                )
                data._set_post_split_transforms()

                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_equal,
                    data.transforms["data"]["normal"](X["X"]),
                    data.transforms["ground_truth"]["normal"](X["X"]),
                )

        with self.subTest("With Discretization"):
            self.mock_sklearn_split(mock_split)
            data = CommonDataModule(
                **self.standard, separate_ground_truth_transform=True, discretize=True
            )
            data.setup("fit")
            for transform_dict in data.transforms.values():
                self.assertEqual(list(transform_dict.keys()), ["normal", "discretized"])

            with self.subTest("With Separate Ground Truth Transform"):
                self.mock_sklearn_split(mock_split)
                data = CommonDataModule(
                    **self.standard,
                    separate_ground_truth_transform=True,
                    scale=True,
                    discretize=True,
                )
                data.setup("fit")
                # Make ground truth something very different
                # Check "normal" even though we're discretizing, harder to mock that
                data.splits["ground_truth"]["normal"]["train"] = pd.DataFrame(
                    np.full_like(data.splits["ground_truth"]["normal"]["train"], 600),
                    columns=data.splits["ground_truth"]["normal"]["train"].columns,
                )
                data._set_post_split_transforms()

                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_equal,
                    data.transforms["data"]["normal"](X["X"]),
                    data.transforms["ground_truth"]["normal"](X["X"]),
                )

    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_fullyobserved_true(self, mock_split):
        self.mock_sklearn_split(mock_split, fully_observed=True)

        data = CommonDataModule(
            **self.standard,
            fully_observed=True,
            scale=False,
            ampute=False,
            discretize=False,
            uniform_prob=False,
        )
        data.setup("fit")

        self.assertTrue(
            isinstance(data.splits["data"]["normal"]["train"], pd.DataFrame)
        )

        for splitname, splitidx in [
            ("train", splits["train_FO"]),
            ("val", splits["val_FO"]),
            ("test", splits["test_FO"]),
        ]:
            self.assertTrue(
                data.splits["data"]["normal"][splitname].equals(X["X"].iloc[splitidx])
            )
            self.assertTrue(data.splits["label"][splitname].equals(y.iloc[splitidx]))

    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_fullyobserved_false(self, mock_split):
        self.mock_sklearn_split(mock_split, fully_observed=False)

        data = CommonDataModule(
            **self.standard,
            fully_observed=False,
            scale=False,
            ampute=False,
            discretize=False,
            uniform_prob=False,
        )
        data.setup("fit")

        self.assertTrue(
            isinstance(data.splits["data"]["normal"]["train"], pd.DataFrame)
        )

        for splitname, splitidx in [
            ("train", splits["train"]),
            ("val", splits["val"]),
            ("test", splits["test"]),
        ]:
            self.assertTrue(
                data.splits["data"]["normal"][splitname].equals(X["X"].iloc[splitidx])
            )
            self.assertTrue(data.splits["label"][splitname].equals(y.iloc[splitidx]))

    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_scale(self, mock_split):
        pd.options.mode.chained_assignment = "raise"  # print the stack
        self.mock_sklearn_split(mock_split, fully_observed=False)

        data = CommonDataModule(
            **self.standard,
            fully_observed=False,
            scale=True,
            ampute=False,
            discretize=False,
            uniform_prob=False,
        )
        data.setup("fit")

        self.assertTrue(
            isinstance(data.splits["data"]["normal"]["train"], pd.DataFrame)
        )
        for splitname, splitidx in [
            ("train", splits["train"]),
            ("val", splits["val"]),
            ("test", splits["test"]),
        ]:
            for dataname in ["data", "ground_truth"]:
                transformed_data = data.transforms[dataname]["normal"](
                    data.splits[dataname]["normal"][splitname]
                )
                self.assertTrue(
                    np.allclose(
                        transformed_data,
                        X["scale"].values[splitidx],
                        atol=0.05,
                        equal_nan=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
