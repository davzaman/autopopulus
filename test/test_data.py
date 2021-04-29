import unittest
from unittest.mock import patch
from Orange.data.variable import DiscreteVariable

import pandas as pd
import numpy as np
from Orange.data import Table, Domain

import sys
import os

import torch

# For running tests in VSCode
sys.path.insert(
    1, os.path.join(sys.path[0], "path/to/autopopulus")
)
from data.transforms import simple_impute, simple_impute_tensor
from data.utils import CommonDataModule

seed = 0
columns = ["age", "weight", "ismale", "fries_s", "fries_m", "fries_l"]
X_nomissing = pd.DataFrame(
    [
        [44, 15.1, 0, 0, 1, 0],
        [49, 57.2, 1, 0, 0, 1],
        [26, 26.3, 0, 0, 1, 0],
        [16, 73.4, 1, 1, 0, 0],
        [13, 56.5, 1, 0, 1, 0],
        [57, 29.6, 0, 1, 0, 0],
    ],
    columns=columns,
)
ctn_cols = ["age", "weight"]
discretizer_dict = {
    # TODO: discretizer eict for nonorange mix/max mocking
    # "age": {0: "0.0 - 20", 1: "20 - 40", 2: "40 - 44.0"},
    # "weight": {0: "0.0 - 15.5", 1: "15.5 - 31.0", 2: "31.0 - 50.5", 3: "50.5 - 57.2"},
    "age": {0: "0 - 20", 1: "20 - 40", 2: "40 - 80"},
    "weight": {0: "0 - 15.5", 1: "15.5 - 31.0", 2: "31.0 - 50.5", 3: "50.5 - 80.4"},
}
cuts = {0: [20, 40], 1: [15.5, 31.0, 50.5]}
num_categories = {"age": 3, "weight": 4, "fries": 3, "": 2}
# TODO: change discretizer so it matches original order
onehot_continuous = [
    f"{col}_{range_str}"
    for col in ctn_cols
    for range_str in discretizer_dict[col].values()
]
discretized_columns = ["ismale", "fries_s", "fries_m", "fries_l"] + onehot_continuous
onehot_prefix_names = ["fries"]
y = pd.Series([1, 0, 1, 0, 1, 0])

X = pd.DataFrame(
    [
        [44, np.nan, 0, 0, 1, 0],
        [39, 57.2, 1, 0, 0, 1],
        [26, 26.3, 0, np.nan, np.nan, np.nan],
        [16, 73.4, 1, 1, 0, 0],
        [np.nan, 56.5, 1, 0, 1, 0],
        [57, 29.6, 0, 1, 0, 0],
    ],
    columns=columns,
)

X_scale = pd.DataFrame(
    [
        [1, np.nan, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [-2.6, -30.9, 0, np.nan, np.nan, np.nan],
        [-4.6, 16.2, 1, 1, 0, 0],
        [np.nan, -0.7, 1, 0, 1, 0],
        [3.6, -27.6, 0, 1, 0, 0],
    ],
    columns=columns,
)

X_disc = pd.DataFrame(
    [
        [0, 0, 1, 0, 0, 0, 1, np.nan, np.nan, np.nan, np.nan],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, np.nan, np.nan, np.nan, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, np.nan, np.nan, np.nan, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    ],
    columns=discretized_columns,
)
X_disc_true = pd.DataFrame(
    [
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    ],
    columns=discretized_columns,
)

X_uniform = pd.DataFrame(
    [
        [0, 0, 1, 0, 0, 0, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 1 / 3, 1 / 3, 1 / 3, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    ],
    columns=discretized_columns,
)

train = [0, 1]
train_FO = [1]
val = [2, 3]
val_FO = [3]
test = [4, 5]
test_FO = [5]

standard = {
    "X": X,
    "y": y,
    "seed": seed,
    "val_test_size": 0.5,
    "test_size": 0.5,
    "batch_size": 2,
    "num_gpus": 0,
}
"""
https://docs.python.org/3/library/unittest.mock.html#where-to-patch
patch where an object is looked up, not where it is defined
"""


# for using github repo for discretization
def set_cuts(self):
    self._cuts = cuts


class TestTransforms(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_simple_impute_tensor(self):
        X_tensor = torch.tensor(X.values)
        imputed = simple_impute(X, ctn_cols, [])[0]
        ctn_col_idx = torch.LongTensor(
            [X.columns.get_loc(c) for c in ctn_cols if c in X]
        )
        cat_col_idx = torch.LongTensor(
            [X.columns.get_loc(c) for c in X.columns if c not in ctn_cols]
        )
        imputed_tensor = simple_impute_tensor(X_tensor, ctn_col_idx, cat_col_idx)
        self.assertTrue(np.allclose(imputed.values, imputed_tensor.numpy()))


class TestCommonDataModule(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.standard = standard

    def mock_discretize(self, mock):
        # TODO: if amputing this doesn't work.
        mock.return_value = (
            X_disc.iloc[train, :],
            X_disc.iloc[val, :],
            X_disc.iloc[test, :],
            X_disc.iloc[train, :],
            X_disc.iloc[val, :],
            X_disc.iloc[test, :],
            discretizer_dict,
        )

    def mock_discretizer_dicts(self, mock_dicts, mock_MDL):
        mock_dicts.return_value = discretizer_dict
        domain = Domain(
            [DiscreteVariable(col) for col in onehot_continuous],
            class_vars=DiscreteVariable("class_var", values=["0", "1"]),
        )
        disc = None
        mock_MDL.return_value = (
            Table.from_numpy(
                domain,
                pd.concat(
                    [X_disc.loc[train, onehot_continuous], y.loc[train]], axis=1
                ).values,
            ),
            disc,
        )

    def mock_sklearn_split(self, mock, fully_observed=False):
        trainidx = train_FO if fully_observed else train
        validx = val_FO if fully_observed else val
        testidx = test_FO if fully_observed else test
        # mock will return the next value from the iterable on each call
        mock.side_effect = [
            # first split: train, valtest
            (
                self.standard["X"].iloc[trainidx, :],
                self.standard["X"].iloc[validx + testidx, :],
                self.standard["y"].iloc[trainidx],
                self.standard["y"].iloc[validx + testidx],
            ),
            # second split: val, test
            (
                self.standard["X"].iloc[validx, :],
                self.standard["X"].iloc[testidx, :],
                self.standard["y"].iloc[validx],
                self.standard["y"].iloc[testidx],
            ),
        ]

    @patch("data.utils.train_test_split")
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
        data.setup()

        self.assertTrue(isinstance(data.X_train, pd.DataFrame))

        self.assertTrue(data.X_train.equals(X.iloc[train_FO, :]))
        self.assertTrue(data.y_train.equals(y.iloc[train_FO]))
        self.assertTrue(data.X_val.equals(X.iloc[val_FO, :]))
        self.assertTrue(data.y_val.equals(y.iloc[val_FO]))
        self.assertTrue(data.X_test.equals(X.iloc[test_FO, :]))
        self.assertTrue(data.y_test.equals(y.iloc[test_FO]))

    @patch("data.utils.train_test_split")
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
        data.setup()

        self.assertTrue(isinstance(data.X_train, pd.DataFrame))

        self.assertTrue(data.X_train.equals(X.iloc[train, :]))
        self.assertTrue(data.y_train.equals(y.iloc[train]))
        self.assertTrue(data.X_val.equals(X.iloc[val, :]))
        self.assertTrue(data.y_val.equals(y.iloc[val]))
        self.assertTrue(data.X_test.equals(X.iloc[test, :]))
        self.assertTrue(data.y_test.equals(y.iloc[test]))

    @patch("data.utils.train_test_split")
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
        data.setup()

        self.assertTrue(isinstance(data.X_train, pd.DataFrame))
        self.assertTrue(
            np.allclose(
                data.X_train.values, X_scale.values[train], atol=0.05, equal_nan=True
            )
        )
        self.assertTrue(
            np.allclose(
                data.X_val.values, X_scale.values[val], atol=0.05, equal_nan=True
            )
        )
        self.assertTrue(
            np.allclose(
                data.X_test.values, X_scale.values[test], atol=0.05, equal_nan=True
            )
        )
        self.assertTrue(
            np.allclose(
                data.X_true_train.values,
                X_scale.values[train],
                atol=0.05,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                data.X_true_val.values, X_scale.values[val], atol=0.05, equal_nan=True
            )
        )
        self.assertTrue(
            np.allclose(
                data.X_true_test.values, X_scale.values[test], atol=0.05, equal_nan=True
            )
        )

    # for using github repo for discretization
    @patch("lib.MDLPC.MDLP.MDLP_Discretizer.all_features_accepted_cutpoints", set_cuts)
    # for using orange for discretization
    @patch("utils.MDLDiscretization.get_discretized_MDL_data")
    @patch("utils.MDLDiscretization.list2dict")
    @patch("utils.MDLDiscretization.range_for_edge_bins", lambda x, y, z: [])
    @patch("data.utils.train_test_split")
    def test_discretize(self, mock_split, mock_disc_dict, mock_MDL):
        self.mock_sklearn_split(mock_split, fully_observed=False)
        self.mock_discretizer_dicts(mock_disc_dict, mock_MDL)

        #### Bad setup ####
        with self.assertRaises(AssertionError):
            CommonDataModule(
                **self.standard,
                fully_observed=False,
                scale=False,
                ampute=False,
                discretize=True,
                uniform_prob=False,
            )
        data = CommonDataModule(
            **self.standard,
            fully_observed=False,
            scale=False,
            ampute=False,
            discretize=True,
            uniform_prob=False,
            ctn_columns=ctn_cols,
            onehot_prefix_names=onehot_prefix_names,
        )
        data.setup()

        self.assertEqual(data.num_categories, num_categories)
        self.assertEqual(data.discretizations, discretizer_dict)

        self.assertTrue(isinstance(data.X_train, pd.DataFrame))
        self.assertTrue(
            np.array_equal(data.X_train.values, X_disc.values[train], equal_nan=True)
        )
        self.assertTrue(
            np.array_equal(data.X_val.values, X_disc.values[val], equal_nan=True)
        )
        self.assertTrue(
            np.array_equal(data.X_test.values, X_disc.values[test], equal_nan=True)
        )
        self.assertTrue(
            np.array_equal(
                data.X_true_train.values, X_disc.values[train], equal_nan=True
            )
        )

    @patch("data.utils.discretize")
    @patch("data.utils.train_test_split")
    def test_uniform_prob(self, mock_split, mock_disc):
        self.mock_sklearn_split(mock_split, fully_observed=False)
        self.mock_discretize(mock_disc)

        #### Bad setup ####
        with self.assertRaises(AssertionError):
            CommonDataModule(
                **self.standard,
                fully_observed=False,
                scale=False,
                ampute=False,
                discretize=False,
                uniform_prob=True,
            )
        with self.assertRaises(AssertionError):
            CommonDataModule(
                **self.standard,
                fully_observed=False,
                scale=False,
                ampute=False,
                discretize=False,
                uniform_prob=True,
                ctn_columns=ctn_cols,
                onehot_prefix_names=onehot_prefix_names,
            )
        with self.assertRaises(AssertionError):
            CommonDataModule(
                **self.standard,
                fully_observed=False,
                scale=False,
                ampute=False,
                discretize=True,
                uniform_prob=True,
            )

        data = CommonDataModule(
            **self.standard,
            fully_observed=False,
            scale=False,
            ampute=False,
            discretize=True,
            uniform_prob=True,
            ctn_columns=ctn_cols,
            onehot_prefix_names=onehot_prefix_names,
        )
        data.setup()
        self.assertTrue(isinstance(data.X_train, pd.DataFrame))
        self.assertTrue(np.array_equal(data.X_train.values, X_uniform.values[train]))
        self.assertTrue(np.array_equal(data.X_val.values, X_uniform.values[val]))
        self.assertTrue(np.array_equal(data.X_test.values, X_uniform.values[test]))
        self.assertTrue(
            np.array_equal(data.X_true_train.values, X_uniform.values[train])
        )


if __name__ == "__main__":
    unittest.main()
