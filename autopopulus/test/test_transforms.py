import unittest
from unittest.mock import patch
from Orange.data.variable import DiscreteVariable

import pandas as pd
import numpy as np
from Orange.data import Table, Domain
import torch


from autopopulus.data.transforms import (
    Discretizer,
    SimpleImpute,
    simple_impute_tensor,
    UniformProbabilityAcrossNans,
    undiscretize_tensor,
)
from test.common_mock_data import (
    columns,
    indices,
    X,
    y,
    splits,
    discretization,
    groupby,
)


class TestTransforms(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    ######################
    #       MOCKS        #
    ######################
    def mock_discretize(self, mock):
        # TODO: if amputing this doesn't work.
        mock.side_effect = [
            # .fit calls .transform sklearn.pipeline.py:303
            (X["disc"].iloc[splits["train"], :], discretization["discretizer_dict"]),
            # we call fit twice (once for data and once for ground_truth)
            (X["disc"].iloc[splits["train"], :], discretization["discretizer_dict"]),
            # transform calls for data, ground_truth for each split
            (X["disc"].iloc[splits["train"], :], discretization["discretizer_dict"]),
            (X["disc"].iloc[splits["train"], :], discretization["discretizer_dict"]),
            (X["disc"].iloc[splits["val"], :], discretization["discretizer_dict"]),
            (X["disc"].iloc[splits["val"], :], discretization["discretizer_dict"]),
            (X["disc"].iloc[splits["test"], :], discretization["discretizer_dict"]),
            (X["disc"].iloc[splits["test"], :], discretization["discretizer_dict"]),
        ]

    def mock_discretizer_dicts(self, mock_MDL):
        domain = Domain(
            [DiscreteVariable(col) for col in columns["onehot_continuous"]],
            class_vars=DiscreteVariable("class_var", values=["0", "1"]),
        )
        disc = None
        mock_MDL.return_value = (
            Table.from_numpy(
                domain,
                pd.concat(
                    [
                        X["disc"].loc[splits["train"], columns["onehot_continuous"]],
                        y.loc[splits["train"]],
                    ],
                    axis=1,
                ).values,
            ),
            disc,
        )

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

    ######################
    #       TESTS        #
    ######################
    def test_simple_impute_tensor(self):
        X_tensor = torch.tensor(X["X"].values)
        imputed = SimpleImpute(columns["ctn_cols"]).fit_transform(X["X"])
        imputed_tensor = simple_impute_tensor(
            X_tensor, indices["ctn_cols"], indices["cat_cols"]
        )
        self.assertTrue(np.allclose(imputed.values, imputed_tensor.numpy()))

    # @patch("autopopulus.data.MDLDiscretization.EntropyMDL._entropy_discretize_sorted")
    # for using orange for discretization
    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    def test_discretize(self, mock_cuts, mock_MDL):
        self.mock_discretizer_dicts(mock_MDL)
        mock_cuts.return_value = discretization["cuts"]

        discretize_transformer = Discretizer(
            columns["columns"], indices["ctn_cols"], return_info_dict=False
        ).fit(X["X"], y)

        # allows comparison of dictionaries with nparrays in them
        np.testing.assert_equal(
            discretize_transformer.map_dict, discretization["discretizer_dict"]
        )

        for splitidx in [splits["train"], splits["val"], splits["test"]]:
            self.assertTrue(
                np.array_equal(
                    discretize_transformer.transform(X["X"].iloc[splitidx, :]),
                    X["disc"].values[splitidx],
                    equal_nan=True,
                )
            )

    def test_uniform_prob(self):
        uniform_transformer = UniformProbabilityAcrossNans(
            groupby["categorical_only"]["discretize"]
        ).fit(X["X"].values[splits["train"]], y[splits["train"]])

        for splitidx in [splits["train"], splits["val"], splits["test"]]:
            self.assertTrue(
                np.array_equal(
                    uniform_transformer.transform(
                        (
                            X["disc"].iloc[splitidx, :],
                            discretization["discretizer_dict"],
                        )
                    ),
                    X["uniform"].values[splitidx],
                    equal_nan=True,
                )
            )

    def test_undiscretize(self):
        undiscretized_tensor = undiscretize_tensor(
            torch.tensor(X["disc_true"].values),
            groupby=groupby["after_fit"]["discretize"]["discretized_ctn_cols"]["data"],
            discretizations=discretization["discretizer_dict"],
            orig_columns=columns["columns"],
        )
        np.array_equal(
            undiscretized_tensor,
            torch.tensor(
                [
                    [np.mean((40, 80)), np.mean((0, 15.5)), 0, 0, 1, 0],
                    [np.mean((20, 40)), np.mean((50.5, 80.4)), 1, 0, 0, 1],
                    [np.mean((20, 40)), np.mean((15.5, 31.0)), 0, 0, 1, 0],
                    [np.mean((0, 20)), np.mean((50.5, 80.4)), 1, 1, 0, 0],
                    [np.mean((0, 20)), np.mean((50.5, 80.4)), 1, 0, 1, 0],
                    [np.mean((40, 80)), np.mean((15.5, 31.0)), 0, 1, 0, 0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
