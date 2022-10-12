import unittest
from itertools import chain
from unittest.mock import patch

import pandas as pd
import numpy as np
from numpy.random import default_rng
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.utils.validation import check_is_fitted

from hypothesis import assume, given, HealthCheck, settings, strategies as st
from hypothesis.extra.pandas import data_frames


# Local imports
from autopopulus.data import CommonDataModule
from autopopulus.test.common_mock_data import (
    hypothesis,
    columns,
    X,
    y,
    seed,
)
from autopopulus.data.dataset_classes import SimpleDatasetLoader
from autopopulus.data.transforms import identity
from autopopulus.test.utils import (
    build_onehot_from_hypothesis,
    create_fake_disc_data,
    mock_disc_data,
)
from data.utils import onehot_multicategorical_column


def get_dataset_loader(data, label) -> SimpleDatasetLoader:
    return SimpleDatasetLoader(
        data,
        label,
        **{
            "continuous_cols": columns["ctn_cols"],
            "categorical_cols": list(
                set(columns["columns"]) - set(columns["ctn_cols"])
            ),
            "onehot_prefixes": columns["onehot_prefix_names"],
        },
    )


standard = {
    # "dataset_loader": get_dataset_loader(X["X"], y),
    "seed": seed,
    "val_test_size": 0.5,
    "test_size": 0.5,
    "batch_size": 2,
    "num_gpus": 0,
    "percent_missing": 0.33,
    # "missingness_mechanism": "MCAR",
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
            filtered_df = get_dataset_loader(X["X"], y).filter_subgroup(
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
    def setUp(self) -> None:
        super().setUp()
        self.standard = standard

    def test_bad_args(self):
        dataset_loader = get_dataset_loader(X["X"], y)
        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["dataset_loader"] = dataset_loader
            new_settings["dataset_loader"].continuous_cols = None

            CommonDataModule(**new_settings, feature_map="discretize_continuous")

        with self.assertRaises(AssertionError):
            # no discretize but uniform prob
            CommonDataModule(
                **self.standard,
                dataset_loader=dataset_loader,
                fully_observed=False,
                scale=False,
                ampute=False,
                feature_map=None,
                uniform_prob=True,
            )

        # ampute with missing necessary components
        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["percent_missing"] = None
            CommonDataModule(**new_settings, dataset_loader=dataset_loader, ampute=True)

        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            new_settings["amputation_patterns"] = None
            CommonDataModule(**new_settings, dataset_loader=dataset_loader, ampute=True)

        with self.assertRaises(AssertionError):
            new_settings = self.standard.copy()
            dataset_loader = get_dataset_loader(X["X"], y)
            dataset_loader.missing_cols = None
            CommonDataModule(**new_settings, dataset_loader=dataset_loader, ampute=True)

        with self.assertRaises(TypeError):
            # Need to pass a dataloader
            CommonDataModule()

        with self.assertRaises(AssertionError):
            # If no val_test_split and test_split size, need split_ids in dataset loader
            CommonDataModule(get_dataset_loader(X["X"], y))

    #########################
    #     BASIC TESTING     #
    #########################
    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_onehot(self, mock_split, mock_disc_cuts, mock_MDL, df):
        """
        Tests all the subfunctions of setup() (including the routines those functions call), ignoring different feature_maps.
        """
        # Ensure all categories/cols present for testing
        assume(
            np.array_equal(
                df.nunique()[hypothesis["onehot_prefixes"]].values, np.array([4, 3])
            )
        )
        df = build_onehot_from_hypothesis(df, hypothesis["onehot_prefixes"])

        nsamples = len(df)
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome
        datasetloader_args = (
            df,
            y,
            hypothesis["onehot"]["ctn_cols"],
            hypothesis["onehot"]["cat_cols"],
        )

        mock_split.return_value = (df.index, df.index)

        data = CommonDataModule(
            dataset_loader=SimpleDatasetLoader(
                *datasetloader_args, hypothesis["onehot_prefixes"]
            ),
            **self.standard,
        )
        data.columns = {"original": df.columns}

        # needs to happen before set_groupby
        with self.subTest("col_idxs_by_type"):  # --------------------------
            data._set_col_idxs_by_type()
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["continuous"],
                hypothesis["onehot"]["ctn_cols_idx"],
            )
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["categorical"],
                hypothesis["onehot"]["cat_cols_idx"],
            )
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["onehot"],
                hypothesis["onehot"]["onehot_cols_idx"],
            )
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["binary"],
                hypothesis["onehot"]["bin_cols_idx"],
            )

        with self.subTest("groupby"):  # -----------------------------------
            data._set_groupby()
            self.assertDictEqual(
                data.groupby,
                {
                    "original": {
                        "binary_vars": dict(
                            zip(
                                hypothesis["onehot"]["bin_cols_idx"],
                                hypothesis["onehot"]["bin_cols"],
                            )
                        ),
                        "categorical_onehots": dict(
                            zip(
                                chain.from_iterable(
                                    hypothesis["onehot"]["onehot_cols_idx"]
                                ),
                                hypothesis["onehot"]["onehot_expanded_prefixes"],
                            )
                        ),
                    }
                },
            )

        with self.subTest("nfeatures"):
            data._set_nfeatures()
            self.assertDictEqual(data.nfeatures, {"original": 10})

        with self.subTest("set_post_split_transforms"):
            # don't allow infinite values for the transforms
            assume(np.isinf(df).values.sum() == 0)

            data._split_dataset(df, df, y)
            data._set_post_split_transforms()
            self.assertEqual(data.transforms, None)

            cuts = [[(0, 1), (1, 2)], [(0, 0.5), (0.5, 1), (1, 1.5)]]
            category_names = [
                ["0 - 1", "1 - 2"],
                ["0 - 0.5", "0.5 - 1", "1 - 1.5"],
            ]
            disc_data = create_fake_disc_data(
                rng, nsamples, cuts, category_names, hypothesis["ctn_cols"]
            )
            mock_disc_data(mock_MDL, disc_data, y)
            mock_disc_cuts.return_value = cuts

            # the point isn't to integration test the transforms
            # it's to check we have the info we need and named correctly
            with self.subTest("No Feature Map, Yes Transform"):
                # assume(not df.empty)
                # We will test this separately
                with self.subTest("No Separate Ground Truth Pipeline"):
                    data = CommonDataModule(
                        dataset_loader=SimpleDatasetLoader(
                            *datasetloader_args, hypothesis["onehot_prefixes"]
                        ),
                        **self.standard,
                        scale=True,
                    )
                    data.columns = {"original": df.columns}
                    data._set_col_idxs_by_type()
                    data._set_groupby()
                    data._set_nfeatures()
                    data._split_dataset(df, df, y)
                    data._set_post_split_transforms()
                    for data_name in ["data", "ground_truth"]:
                        # since the transform is a bound method get the instance via __self__
                        self.assertEqual(
                            list(
                                data.transforms["original"][
                                    data_name
                                ].__self__.named_steps.keys()
                            ),
                            ["enforce-numpy", "continuous-scale", "enforce-pandas"],
                        )

                    np.testing.assert_array_equal(
                        data.transforms["original"]["data"](df),
                        data.transforms["original"]["ground_truth"](df),
                    )
            with self.subTest("No Transform, Feature Map Integration"):
                with self.subTest("discretize_continuous"):
                    data = CommonDataModule(
                        dataset_loader=SimpleDatasetLoader(
                            *datasetloader_args,
                            onehot_prefixes=hypothesis["onehot_prefixes"],
                        ),
                        **self.standard,
                        feature_map="discretize_continuous",
                        uniform_prob=True,
                    )
                    data.columns = {"original": df.columns}
                    data._set_col_idxs_by_type()
                    data._set_groupby()
                    data._set_nfeatures()
                    data._split_dataset(df, df, y)
                    data._set_post_split_transforms()

                    # should have both original and mapped now
                    self.assertEqual(
                        list(data.transforms.keys()), ["original", "mapped"]
                    )
                    for data_name in ["data", "ground_truth"]:
                        # original transform should do nothing (no transform set)
                        self.assertEqual(
                            data.transforms["original"][data_name], identity
                        )

                        self.assertEqual(
                            list(
                                data.transforms["mapped"][
                                    data_name
                                ].__self__.named_steps.keys()
                            ),
                            ["discretize", "uniform_probability_across_nans"],
                        )

                        discretizer_obj = data.transforms["mapped"][
                            data_name
                        ].__self__.named_steps["discretize"]
                        # yes since we are uniformprob
                        self.assertTrue(discretizer_obj.return_info_dict)
                        np.testing.assert_array_equal(
                            discretizer_obj.orig_cols, df.columns
                        )
                        np.testing.assert_array_equal(
                            discretizer_obj.ctn_cols_idx,
                            hypothesis["onehot"]["ctn_cols_idx"],
                        )

                        # doesn't have discretizer dict
                        self.assertDictEqual(
                            data.transforms["mapped"][data_name]
                            .__self__.named_steps["uniform_probability_across_nans"]
                            .groupby_categorical_only,
                            {
                                "categorical_onehots": dict(
                                    zip(
                                        hypothesis["onehot"]["onehot_cols"],
                                        hypothesis["onehot"][
                                            "onehot_expanded_prefixes"
                                        ],
                                    )
                                ),
                                "binary_vars": dict(
                                    zip(
                                        hypothesis["onehot"]["bin_cols"],
                                        hypothesis["onehot"]["bin_cols"],
                                    )
                                ),
                            },
                        )

                with self.subTest("target_encode_categorical"):
                    data = CommonDataModule(
                        dataset_loader=SimpleDatasetLoader(
                            *datasetloader_args, hypothesis["onehot_prefixes"]
                        ),
                        **self.standard,
                        feature_map="target_encode_categorical",
                    )
                    data.columns = {"original": df.columns}
                    data._set_col_idxs_by_type()
                    data._set_groupby()
                    data._set_nfeatures()
                    data._split_dataset(df, df, y)
                    data._set_post_split_transforms()
                    # If i called setup this would set auxilliary data, I want the intermediate stuff

                    # should have both original and mapped now
                    self.assertEqual(
                        list(data.transforms.keys()), ["original", "mapped"]
                    )
                    for data_name in [
                        "data",
                        "ground_truth",
                    ]:  # Do nothing, no transform
                        self.assertEqual(
                            data.transforms["original"][data_name], identity
                        )
                    # since there's onehots we should be combining onehots
                    for data_name in ["data", "ground_truth"]:
                        self.assertEqual(
                            list(
                                data.transforms["mapped"][
                                    data_name
                                ].__self__.named_steps.keys()
                            ),
                            ["combine_onehots", "target_encode"],
                        )

                        # The onehot_groupby and columns passed to combineonehots should be in original space
                        combine_onehots_obj = data.transforms["mapped"][
                            data_name
                        ].__self__.named_steps["combine_onehots"]

                        self.assertDictEqual(
                            combine_onehots_obj.onehot_groupby,
                            dict(
                                zip(
                                    hypothesis["onehot"]["onehot_cols"],
                                    hypothesis["onehot"]["onehot_expanded_prefixes"],
                                )
                            ),
                        )

                        # The categorical columns for the targetencoder should be in combined space
                        passed_cat_cols = (
                            data.transforms["mapped"][data_name]
                            .__self__.named_steps["target_encode"]
                            .cols
                        )
                        np.testing.assert_array_equal(
                            passed_cat_cols, ["bin", "mult1", "mult2"]
                        )

            with self.subTest("Yes Transform, Yes Feature Map"):
                data = CommonDataModule(
                    dataset_loader=SimpleDatasetLoader(
                        *datasetloader_args, hypothesis["onehot_prefixes"]
                    ),
                    **self.standard,
                    feature_map="target_encode_categorical",
                    scale=True,
                )
                data.columns = {"original": df.columns}
                data._set_col_idxs_by_type()
                data._set_groupby()
                data._set_nfeatures()
                data._split_dataset(df, df, y)
                data._set_post_split_transforms()
                for data_name in ["data", "ground_truth"]:
                    self.assertEqual(
                        list(
                            data.transforms["original"][
                                data_name
                            ].__self__.named_steps.keys()
                        ),
                        ["enforce-numpy", "continuous-scale", "enforce-pandas"],
                    )
                for data_name in ["data", "ground_truth"]:
                    self.assertEqual(
                        list(
                            data.transforms["mapped"][
                                data_name
                            ].__self__.named_steps.keys()
                        ),
                        [
                            "enforce-numpy",
                            "continuous-scale",
                            "enforce-pandas",
                            "combine_onehots",
                            "target_encode",
                        ],
                    )

    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    @settings(suppress_health_check=[HealthCheck(3)], deadline=None)
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_multicat(self, mock_split, mock_disc_cuts, mock_MDL, df):
        """
        Tests all the subfunctions of setup() (including the routines those functions call), ignoring different feature_maps.
        """
        nsamples = len(df)
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome
        datasetloader_args = (df, y, hypothesis["ctn_cols"], hypothesis["cat_cols"])

        mock_split.return_value = (df.index, df.index)

        data = CommonDataModule(
            dataset_loader=SimpleDatasetLoader(*datasetloader_args),
            **self.standard,
        )
        data.columns = {"original": df.columns}

        # needs to happen before set_groupby
        with self.subTest("col_idxs_by_type"):  # --------------------------
            data._set_col_idxs_by_type()
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["continuous"],
                hypothesis["ctn_cols_idx"],
            )
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["categorical"],
                hypothesis["cat_cols_idx"],
            )
            # all cat cols assumed to be binary since not target encoding
            np.testing.assert_equal(
                data.col_idxs_by_type["original"]["binary"],
                hypothesis["cat_cols_idx"],
            )
            self.assertTrue("onehot" not in data.col_idxs_by_type["original"])

        with self.subTest("groupby"):  # -----------------------------------
            data._set_groupby()
            self.assertDictEqual(
                data.groupby,
                {
                    "original": {
                        "binary_vars": {
                            idx: df.columns[idx] for idx in hypothesis["cat_cols_idx"]
                        }
                    }
                },
            )

        with self.subTest("nfeatures"):
            data._set_nfeatures()
            self.assertDictEqual(data.nfeatures, {"original": 5})

        with self.subTest("set_post_split_transforms"):
            # don't allow infinite values for the transforms
            assume(np.isinf(df).values.sum() == 0)

            data._split_dataset(df, df, y)
            data._set_post_split_transforms()
            self.assertEqual(data.transforms, None)

            cuts = [[(0, 1), (1, 2)], [(0, 0.5), (0.5, 1), (1, 1.5)]]
            category_names = [
                ["0 - 1", "1 - 2"],
                ["0 - 0.5", "0.5 - 1", "1 - 1.5"],
            ]
            disc_data = create_fake_disc_data(
                rng, nsamples, cuts, category_names, hypothesis["ctn_cols"]
            )
            mock_disc_data(mock_MDL, disc_data, y)
            mock_disc_cuts.return_value = cuts

            # the point isn't to integration test the transforms
            # it's to check we have the info we need and named correctly
            with self.subTest("No Feature Map, Yes Transform"):
                assume(not df.empty)
                # We will test this separately
                with self.subTest("No Separate Ground Truth Pipeline"):
                    data = CommonDataModule(
                        dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                        **self.standard,
                        scale=True,
                    )
                    data.columns = {"original": df.columns}
                    data._set_col_idxs_by_type()
                    data._set_groupby()
                    data._set_nfeatures()
                    data._split_dataset(df, df, y)
                    data._set_post_split_transforms()
                    for data_name in ["data", "ground_truth"]:
                        # since the transform is a bound method get the instance via __self__
                        self.assertEqual(
                            list(
                                data.transforms["original"][
                                    data_name
                                ].__self__.named_steps.keys()
                            ),
                            ["enforce-numpy", "continuous-scale", "enforce-pandas"],
                        )

                    np.testing.assert_array_equal(
                        data.transforms["original"]["data"](df),
                        data.transforms["original"]["ground_truth"](df),
                    )
            with self.subTest("No Transform, Feature Map Integration"):
                with self.subTest("discretize_continuous"):
                    data = CommonDataModule(
                        dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                        **self.standard,
                        feature_map="discretize_continuous",
                        uniform_prob=True,
                    )
                    data.columns = {"original": df.columns}
                    data._set_col_idxs_by_type()
                    data._set_groupby()
                    data._set_nfeatures()
                    data._split_dataset(df, df, y)
                    data._set_post_split_transforms()

                    # should have both original and mapped now
                    self.assertEqual(
                        list(data.transforms.keys()), ["original", "mapped"]
                    )
                    for data_name in ["data", "ground_truth"]:
                        # original transform should do nothing (no transform set)
                        self.assertEqual(
                            data.transforms["original"][data_name], identity
                        )

                        self.assertEqual(
                            list(
                                data.transforms["mapped"][
                                    data_name
                                ].__self__.named_steps.keys()
                            ),
                            ["discretize", "uniform_probability_across_nans"],
                        )

                        discretizer_obj = data.transforms["mapped"][
                            data_name
                        ].__self__.named_steps["discretize"]
                        # yes since we are uniformprob
                        self.assertTrue(discretizer_obj.return_info_dict)
                        np.testing.assert_array_equal(
                            discretizer_obj.orig_cols, df.columns
                        )
                        np.testing.assert_array_equal(
                            discretizer_obj.ctn_cols_idx, hypothesis["ctn_cols_idx"]
                        )

                        # doesn't have discretizer dict
                        self.assertDictEqual(
                            data.transforms["mapped"][data_name]
                            .__self__.named_steps["uniform_probability_across_nans"]
                            .groupby_categorical_only,
                            {
                                "binary_vars": dict(
                                    zip(hypothesis["cat_cols"], hypothesis["cat_cols"])
                                )
                            },
                        )

                with self.subTest("target_encode_categorical"):
                    data = CommonDataModule(
                        dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                        **self.standard,
                        feature_map="target_encode_categorical",
                    )
                    data.columns = {"original": df.columns}
                    data._set_col_idxs_by_type()
                    data._set_groupby()
                    data._set_nfeatures()
                    data._split_dataset(df, df, y)
                    data._set_post_split_transforms()
                    # should have both original and mapped now
                    self.assertEqual(
                        list(data.transforms.keys()), ["original", "mapped"]
                    )
                    for data_name in ["data", "ground_truth"]:
                        # original transform should do nothing (no transform set)
                        self.assertEqual(
                            data.transforms["original"][data_name], identity
                        )

                        self.assertEqual(
                            list(
                                data.transforms["mapped"][
                                    data_name
                                ].__self__.named_steps.keys()
                            ),
                            ["target_encode"],
                        )

                        # The categorical columns for the targetencoder should be the same as the categorical cols for original data
                        passed_cat_cols = (
                            data.transforms["mapped"][data_name]
                            .__self__.named_steps["target_encode"]
                            .cols
                        )
                        self.assertEqual(passed_cat_cols, hypothesis["cat_cols"])

            with self.subTest("Yes Transform, Yes Feature Map"):
                data = CommonDataModule(
                    dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                    **self.standard,
                    feature_map="target_encode_categorical",
                    scale=True,
                )
                data.columns = {"original": df.columns}
                data._set_col_idxs_by_type()
                data._set_groupby()
                data._set_nfeatures()
                data._split_dataset(df, df, y)
                data._set_post_split_transforms()
                for data_name in ["data", "ground_truth"]:
                    self.assertEqual(
                        list(
                            data.transforms["original"][
                                data_name
                            ].__self__.named_steps.keys()
                        ),
                        ["enforce-numpy", "continuous-scale", "enforce-pandas"],
                    )
                for data_name in ["data", "ground_truth"]:
                    self.assertEqual(
                        list(
                            data.transforms["mapped"][
                                data_name
                            ].__self__.named_steps.keys()
                        ),
                        [
                            "enforce-numpy",
                            "continuous-scale",
                            "enforce-pandas",
                            "target_encode",
                        ],
                    )

    #########################
    #    AUXILLIARY INFO    #
    #########################
    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_set_auxilliary_info_post_mapping_static_onehot(
        self, mock_split, mock_cuts, mock_MDL, df
    ):
        # Ensure all categories/cols present for testing
        assume(
            np.array_equal(
                df.nunique()[hypothesis["onehot_prefixes"]].values, np.array([4, 3])
            )
        )
        df = build_onehot_from_hypothesis(df, hypothesis["onehot_prefixes"])

        nsamples = len(df)
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome
        datasetloader_args = (
            df,
            y,
            hypothesis["onehot"]["ctn_cols"],
            hypothesis["onehot"]["cat_cols"],
        )

        mock_split.return_value = (df.index, df.index)

        data = CommonDataModule(
            dataset_loader=SimpleDatasetLoader(
                *datasetloader_args, hypothesis["onehot_prefixes"]
            ),
            **self.standard,
        )
        data.columns = {"original": df.columns}

        data.setup("fit")
        self.assertIsNone(data.discretizations)
        self.assertIsNone(data.inverse_target_encode_map)
        # nothing should have a "mapped"
        self.assertTrue("mapped" not in data.groupby)
        self.assertTrue("mapped" not in data.nfeatures)
        self.assertTrue("mapped" not in data.col_idxs_by_type)
        self.assertTrue("mapped" not in data.columns)

        with self.subTest("With Discretization"):
            cuts = [[(0, 1), (1, 2)], [(0, 0.5), (0.5, 1), (1, 1.5)]]
            category_names = [["0 - 1", "1 - 2"], ["0 - 0.5", "0.5 - 1", "1 - 1.5"]]
            disc_data = create_fake_disc_data(
                rng, nsamples, cuts, category_names, hypothesis["ctn_cols"]
            )

            mock_disc_data(mock_MDL, disc_data, y)
            mock_cuts.return_value = cuts

            data = CommonDataModule(
                dataset_loader=SimpleDatasetLoader(
                    *datasetloader_args, hypothesis["onehot_prefixes"]
                ),
                **self.standard,
                feature_map="discretize_continuous",
            )
            data.columns = {"original": df.columns}
            data.setup("fit")

            self.assertIsNotNone(data.discretizations)

            # leave it up to test_transforms to see if discretized_ctn_cols is computed correctly
            self.assertEqual(
                list(data.groupby["mapped"].keys()),
                ["categorical_onehots", "binary_vars", "discretized_ctn_cols"],
            )
            # shifted indices
            binary_idxs = [0]
            self.assertDictEqual(
                data.groupby["mapped"]["binary_vars"],
                dict(
                    zip(
                        binary_idxs,
                        hypothesis["onehot"]["bin_cols"],
                    )
                ),
            )
            onehot_groups = [[1, 2, 3, 4], [5, 6, 7]]
            self.assertDictEqual(
                data.groupby["mapped"]["categorical_onehots"],
                dict(
                    zip(
                        chain.from_iterable(onehot_groups),
                        hypothesis["onehot"]["onehot_expanded_prefixes"],
                    )
                ),
            )
            self.assertEqual(data.nfeatures["mapped"], 10 + 5 - 2)
            np.testing.assert_array_equal(
                data.columns["mapped"],
                ["bin"]
                + hypothesis["onehot"]["onehot_cols"]
                + [
                    f"{col}_{bin_name}"
                    for i, col in enumerate(hypothesis["onehot"]["ctn_cols"])
                    for bin_name in category_names[i]
                ],
            )
            self.assertEqual(
                data.col_idxs_by_type["mapped"],
                {
                    "categorical": list(range(13)),
                    "binary": binary_idxs,
                    # add discretized
                    "onehot": onehot_groups + [[8, 9], [10, 11, 12]],
                    "continuous": [],
                },
            )
            self.assertListEqual(
                data.columns["map-inverted"].tolist(),
                ["bin"] + hypothesis["onehot"]["onehot_cols"] + ["ctn1", "ctn2"],
            )

        with self.subTest("With Target Encoding"):
            data = CommonDataModule(
                dataset_loader=SimpleDatasetLoader(
                    *datasetloader_args, hypothesis["onehot_prefixes"]
                ),
                **self.standard,
                feature_map="target_encode_categorical",
            )
            data.columns = {"original": df.columns}
            data.setup("fit")

            self.assertEqual(
                list(data.inverse_target_encode_map.keys()),
                ["inverse_transform", "mapping"],
            )
            self.assertIsNotNone(data.inverse_target_encode_map["inverse_transform"])

            # will have reordering and all the binary and multicat vars should be encoded
            self.assertEqual(
                list(data.inverse_target_encode_map["mapping"].keys()),
                [0, 3, 4],
            )

            # Leave it to test_transforms to see if combined_onehots is right
            self.assertEqual(
                list(data.groupby["mapped"].keys()),
                ["combined_onehots"],
            )

            self.assertEqual(data.nfeatures["mapped"], 5)
            self.assertEqual(
                data.col_idxs_by_type["mapped"],
                {
                    "continuous": list(range(5)),
                    "binary": [],
                    "onehot": [],
                    "categorical": [],
                },
            )
            self.assertEqual(
                data.columns["mapped"].tolist(),
                ["bin", "ctn1", "ctn2", "mult1", "mult2"],
            )
            self.assertEqual(
                data.columns["map-inverted"].tolist(),
                ["bin", "ctn1", "ctn2"] + hypothesis["onehot"]["onehot_cols"],
            )

    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    @settings(suppress_health_check=[HealthCheck(3)], deadline=None)
    @given(data_frames(columns=hypothesis["columns"]))
    def test_set_auxilliary_info_post_mapping_static_multicat(
        self, mock_split, mock_cuts, mock_MDL, df
    ):
        nsamples = len(df)
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome
        datasetloader_args = (df, y, hypothesis["ctn_cols"], hypothesis["cat_cols"])

        mock_split.return_value = (df.index, df.index)

        data = CommonDataModule(
            dataset_loader=SimpleDatasetLoader(*datasetloader_args),
            **self.standard,
        )
        data.columns = {"original": df.columns}

        data.setup("fit")
        self.assertIsNone(data.discretizations)
        self.assertIsNone(data.inverse_target_encode_map)
        # nothing should have a "mapped"
        self.assertTrue("mapped" not in data.groupby)
        self.assertTrue("mapped" not in data.nfeatures)
        self.assertTrue("mapped" not in data.col_idxs_by_type)
        self.assertTrue("mapped" not in data.columns)

        with self.subTest("With Discretization"):
            cuts = [[(0, 1), (1, 2)], [(0, 0.5), (0.5, 1), (1, 1.5)]]
            category_names = [["0 - 1", "1 - 2"], ["0 - 0.5", "0.5 - 1", "1 - 1.5"]]
            disc_data = create_fake_disc_data(
                rng, nsamples, cuts, category_names, hypothesis["ctn_cols"]
            )

            mock_disc_data(mock_MDL, disc_data, y)
            mock_cuts.return_value = cuts

            data = CommonDataModule(
                dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                **self.standard,
                feature_map="discretize_continuous",
            )
            data.columns = {"original": df.columns}
            data.setup("fit")

            self.assertIsNotNone(data.discretizations)
            # leave it up to test_transforms to see if discretized_ctn_cols is computed correctly
            self.assertEqual(
                list(data.groupby["mapped"].keys()),
                ["binary_vars", "discretized_ctn_cols"],
            )
            # remember nothing was onehot-d so we assume every cat var is binary
            self.assertDictEqual(
                data.groupby["mapped"]["binary_vars"],
                {0: "bin", 1: "mult1", 2: "mult2"},
            )
            self.assertEqual(data.nfeatures["mapped"], 8)
            self.assertEqual(
                data.col_idxs_by_type["mapped"],
                {
                    "categorical": list(range(8)),
                    "binary": [0, 1, 2],
                    "onehot": [[3, 4], [5, 6, 7]],
                    "continuous": [],
                },
            )
            np.testing.assert_array_equal(
                data.columns["mapped"],
                hypothesis["cat_cols"]
                + [
                    f"{col}_{bin_name}"
                    for i, col in enumerate(hypothesis["onehot"]["ctn_cols"])
                    for bin_name in category_names[i]
                ],
            )
            self.assertListEqual(
                data.columns["map-inverted"].tolist(),
                ["bin", "mult1", "mult2", "ctn1", "ctn2"],
            )

        with self.subTest("With Target Encoding"):
            data = CommonDataModule(
                dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                **self.standard,
                feature_map="target_encode_categorical",
            )
            data.columns = {"original": df.columns}
            data.setup("fit")

            self.assertEqual(
                list(data.inverse_target_encode_map.keys()),
                ["inverse_transform", "mapping"],
            )
            self.assertIsNotNone(data.inverse_target_encode_map["inverse_transform"])
            # nothing should be reordered, and all the binary and multicat vars should be encoded
            self.assertEqual(
                list(data.inverse_target_encode_map["mapping"].keys()),
                hypothesis["cat_cols_idx"],
            )

            # no combining so no mapped groupby
            self.assertTrue("mapped" not in data.groupby)
            if "categorical_onehots" not in data.groupby["original"]:
                self.assertTrue("binary_vars" not in data.groupby["original"])

            self.assertTrue("mapped" not in data.nfeatures)
            self.assertTrue("mapped" not in data.col_idxs_by_type)
            self.assertEqual(
                data.columns["original"].tolist(), data.columns["mapped"].tolist()
            )

    def test_separate_ground_truth(self):
        return

    def test_batch_collate(self):
        # TODO: fill this in!
        return

    # we just want ot make sure the helper function works and the data goes where it's supposed to
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_ampute(self, mock_split):
        rng = default_rng(seed)
        df = pd.DataFrame(
            rng.random((5, 5)), columns=["bin", "mult1", "ctn1", "mult2", "ctn2"]
        )
        nsamples = len(df)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome
        datasetloader_args = (df, y, hypothesis["ctn_cols"], hypothesis["cat_cols"])

        mock_split.return_value = (df.index, df.index)

        with self.subTest("ampute"):
            data = CommonDataModule(
                dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                ampute=True,
                amputation_patterns=[
                    {"incomplete_vars": ["ctn1"], "mechanism": "MNAR(G)"}
                ],
                **self.standard,
            )
            data.columns = {"original": df.columns}
            data.setup("fit")
            # columns should net the same
            np.testing.assert_array_equal(
                data.splits["data"]["train"].columns, df.columns
            )
            # there should be missing values
            self.assertGreater(data.splits["data"]["train"].isna().sum().sum(), 0)

            with self.subTest("Onehot nans"):
                onehot_df = onehot_multicategorical_column(
                    hypothesis["onehot_prefixes"]
                )(df)
                dataset_loader = SimpleDatasetLoader(
                    onehot_df,
                    y,
                    hypothesis["onehot"]["ctn_cols"],
                    hypothesis["onehot"]["cat_cols"],
                    hypothesis["onehot_prefixes"],
                )
                data = CommonDataModule(
                    dataset_loader=dataset_loader,
                    ampute=True,
                    amputation_patterns=[
                        {
                            "incomplete_vars": [onehot_df.columns[3]],
                            "mechanism": "MNAR(G)",
                        }
                    ],
                    **self.standard,
                )
                data.columns = {"original": onehot_df.columns}
                data.setup("fit")
                # there should be missing values
                self.assertGreater(data.splits["data"]["train"].isna().sum().sum(), 0)
                # the onehot groups should all be nan if one is nan
                for onehot_group in data.col_idxs_by_type["original"]["onehot"]:
                    onehot_group_isna = (
                        data.splits["data"]["train"].iloc[:, onehot_group].isna()
                    )
                    self.assertTrue(
                        (
                            onehot_group_isna.all(axis=1)
                            | (~onehot_group_isna.any(axis=1))
                        ).all()
                    )

        with self.subTest("_add_latent_features"):
            data = CommonDataModule(
                dataset_loader=SimpleDatasetLoader(*datasetloader_args),
                **self.standard,
            )
            data.columns = {"original": df.columns}
            with self.subTest("MNAR (not recoverable"):
                # should do nothing
                data.amputation_patterns = [{"mechanism": "MNAR"}]
                res = data._add_latent_features(df)
                # shouldn't add any features
                np.testing.assert_array_equal(res.columns, df.columns)
                # shouldn't change patterns
                self.assertEqual(data.amputation_patterns, [{"mechanism": "MNAR"}])

            with self.subTest("MNAR (recoverable)"):
                data.amputation_patterns = [{"mechanism": "MNAR(G)"}]
                res = data._add_latent_features(df)
                np.testing.assert_array_equal(
                    res.columns, df.columns.append(pd.Index(["latent_p0_g0"]))
                )
                self.assertEqual(
                    data.amputation_patterns,
                    [{"mechanism": "MNAR", "weights": {"latent_p0_g0": 1}}],
                )
                with self.subTest("Multiple Latent Features"):
                    data.amputation_patterns = [{"mechanism": "MNAR(G, G)"}]
                    res = data._add_latent_features(df)
                    np.testing.assert_array_equal(
                        res.columns,
                        df.columns.append(pd.Index(["latent_p0_g0", "latent_p0_g1"])),
                    )
                    self.assertEqual(
                        data.amputation_patterns,
                        [
                            {
                                "mechanism": "MNAR",
                                "weights": {"latent_p0_g0": 1, "latent_p0_g1": 1},
                            }
                        ],
                    )

                with self.subTest("Multiple Patterns"):
                    data.amputation_patterns = [
                        {"mechanism": "MNAR(Y)"},
                        {"mechanism": "MNAR(G)"},
                    ]
                    res = data._add_latent_features(df)
                    np.testing.assert_array_equal(
                        res.columns,
                        df.columns.append(
                            pd.Index(["latent_p0_y0", "latent_p1_g0"]),
                        ),
                    )
                    self.assertEqual(
                        data.amputation_patterns,
                        [
                            {
                                "mechanism": "MNAR",
                                "weights": {"latent_p0_y0": 1},
                            },
                            {
                                "mechanism": "MNAR",
                                "weights": {"latent_p1_g0": 1},
                            },
                        ],
                    )

                with self.subTest("Existing Weights"):
                    with self.subTest("Dict"):
                        data.amputation_patterns = [
                            {"mechanism": "MNAR(G)", "weights": {"bin": 1}}
                        ]
                        res = data._add_latent_features(df)
                        # add to the weights
                        self.assertEqual(
                            data.amputation_patterns,
                            [
                                {
                                    "mechanism": "MNAR",
                                    "weights": {"bin": 1, "latent_p0_g0": 1},
                                }
                            ],
                        )
                    with self.subTest("List"):
                        data.amputation_patterns = [
                            {"mechanism": "MNAR(G)", "weights": [1, 0, 0, 0, 0]}
                        ]
                        res = data._add_latent_features(df)
                        # add to the weights
                        self.assertEqual(
                            data.amputation_patterns,
                            [{"mechanism": "MNAR", "weights": [1, 0, 0, 0, 0, 1]}],
                        )


if __name__ == "__main__":
    unittest.main()
