import re
from itertools import chain
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from numpy.random import default_rng
import torch
from category_encoders import TargetEncoder

from hypothesis import (
    assume,
    given,
    HealthCheck,
    settings,
    strategies as st,
)
from hypothesis.extra.pandas import data_frames

from autopopulus.data.utils import onehot_multicategorical_column
from autopopulus.data.transforms import (
    CombineOnehots,
    Discretizer,
    invert_target_encoding_tensor,
    UniformProbabilityAcrossNans,
    invert_discretize_tensor,
    simple_impute_tensor,
)
from autopopulus.test.common_mock_data import hypothesis, seed
from autopopulus.test.utils import (
    build_onehot_from_hypothesis,
    create_fake_disc_data,
    mock_disc_data,
)
from autopopulus.models.ae import AEDitto


def unpack_tuples(nested_tuples):
    """
    We receive a List[Tuple[int, List[int]]].
    The first int is the numerical id, and the second is the "time point".
    We want to flatten this into a List[Tuple[int, int]] with the same
    id for multiple time points.
    E.g. [(0,[0,1,2]), (1,[0,2])] => [(0,0), (0,1), (0,2), (1,0), (1,2)]
    """
    return [
        (pt_id, time_pt) for pt_id, time_pts in nested_tuples for time_pt in time_pts
    ]


class TestTransforms(unittest.TestCase):
    """Some transforms can be tested with hypothesis, the rest by hand."""

    def setUp(self) -> None:
        super().setUp()

    ######################
    #  HYPOTHESIS TESTS  #
    ######################
    # For debugging the test
    @settings(suppress_health_check=[HealthCheck(3)], deadline=None)
    @given(
        data_frames(
            columns=hypothesis["columns"],
            index=st.builds(
                pd.MultiIndex.from_tuples,
                st.lists(
                    st.tuples(
                        st.integers(0), st.lists(st.integers(0), min_size=1, max_size=5)
                    ),
                    min_size=2,
                ).map(unpack_tuples),
            ),
        ).map(onehot_multicategorical_column(hypothesis["onehot_prefixes"]))
    )
    def test_longitudinal_onehot_data(self, df):
        return

    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_onehot(self, mock_disc_cuts, mock_MDL, df):
        # Ensure all categories/cols present for testing
        assume(
            np.array_equal(
                df.nunique()[hypothesis["onehot_prefixes"]].values, np.array([4, 3])
            )
        )
        onehot_df = build_onehot_from_hypothesis(df, hypothesis["onehot_prefixes"])

        nsamples = len(df)
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome

        with self.subTest("Simple Imputation"):
            # weird numpy bug: https://github.com/numpy/numpy/issues/22347
            assume(
                # ignore large values (where not nan/inf)
                ((df.isna() | np.isinf(df)) | (df.abs() < 1e305)).all().all()
                and not df.isna().all(0).any()
                and not np.isinf(df).any().any()
            )
            self._test_simple_impute(
                onehot_df,
                AEDitto._idxs_to_tensor(hypothesis["onehot"]["ctn_cols_idx"]),
                AEDitto._idxs_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
                AEDitto._idxs_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
            )

        with self.subTest("Uniform Probability Across Nans"):
            onehot_groupby = dict(
                zip(
                    chain.from_iterable(hypothesis["onehot"]["onehot_cols_idx"]),
                    hypothesis["onehot"]["onehot_expanded_prefixes"],
                )
            )
            # no discretizer dict yet but we have onehots
            # nothing gets shifted since no discretize
            self._test_uniform_prob(
                onehot_df,
                {"categorical_onehots": onehot_groupby, "binary_vars": {0: "bin"}},
                {},
            )

        discretized_data = None
        with self.subTest("Discretizer"):
            # if i add continuous columns i need to adjust these
            nfeatures = 10 + 5 - 2
            cuts = [[(0, 1), (1, 2)], [(0, 0.5), (0.5, 1), (1, 1.5)]]
            category_names = [["0 - 1", "1 - 2"], ["0 - 0.5", "0.5 - 1", "1 - 1.5"]]

            # Create fake discretized data
            disc_data = create_fake_disc_data(
                rng, nsamples, cuts, category_names, hypothesis["onehot"]["ctn_cols"]
            )
            mock_disc_data(mock_MDL, disc_data, y)
            mock_disc_cuts.return_value = cuts

            # True values to compare to
            # added to the end, there are N features that are not continuous
            existing_col_offset = len(hypothesis["onehot"]["cat_cols"])
            discretizer_dict = {
                colname: {  # 2 bins
                    "bins": cuts[i],
                    "labels": category_names[i],
                    "indices": [
                        # offset the existing and cumulative disc cols
                        j + existing_col_offset + sum([len(cuts[k]) for k in range(i)])
                        for j in range(len(cuts[i]))
                    ],
                }
                for i, colname in enumerate(hypothesis["onehot"]["ctn_cols"])
            }
            disc_groupby = {
                idx: ctn_col
                for ctn_col, col_info in discretizer_dict.items()
                for idx in col_info["indices"]
            }
            # 10 features + (2-1) + (3-1)
            true_df = pd.concat(
                [onehot_df[hypothesis["onehot"]["cat_cols"]], disc_data], axis=1
            )

            # enforce my data to fall into the bins I generated in
            enc = CombineOnehots(  # remove the offset since only looking at disc data
                {
                    idx - len(hypothesis["onehot"]["cat_cols"]): ctn_col
                    for idx, ctn_col in disc_groupby.items()
                },
                disc_data.columns,
            ).fit(disc_data, None)
            # grabs the range strings and replaces with the mean
            # find each number, split into list, convert ot float, average
            bin_means = enc.transform(disc_data).applymap(
                lambda s: sum([float(v) for v in re.findall(f"((?:\d*\.)?\d+)", s)]) / 2
            )
            # plop into df
            mocked_df = onehot_df.copy()
            mocked_df[hypothesis["onehot"]["ctn_cols"]] = bin_means

            discretized_data = self._test_discretize(
                mocked_df,
                y,
                hypothesis["onehot"]["ctn_cols_idx"],
                disc_groupby,
                discretizer_dict,
                nfeatures,
                true_df,
            )

            with self.subTest("Discretizer + Uniform Probability Across Nans"):
                # shifted since we're discretizing
                self._test_uniform_prob(
                    discretized_data,
                    {
                        "categorical_onehots": {
                            1 + i: prefix
                            for i, prefix in enumerate(
                                hypothesis["onehot"]["onehot_expanded_prefixes"]
                            )
                        },
                        "binary_vars": {0: "bin"},
                    },
                    discretizer_dict,
                )
                # shifted categorical by 1 since bin is 1st

            with self.subTest("Invert Discretize"):  # fn(fn-1) = Identity fn
                self._test_invert_discretize(
                    torch.tensor(discretized_data.values),
                    disc_groupby,
                    discretizer_dict,
                    onehot_df.columns,
                    torch.tensor(mocked_df.values),
                )

        with self.subTest("Invert Target Encoding"):
            with self.subTest("CombineOnehots"):
                combined_groupby = {3: "mult1", 4: "mult2"}
                combined_df = self._test_combine_onehots(
                    onehot_df, y, df, onehot_groupby, combined_groupby
                )

            categories = [[0, 1], [0, 1, 2, 3], [0, 1, 2]]
            # mock an encoding for each categorical column
            ordinal_to_mean_map = {
                combined_df.columns.get_loc(col): {
                    category: rng.random() for category in categories[i]
                }
                for i, col in enumerate(
                    hypothesis["onehot"]["bin_cols"] + hypothesis["onehot_prefixes"]
                )
            }
            # enforce float for nans bc for some reason df's dype was object
            self._test_invert_target_encode(
                combined_df.astype(float),
                y,
                onehot_df.astype(float),
                ["bin", "mult1", "mult2"],
                onehot_df.columns,
                combined_groupby,
            )

    # @patch("autopopulus.data.MDLDiscretization.EntropyMDL._entropy_discretize_sorted")
    # for using orange for discretization
    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_multicat(self, mock_disc_cuts, mock_MDL, df):
        nsamples = len(df)
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome

        with self.subTest("Simple Imputation"):
            assume(
                # ignore large values (where not nan/inf)
                ((df.isna() | np.isinf(df)) | (df.abs() < 1e305)).all().all()
                and not df.isna().all(0).any()
                and not np.isinf(df).any().any()
            )
            self._test_simple_impute(
                df,
                torch.tensor(hypothesis["ctn_cols_idx"]),
                torch.tensor(
                    hypothesis["cat_cols_idx"]
                ),  # multicat will work find as considered a binary col here mode will work
                torch.tensor([]),
            )

        with self.subTest("onehots"):  # do in multicat since we have the "truth"
            # Ensure all categories/cols present for testing
            assume(
                np.array_equal(
                    df.nunique()[hypothesis["onehot_prefixes"]].values, np.array([4, 3])
                )
            )

            with self.subTest("onehot_multicategorical_column"):
                self._test_onehot_cols(df, hypothesis["onehot_prefixes"])

        with self.subTest("Uniform Probability Across Nans"):
            # no onehot prefixes since just multicat data, also no discretizer dict
            self._test_uniform_prob(df, {}, {})

        discretized_data = None
        with self.subTest("Discretizer"):
            assume(not df.empty)  # onehot'ing the bins will not work on empty

            # if i add continuous columns i need to adjust these
            nfeatures = 8
            cuts = [[(0, 1), (1, 2)], [(0, 0.5), (0.5, 1), (1, 1.5)]]
            category_names = [["0 - 1", "1 - 2"], ["0 - 0.5", "0.5 - 1", "1 - 1.5"]]

            # Create fake discretized data
            disc_data = create_fake_disc_data(
                rng, nsamples, cuts, category_names, hypothesis["ctn_cols"]
            )
            mock_disc_data(mock_MDL, disc_data, y)
            mock_disc_cuts.return_value = cuts

            # True values to compare to
            # added to the end, there are N features that are not continuous
            existing_col_offset = len(hypothesis["cat_cols"])
            discretizer_dict = {
                colname: {  # 2 bins
                    "bins": cuts[i],
                    "labels": category_names[i],
                    "indices": [
                        # offset the existing and cumulative disc cols
                        j + existing_col_offset + sum([len(cuts[k]) for k in range(i)])
                        for j in range(len(cuts[i]))
                    ],
                }
                for i, colname in enumerate(hypothesis["ctn_cols"])
            }
            disc_groupby = {
                idx: ctn_col
                for ctn_col, col_info in discretizer_dict.items()
                for idx in col_info["indices"]
            }
            # 5 features + (2-1) + (3-1)
            true_df = pd.concat([df[hypothesis["cat_cols"]], disc_data], axis=1)

            # enforce my data to fall into the bins I generated in
            enc = CombineOnehots(  # remove the offset since only looking at disc data
                {
                    idx - len(hypothesis["cat_cols"]): ctn_col
                    for idx, ctn_col in disc_groupby.items()
                },
                disc_data.columns,
            ).fit(disc_data, None)
            # grabs the range strings and replaces with the mean
            # find each number, split into list, convert ot float, average
            bin_means = enc.transform(disc_data).applymap(
                lambda s: sum([float(v) for v in re.findall(f"((?:\d*\.)?\d+)", s)]) / 2
            )
            # plop into df
            mocked_df = df.copy()
            mocked_df[hypothesis["ctn_cols"]] = bin_means

            discretized_data = self._test_discretize(
                mocked_df,
                y,
                hypothesis["ctn_cols_idx"],
                disc_groupby,
                discretizer_dict,
                nfeatures,
                true_df,
            )

            with self.subTest("Discretizer + Uniform Probability Across Nans"):
                self._test_uniform_prob(
                    discretized_data,
                    {"binary_vars": {0: "bin", 1: "mult1", 2: "mult2"}},
                    discretizer_dict,
                )

            with self.subTest("Invert Discretize"):  # fn(fn-1) = Identity fn
                self._test_invert_discretize(
                    torch.tensor(discretized_data.values),
                    disc_groupby,
                    discretizer_dict,
                    df.columns,
                    torch.tensor(mocked_df.values),
                )

        with self.subTest("Invert Target Encoding"):
            # do not test combineonehot since no onehot is set
            # ordinal encoding won't work if empty
            assume(not df.empty)
            categories = [[0, 1], [0, 1, 2, 3], [0, 1, 2]]
            # mock an encoding for each categorical column
            ordinal_to_mean_map = {
                df.columns.get_loc(col): {
                    category: rng.random() for category in categories[i]
                }
                for i, col in enumerate(hypothesis["cat_cols"])
            }
            self._test_invert_target_encode(
                df,
                y,
                df,
                hypothesis["cat_cols"],
                df.columns,
            )

    ##################################
    #    Tests for each transform    #
    ##################################
    #     to use with hypothesis     #
    def _test_simple_impute(
        self,
        df: pd.DataFrame,
        ctn_cols_idxs: torch.Tensor,
        bin_cols_idxs: torch.Tensor,
        onehot_group_idxs: torch.Tensor,
    ):
        X = torch.tensor(df.values)
        non_missing_mask = torch.tensor(~(df.isna().astype(bool)).values)
        # impute as tensor and convert back to df for testing
        imputed, means, modes = simple_impute_tensor(
            X,
            non_missing_mask,
            ctn_cols_idxs,
            bin_cols_idxs,
            onehot_group_idxs,
            return_learned_stats=True,
        )
        # should be no nans
        self.assertFalse(torch.isnan(imputed).any())

        # bin cols should still be bin but can be homogeneous
        for bin_col_idx in bin_cols_idxs:
            X_col = X[:, bin_col_idx]
            # deal with Nans in X by ignoring them because we're only checking still bin
            np.testing.assert_array_equal(
                imputed[:, bin_col_idx].unique(), X_col[~X_col.isnan()].unique()
            )
        # onehots should still be onehot
        for onehot_group_idx in onehot_group_idxs:
            # ignore pads of -1
            onehot_group_idx = onehot_group_idx[onehot_group_idx != -1]
            self.assertTrue(
                # all are either 1 or 0
                (
                    (imputed[:, onehot_group_idx] == 1)
                    | (imputed[:, onehot_group_idx] == 0)
                )
                .all()
                .all()
                # and the rows sum to 1
                and (imputed[:, onehot_group_idx].sum(axis=1) == 1).all()
            )

        # by imputing with the means the mean should not change (same with mode)
        imputed_again, new_means, new_modes = simple_impute_tensor(
            imputed,
            non_missing_mask,
            ctn_cols_idxs,
            bin_cols_idxs,
            onehot_group_idxs,
            return_learned_stats=True,
        )
        np.testing.assert_allclose(means, new_means, atol=1e-4)
        np.testing.assert_allclose(modes, new_modes)
        # nothing should get changed
        torch.testing.assert_close(imputed, imputed_again)

    def _test_onehot_cols(
        self,
        df: pd.DataFrame,
        onehot_prefixes: List[str],
        onehot_true_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        onehot = onehot_multicategorical_column(onehot_prefixes)(df)
        # ensure nans are kept properly, since exploding columns there may be more nans
        self.assertTrue(all(onehot.isna().sum(axis=1) >= df.isna().sum(axis=1)))

        if onehot_true_df is not None:
            # need to reorder columns
            pd.testing.assert_frame_equal(
                onehot[onehot_true_df.columns], onehot_true_df, check_dtype=False
            )
        return onehot

    def _test_combine_onehots(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        true_df: pd.DataFrame,
        onehot_groupby: Dict[int, str],
        combined_groupby: Dict[int, str],
    ) -> pd.DataFrame:
        with self.subTest("No Onehots"):
            # Should do nothing
            combine = CombineOnehots({}, df.columns)
            combine.fit(df, y)
            self.assertEqual(combine.combined_onehot_groupby, {})
            self.assertEqual(combine.nfeatures, len(df.columns))
            transformed = combine.transform(df)[df.columns]
            pd.testing.assert_frame_equal(
                transformed.astype(float), df, check_dtype=False
            )

        combine = CombineOnehots(onehot_groupby, df.columns)

        # Test groupby and nfeatures is right after fit
        combine.fit(df, y)
        self.assertEqual(combine.combined_onehot_groupby, combined_groupby)
        self.assertEqual(combine.nfeatures, len(true_df.columns))

        # reorder columns to match
        transformed = combine.transform(df)[true_df.columns]
        # the combined cols will be dtype obj since the values are pulled from the column name and we don't know if the intention is to be str/float
        # but with the columns i specified they're numbers
        pd.testing.assert_frame_equal(
            transformed.astype(float), true_df, check_dtype=False
        )
        return transformed

    def _test_uniform_prob(
        self,
        df: pd.DataFrame,
        groupby_mapped: Dict[str, Dict[int, str]],
        discretizer_dict: Dict[str, Dict[str, List[int]]],
        true_df: Optional[pd.DataFrame] = None,
    ):
        # groupby will not have discretized_ctn_cols yet when instantiating
        uniform_transformer = UniformProbabilityAcrossNans(
            groupby_mapped, df.columns
        ).fit(df, None)
        transformed = uniform_transformer.transform((df, discretizer_dict))

        # there should be no nans
        if discretizer_dict:
            # check all: discretized_ctn_cols, onehot, binary cols
            self.assertFalse(transformed.isna().any().any())
        else:  # we don't care about continuous columns
            for groupby in uniform_transformer.groupby_categorical_only.values():
                np.testing.assert_array_equal(
                    transformed.isna().any().groupby(groupby).any(), False
                )

        # all discretized groups should sum to 1 (whether or not missing)
        for col_info in discretizer_dict.values():
            np.testing.assert_array_equal(
                transformed.iloc[:, col_info["indices"]].sum(1), 1
            )
        # all onehot groups should sum to 1 (whether or not missing)
        if "categorical_onehots" in groupby_mapped:
            np.testing.assert_array_equal(
                transformed.groupby(
                    uniform_transformer.groupby_categorical_only["categorical_onehots"],
                    axis=1,
                ).sum(),
                1,
            )
        # binary where missing should all be 0.5
        if "binary_vars" in groupby_mapped:
            for bin_idx in uniform_transformer.groupby_categorical_only[
                "binary_vars"
            ].values():
                np.testing.assert_array_equal(
                    transformed[bin_idx][df[bin_idx].isna()], 0.5
                )

        if true_df is not None:
            pd.testing.assert_frame_equal(transformed, true_df)

    def _test_discretize(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        ctn_cols_idxs: List[int],
        true_disc_groupby: Dict[int, str],
        true_map_dict: Dict[str, Dict[str, List[int]]],
        true_nfeatures: int,
        true_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        discretize_transformer = Discretizer(
            df.columns, ctn_cols_idxs, return_info_dict=False
        ).fit(df, y)

        # allows comparison of dictionaries with nparrays in them
        np.testing.assert_equal(discretize_transformer.map_dict, true_map_dict)
        # Test groupby and nfeatures is right after fit
        self.assertEqual(discretize_transformer.discretized_groupby, true_disc_groupby)
        self.assertEqual(discretize_transformer.nfeatures, true_nfeatures)

        transformed = discretize_transformer.transform(df)

        if true_df is not None:
            pd.testing.assert_frame_equal(transformed, true_df, check_dtype=False)

        return transformed

    def _test_invert_discretize(
        self,
        df: torch.Tensor,
        disc_groupby: Dict[int, str],
        map_dict: Dict[str, Dict[str, List[int]]],
        true_columns: List[str],
        true_tensor: torch.Tensor,
    ):
        undiscretized_tensor = invert_discretize_tensor(
            df,
            disc_groupby=disc_groupby,
            discretizations=map_dict,
            orig_columns=true_columns,
        )
        torch.testing.assert_close(
            undiscretized_tensor, true_tensor, check_dtype=False, equal_nan=True
        )

    def _test_invert_target_encode(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        true_df: pd.DataFrame,
        cat_cols: List[str],
        orig_cols: List[str],
        combined_onehots_groupby: Optional[Dict[int, int]] = None,
    ):
        enc = TargetEncoder(
            cols=cat_cols, handle_missing="return_nan", handle_unknown="return_nan"
        )
        enc.fit(df, y)
        # make the encoding random
        rng = default_rng()
        enc.mapping = {
            colname: series_mapping.map(lambda enc_v: rng.random())
            for colname, series_mapping in enc.mapping.items()
        }
        unencoded_tensor = invert_target_encoding_tensor(
            torch.tensor(enc.transform(df, y).values),
            {
                "inverse_transform": enc.ordinal_encoder.inverse_transform,
                "mapping": {
                    df.columns.get_loc(col): col_mapping.to_dict()
                    for col, col_mapping in enc.mapping.items()
                },
            },
            df.columns,
            orig_cols,
            combined_onehots_groupby,
        )
        # rounding error with float point in torch?
        np.testing.assert_allclose(
            unencoded_tensor, torch.tensor(true_df.values).float(), atol=1
        )


if __name__ == "__main__":
    unittest.main()
