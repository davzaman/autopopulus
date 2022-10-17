from math import pi
import unittest
import numpy as np
import pandas as pd
import torch
from hypothesis import assume, given, HealthCheck, settings, strategies as st
from hypothesis.extra.pandas import data_frames

from autopopulus.utils.impute_metrics import (
    CWMAAPE,
    CWRMSE,
    EPSILON,
    AccuracyMetric,
    MAAPEMetric,
    RMSEMetric,
    categorical_accuracy,
)
from autopopulus.test.common_mock_data import (
    seed,
    hypothesis,
)
from autopopulus.test.utils import build_onehot_from_hypothesis
from models.ae import AEDitto


# when subtracting, tensors add in a little margin of error that accumulates, so we want to get close within WITHIN decimal places.
WITHIN = 6


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_onehot_categorical_accuracy(self, df):
        # don't need to test continuous
        # Ensure all categories/cols present for testing
        assume(
            np.array_equal(
                df.nunique()[hypothesis["onehot_prefixes"]].values, np.array([4, 3])
            )
        )
        df = df.replace({np.nan: 400, np.inf: 500})
        df = build_onehot_from_hypothesis(df, hypothesis["onehot_prefixes"])
        tensor_df = torch.tensor(df.values)
        onehot_group = hypothesis["onehot"]["onehot_cols_idx"][0]
        accuracy_fn = categorical_accuracy(
            AEDitto._idxs_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
            AEDitto._idxs_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
        )
        accuracy_elwise = AccuracyMetric(
            AEDitto._idxs_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
            AEDitto._idxs_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
        )

        with self.subTest("No Mask"):
            with self.subTest("All Equal"):
                self.assertAlmostEqual(
                    1,
                    accuracy_fn(tensor_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    1,
                    accuracy_elwise(tensor_df, tensor_df).item(),
                    places=WITHIN,
                )

            N = len(df)  # num samples
            F = len(hypothesis["cat_cols_idx"])  # num features
            C = len(onehot_group)  # num categories
            with self.subTest("Not Equal"):
                # Create an error in one of the onehots
                error_df = df.copy()
                # change one of the onehots
                old_cat = np.argmax(df.iloc[0, onehot_group])
                new_cat = C - old_cat
                error_df.iloc[0, onehot_group] = np.eye(C)[
                    new_cat - 1 if (new_cat == old_cat or old_cat == 0) else new_cat
                ]

                error_df = torch.tensor(error_df.values)
                self.assertAlmostEqual(  # 1 error for 1 feature out of F
                    (((N - 1) / N) + F - 1) / F,
                    accuracy_fn(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(  # one error out of all the cells
                    (N * F - 1) / (N * F),
                    accuracy_elwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )

        with self.subTest("Mask"):
            missing_mask = torch.zeros_like(tensor_df).to(bool)
            missing_mask[0, onehot_group] = True
            with self.subTest("Not Equal inside Mask"):
                self.assertAlmostEqual(
                    1,
                    accuracy_fn(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    1,
                    accuracy_elwise(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )

            with self.subTest("Not Equal outside mask"):
                error_df = df.copy()
                # change one of the onehots
                old_cat = np.argmax(df.iloc[0, onehot_group])
                new_cat = C - old_cat
                error_df.iloc[0, onehot_group] = np.eye(C)[
                    new_cat - 1 if (new_cat == old_cat or old_cat == 0) else new_cat
                ]
                # change one of the onehots
                old_cat = np.argmax(df.iloc[1, onehot_group])
                new_cat = C - old_cat
                error_df.iloc[1, onehot_group] = np.eye(C)[
                    new_cat - 1 if (new_cat == old_cat or old_cat == 0) else new_cat
                ]
                error_df = torch.tensor(error_df.values)
                # 2 error, but ignoring 1 position
                self.assertAlmostEqual(
                    (((N - 1 - 1) / (N - 1)) + F - 1) / F,
                    accuracy_fn(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    ((N * F - 1) - 1)
                    / (
                        N * F - 1
                    ),  # error in 1 non-masked cells when theres 1 cell less
                    accuracy_elwise(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_multicat_continuous(self, df):
        assume(
            (not df.isna().any().any())
            and (len(df) > 1)
            and (np.isinf(df).values.sum() == 0)
            # large values in float aren't properly represented and I will get the wrong results
            and ((df.values > 1e16).sum() == 0)
        )
        tensor_df = torch.tensor(df.values)

        maape_elwise = MAAPEMetric(scale_to_01=True)
        rmse_elwise = RMSEMetric()

        with self.subTest("No Mask"):
            self.assertAlmostEqual(
                0, CWRMSE(tensor_df, tensor_df).item(), places=WITHIN
            )
            self.assertAlmostEqual(
                0, CWMAAPE(tensor_df, tensor_df).item(), places=WITHIN
            )
            self.assertAlmostEqual(
                0, rmse_elwise(tensor_df, tensor_df).item(), places=WITHIN
            )
            self.assertAlmostEqual(
                0, maape_elwise(tensor_df, tensor_df).item(), places=WITHIN
            )

            ctn_col_idx = hypothesis["ctn_cols_idx"][0]

            # Now if they dont exactly equal each other
            with self.subTest("Not Equal"):
                # Create an error in one of the places
                error_df = df.copy()
                diff = 6
                # subtraction happens in np cuz with torch i was getting the wrong values
                error_df.iloc[0, ctn_col_idx] = df.iloc[0, ctn_col_idx] - diff
                error_df = torch.tensor(error_df.values)

                self.assertAlmostEqual(
                    (diff**2 / len(df) / df.shape[1]) ** 0.5,
                    CWRMSE(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    np.arctan(abs(diff / tensor_df[0, ctn_col_idx] + EPSILON)).item()
                    / len(df)
                    / df.shape[1],
                    CWMAAPE(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    (diff**2 / (len(df) * df.shape[1])) ** 0.5,
                    rmse_elwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    (2 / pi)
                    * np.arctan(abs(diff / tensor_df[0, ctn_col_idx] + EPSILON)).item()
                    / (len(df) * df.shape[1]),
                    maape_elwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )

        with self.subTest("Mask"):
            missing_mask = torch.zeros_like(tensor_df).to(bool)
            missing_mask[0, ctn_col_idx] = True
            self.assertAlmostEqual(
                0, CWRMSE(tensor_df, tensor_df, missing_mask).item(), places=WITHIN
            )
            self.assertAlmostEqual(
                0, CWMAAPE(tensor_df, tensor_df, missing_mask).item(), places=WITHIN
            )
            self.assertAlmostEqual(
                0, rmse_elwise(tensor_df, tensor_df, missing_mask).item(), places=WITHIN
            )
            self.assertAlmostEqual(
                0,
                maape_elwise(tensor_df, tensor_df, missing_mask).item(),
                places=WITHIN,
            )

            # Now if they dont exactly equal each other inside the mask
            with self.subTest("Not Equal inside mask"):
                error_df = df.copy()
                error_df.iloc[0, ctn_col_idx] = df.iloc[0, ctn_col_idx] - diff
                error_df = torch.tensor(error_df.values)
                self.assertAlmostEqual(
                    0,
                    CWRMSE(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    0,
                    CWMAAPE(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    0,
                    rmse_elwise(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    0,
                    maape_elwise(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )

                # make sure that the value is still the same even if the values outside the mask don't match, since we don't care about them and don't want to count them
                with self.subTest("Not Equal outside mask"):
                    new_diff = 4
                    error_df = df.copy()
                    error_df.iloc[0, ctn_col_idx] = error_df.iloc[0, ctn_col_idx] - diff
                    error_df.iloc[1, ctn_col_idx] = (
                        error_df.iloc[1, ctn_col_idx] - new_diff
                    )
                    error_df = torch.tensor(error_df.values)
                    self.assertAlmostEqual(
                        # -1 from # samples bc we are ignoring 1 element in the same feature
                        ((new_diff**2 / (len(df) - 1) / df.shape[1]) ** 0.5),
                        CWRMSE(error_df, tensor_df, missing_mask).item(),
                        places=WITHIN,
                    )
                    self.assertAlmostEqual(
                        np.arctan(
                            abs(new_diff / (tensor_df[1, ctn_col_idx] + EPSILON))
                        ).item()
                        / (len(df) - 1)
                        / df.shape[1],
                        CWMAAPE(error_df, tensor_df, missing_mask).item(),
                        places=WITHIN,
                    )
                    self.assertAlmostEqual(
                        (new_diff**2 / ((len(df) * df.shape[1]) - 1)) ** 0.5,
                        rmse_elwise(error_df, tensor_df, missing_mask).item(),
                        places=WITHIN,
                    )
                    self.assertAlmostEqual(
                        (2 / pi)
                        * np.arctan(
                            abs(new_diff / (tensor_df[1, ctn_col_idx] + EPSILON))
                        ).item()
                        / ((len(df) * df.shape[1]) - 1),
                        maape_elwise(error_df, tensor_df, missing_mask).item(),
                        places=WITHIN,
                    )

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_multicat_categorical(self, df):
        assume(
            (not df.isna().any().any())
            and (len(df) > 1)
            and (np.isinf(df).values.sum() == 0)
        )
        tensor_df = torch.tensor(df.values)
        bin_col_idx = hypothesis["cat_cols_idx"][0]

        accuracy_fn = categorical_accuracy(hypothesis["cat_cols_idx"], [])
        accuracy_elwise = AccuracyMetric(hypothesis["cat_cols_idx"], [])

        with self.subTest("No Mask"):
            with self.subTest("All Equal"):
                self.assertAlmostEqual(
                    1, accuracy_fn(tensor_df, tensor_df).item(), places=WITHIN
                )
                self.assertAlmostEqual(
                    1, accuracy_elwise(tensor_df, tensor_df).item(), places=WITHIN
                )

            N = len(df)
            F = len(hypothesis["cat_cols_idx"])
            with self.subTest("Not Equal"):
                # Create an error in one of the places
                error_df = df.copy()
                # flip a binary col
                error_df.iloc[0, bin_col_idx] = 1 - df.iloc[0, bin_col_idx]
                error_df = torch.tensor(error_df.values)
                self.assertAlmostEqual(  # 1 error for 1 feature out of F
                    (((N - 1) / N) + F - 1) / F,
                    accuracy_fn(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(  # 1 error out of all the cells
                    (N * F - 1) / (N * F),
                    accuracy_fn(error_df, tensor_df).item(),
                    places=WITHIN,
                )

        with self.subTest("Mask"):
            missing_mask = torch.zeros_like(tensor_df).to(bool)
            missing_mask[0, bin_col_idx] = True
            with self.subTest("Not Equal inside Mask"):
                self.assertAlmostEqual(
                    1,
                    accuracy_fn(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    1,
                    accuracy_elwise(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )

            with self.subTest("Not Equal outside mask"):
                # Create an error in one of the places
                error_df = df.copy()
                # flip a binary col
                error_df.iloc[0, bin_col_idx] = 1 - df.iloc[0, bin_col_idx]
                error_df.iloc[1, bin_col_idx] = 1 - df.iloc[1, bin_col_idx]
                error_df = torch.tensor(error_df.values)
                self.assertAlmostEqual(
                    # 2 error, but ignoring 1 position
                    (((N - 1 - 1) / (N - 1)) + F - 1) / F,
                    categorical_accuracy(hypothesis["cat_cols_idx"], [])(
                        error_df, tensor_df, missing_mask
                    ).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    ((N * F - 1) - 1)
                    / (
                        N * F - 1
                    ),  # error in 1 non-masked cells when theres 1 cell less
                    accuracy_elwise(error_df, tensor_df, missing_mask).item(),
                    places=WITHIN,
                )


if __name__ == "__main__":
    unittest.main()
