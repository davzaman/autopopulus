from math import pi, sqrt
from typing import List
import unittest
import numpy as np
import pandas as pd
import torch
from hypothesis import assume, given, HealthCheck, settings, strategies as st
from hypothesis.extra.pandas import data_frames
from torchmetrics import Metric

from autopopulus.utils.impute_metrics import (
    EPSILON,
    CategoricalErrorMetric,
    MAAPEMetric,
    RMSEMetric,
    universal_metric,
)
from autopopulus.test.common_mock_data import hypothesis
from autopopulus.test.utils import build_onehot_from_hypothesis
from autopopulus.data.transforms import list_to_tensor


# when subtracting, tensors add in a little margin of error that accumulates, so we want to get close within WITHIN decimal places.
WITHIN = 6


"""
Tests are Organized as:
class FeatureSpace_FeatureType:
    - no mask
        - all equal
            compare metric(df, df)
        - 1 error in 1 column
            compare metric(error_df, df)
    - mask
        === All share missing indicator, but not error_df ===
        - all equal
            compare metric(df, df, missing_indicator)
        - observed (ignored) value not equal
            compare metric(error_df, df, missing_indicator)
        - observed (ignored) and missing (calculated) value not equal
            compare metric(error_df, df, missing_indicator)
    - mask nomissing (the error_df can mirror the obs+missing neq example)
        === All share error_df, but not missing indictor ===
        - no missing values at all
            compare metric(error_df, df, missing_indicator)
        - only 1 column missing (calculated), the rest observed (ignored)
            compare metric(error_df, df, missing_indicator)
"""


def batched_input_equals(self, metric, *inputs):
    """
    Assume the metric on the whole dataset is correct,
    now make sure it works when data passed in batches
    (aka metric(data) == metric(data_part1) synced with metric(data_part2))
    """
    # want to split it into parts, and also we don't care about universal_metric versions
    if len(inputs[0]) <= 1 or not isinstance(metric, Metric):
        return

    with self.subTest("Test Batch Inputs"):
        nparts = 2
        batch_metric = metric.clone()
        # this will be a list of #inputs x #parts, convert to #parts x #inputs
        # e.g. [["a1","a2"], ["b1", "b2"]] -> [["a1", "b1"], ["a2", "b2"]]
        split_inputs = list(
            zip(*[torch.split(data, len(data) // nparts) for data in inputs])
        )
        for part in split_inputs:
            batch_metric.update(*part)
        self.assertAlmostEqual(batch_metric.compute().item(), metric(*inputs).item())
        metric.reset()


class TestStaticMulticatContinuousMetrics(unittest.TestCase):
    def batched_input_equals(self, metric, *inputs):
        batched_input_equals(self, metric, *inputs)

    def setup_metrics(self, df: pd.DataFrame) -> None:
        ctn_cols_idx = list_to_tensor(range(df.shape[1]))
        self.maape_elwise = MAAPEMetric(ctn_cols_idx=ctn_cols_idx)
        self.rmse_elwise = RMSEMetric(ctn_cols_idx=ctn_cols_idx)
        self.rmse_colwise = RMSEMetric(ctn_cols_idx=ctn_cols_idx, columnwise=True)
        self.maape_colwise = MAAPEMetric(ctn_cols_idx=ctn_cols_idx, columnwise=True)
        self.torch_metrics = [
            self.rmse_colwise,
            self.rmse_elwise,
            self.maape_elwise,
            self.maape_colwise,
        ]
        self.metrics_list = self.torch_metrics + [
            universal_metric(metric.clone()) for metric in self.torch_metrics
        ]

    def hypothesis_assumptions(self, df: pd.DataFrame):
        assume(
            (not df.isna().any().any())
            and (np.isinf(df).values.sum() == 0)
            # large values in float aren't properly represented and I will get the wrong results
            # numbers are represented with: sign, exponent, fraction
            # so even if a float is just an integer, 1003 => 1.003e3
            # the last digit is at most risk of being messed up/unreliable.
            and ((df.values > 1e10).sum() == 0)
        )

    def test_no_continuous_features(self):
        ctn_cols_idx = list_to_tensor([])
        maape_elwise = MAAPEMetric(ctn_cols_idx=ctn_cols_idx)
        rmse_elwise = RMSEMetric(ctn_cols_idx=ctn_cols_idx)
        rmse_colwise = RMSEMetric(ctn_cols_idx=ctn_cols_idx, columnwise=True)
        maape_colwise = MAAPEMetric(ctn_cols_idx=ctn_cols_idx, columnwise=True)
        torch_metrics = [rmse_colwise, rmse_elwise, maape_elwise, maape_colwise]
        metrics_list = torch_metrics + [
            universal_metric(metric.clone()) for metric in torch_metrics
        ]

        N = 100
        F = 10
        tensor_df = torch.randn((N, F))
        true_df = torch.randn((N, F))

        # Error should be 0 since there's no columns to compute on even if it's EW
        for metric in metrics_list:
            self.assertAlmostEqual(0, metric(tensor_df, true_df).item(), places=WITHIN)
            self.batched_input_equals(metric, tensor_df, true_df)

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_no_mask(self, df):
        self.hypothesis_assumptions(df)
        # apparently hypothesis can't satisfy this assumption along with the others
        # I can't return early or else the whole test dies
        if len(df) > 1:
            tensor_df = torch.tensor(df.values)
            self.setup_metrics(df)

            with self.subTest("All Equal"):
                for metric in self.metrics_list:
                    self.assertAlmostEqual(
                        0, metric(tensor_df, tensor_df).item(), places=WITHIN
                    )
                    self.batched_input_equals(metric, tensor_df, tensor_df)
                for metric in self.torch_metrics:
                    metric.reset()

            # Now if they dont exactly equal each other
            with self.subTest("Not Equal 1 Error in 1 Column"):
                # Create an error in one of the places
                error_df = df.copy()
                diff = 6
                # subtraction happens in np cuz with torch i was getting the wrong values
                ctn_col_idx = hypothesis["ctn_cols_idx"][0]
                error_df.iloc[0, ctn_col_idx] = df.iloc[0, ctn_col_idx] - diff
                error_df = torch.tensor(error_df.values)

                with self.subTest("RMSE"):
                    with self.subTest("CWRMSE"):
                        self.assertAlmostEqual(
                            ((diff**2 / len(df)) ** 0.5) / df.shape[1],
                            self.rmse_colwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.rmse_colwise, error_df, tensor_df
                        )
                    with self.subTest("EWRMSE"):
                        ew_rmse_true = ((diff**2) / len(df) / df.shape[1]) ** 0.5
                        self.assertAlmostEqual(
                            ew_rmse_true,
                            self.rmse_elwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(self.rmse_elwise, error_df, tensor_df)
                with self.subTest("MAAPE"):
                    # cw = ew when there's no mask
                    maape_true = (
                        np.arctan(
                            abs(diff / tensor_df[0, ctn_col_idx] + EPSILON)
                        ).item()
                        / (len(df) * df.shape[1])
                        * 2
                        / pi
                    )
                    with self.subTest("CWMAAPE"):
                        self.assertAlmostEqual(
                            maape_true,
                            self.maape_colwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.maape_colwise, error_df, tensor_df
                        )
                    with self.subTest("EWMAAPE"):
                        self.assertAlmostEqual(
                            maape_true,
                            self.maape_elwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.maape_elwise, error_df, tensor_df
                        )
                for metric in self.torch_metrics:
                    metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_mask(self, df):
        self.hypothesis_assumptions(df)
        # apparently hypothesis can't satisfy this assumption along with the others
        # I can't return early or else the whole test dies
        if len(df) > 1:
            tensor_df = torch.tensor(df.values)
            self.setup_metrics(df)

            ctn_col_idx = hypothesis["ctn_cols_idx"][0]
            missing_indicators = torch.ones_like(tensor_df).to(bool)
            # value at [0,0] will be ignored b/c it is "observed"
            missing_indicators[0, ctn_col_idx] = False

            with self.subTest("All Equal"):
                for metric in self.metrics_list:
                    self.assertAlmostEqual(
                        0,
                        metric(tensor_df, tensor_df, missing_indicators).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(
                        metric, tensor_df, tensor_df, missing_indicators
                    )
                for metric in self.torch_metrics:
                    metric.reset()

            with self.subTest("Observed Value Not Equal"):
                error_df = df.copy()
                diff = 6
                error_df.iloc[0, ctn_col_idx] = df.iloc[0, ctn_col_idx] - diff
                error_df = torch.tensor(error_df.values)

                for metric in self.metrics_list:
                    self.assertAlmostEqual(
                        0,
                        metric(error_df, tensor_df, missing_indicators).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(
                        metric, error_df, tensor_df, missing_indicators
                    )
                for metric in self.torch_metrics:
                    metric.reset()

            with self.subTest("Observed NEQ and Missing NEQ"):
                """
                . = value we don't care about becaues they're the same
                [[..., y1(obs), ..., y3, ...],
                    [..., y2,      ..., y4, ...],
                    [..., .,       ..., .,  ...]]
                Column stats: (covering different test cases)
                    # ignored vals: 1 0 0(...)
                    # missing vals with error: 1 2 0(...)
                """
                diffs = [6, 4, 9, 2]
                err_locs = [
                    [0, hypothesis["ctn_cols_idx"][0]],  # same col (obs/ignored)
                    [1, hypothesis["ctn_cols_idx"][0]],  # same col (missing)
                    [0, hypothesis["ctn_cols_idx"][1]],  # different col (missing)
                    [1, hypothesis["ctn_cols_idx"][1]],  # different col (missing)
                ]
                error_df = df.copy()
                for diff, err_loc in zip(diffs, err_locs):
                    error_df.iloc[err_loc[0], err_loc[1]] = (
                        error_df.iloc[err_loc[0], err_loc[1]] - diff
                    )
                error_df = torch.tensor(error_df.values)
                true_at_err = [  # true values y_i at error locations for MAAPE
                    tensor_df[err_loc[0], err_loc[1]] for err_loc in err_locs
                ]
                nsamples, nfeatures = df.shape
                with self.subTest("RMSE"):
                    with self.subTest("CWRMSE"):
                        self.assertAlmostEqual(
                            (
                                (
                                    # obs value in this column ignored, so 1 less #samples
                                    # 1 error
                                    sqrt(diffs[1] ** 2 / (nsamples - 1))
                                    # no obs values in this column, 2 errors
                                    + sqrt(
                                        ((diffs[2] ** 2) + (diffs[3] ** 2)) / nsamples
                                    )
                                )  # all other columns the same / 0 error
                                / nfeatures  # average over all columns
                            ),
                            self.rmse_colwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.rmse_colwise, error_df, tensor_df, missing_indicators
                        )
                    with self.subTest("EWRMSE"):
                        # all elements but 1 are observed
                        ew_rmse_true = sqrt(
                            (diffs[1] ** 2 + diffs[2] ** 2 + diffs[3] ** 2)
                            / (nsamples * nfeatures - 1)
                        )
                        self.assertAlmostEqual(
                            ew_rmse_true,
                            self.rmse_elwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.rmse_elwise, error_df, tensor_df, missing_indicators
                        )
                with self.subTest("MAAPE"):
                    scale = 2 / pi
                    with self.subTest("CWMAAPE"):
                        self.assertAlmostEqual(
                            (
                                (  # 1 obs value ignored in this col
                                    np.arctan(
                                        abs(diffs[1] / (true_at_err[1] + EPSILON))
                                    )
                                    / (nsamples - 1)
                                )
                                + (  # no obs values ignored in this col, 2 errors
                                    (
                                        np.arctan(
                                            abs(diffs[2] / (true_at_err[2] + EPSILON))
                                        )
                                        + np.arctan(
                                            abs(diffs[3] / (true_at_err[3] + EPSILON))
                                        )
                                    )
                                    / nsamples
                                )
                            ).item()
                            / nfeatures  # average over al cols
                            * scale,  # scale
                            self.maape_colwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.maape_colwise, error_df, tensor_df, missing_indicators
                        )
                    with self.subTest("EWMAAPE"):
                        self.assertAlmostEqual(
                            (
                                np.arctan(abs(diffs[1] / (true_at_err[1] + EPSILON)))
                                + np.arctan(abs(diffs[2] / (true_at_err[2] + EPSILON)))
                                + np.arctan(abs(diffs[3] / (true_at_err[3] + EPSILON)))
                            ).item()  # 1 obs value ignored in entire df
                            / (nsamples * nfeatures - 1)
                            * scale,  # scale
                            self.maape_elwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.maape_elwise, error_df, tensor_df, missing_indicators
                        )
                for metric in self.torch_metrics:
                    metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_mask_nomissing(self, df):
        """
        When we pass a mask with no missing values
            1. everywhere (EW edge case)
            2. some columns have no missing values (CW edge case)
        """
        self.hypothesis_assumptions(df)
        # apparently hypothesis can't satisfy this assumption along with the others
        # I can't return early or else the whole test dies
        if len(df) > 1:
            tensor_df = torch.tensor(df.values)
            self.setup_metrics(df)
            diffs = [6, 4, 9, 2]
            err_locs = [
                [0, hypothesis["ctn_cols_idx"][0]],
                [1, hypothesis["ctn_cols_idx"][0]],
                [0, hypothesis["ctn_cols_idx"][1]],
                [1, hypothesis["ctn_cols_idx"][1]],
            ]
            error_df = df.copy()
            for diff, err_loc in zip(diffs, err_locs):
                error_df.iloc[err_loc[0], err_loc[1]] = (
                    error_df.iloc[err_loc[0], err_loc[1]] - diff
                )
            error_df = torch.tensor(error_df.values)
            true_at_err = [  # true values y_i at error locations for MAAPE
                tensor_df[err_loc[0], err_loc[1]] for err_loc in err_locs
            ]
            nsamples, nfeatures = df.shape

            with self.subTest("No Missing"):
                missing_indicators = torch.zeros_like(tensor_df).to(bool)
                for metric in self.metrics_list:
                    # because there are no missing values at all the errors for missingonly should be 0
                    self.assertAlmostEqual(
                        0,
                        metric(error_df, tensor_df, missing_indicators).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(
                        metric, error_df, tensor_df, missing_indicators
                    )
                for metric in self.torch_metrics:
                    metric.reset()
            with self.subTest(
                "Columns with no Missingness (1 Col with Missing Values)"
            ):
                missing_indicators = torch.zeros_like(tensor_df).to(bool)
                # make the second ctn column to be missing
                missing_indicators[:, hypothesis["ctn_cols_idx"][1]] = True
                with self.subTest("RMSE"):
                    with self.subTest("CWRMSE"):
                        self.assertAlmostEqual(
                            (  # no obs values in 2nd column, 2 errors
                                (sqrt(((diffs[2] ** 2) + (diffs[3] ** 2)) / nsamples))
                                / 1  # average over all 1 missing column
                            ),
                            self.rmse_colwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.rmse_colwise,
                            error_df,
                            tensor_df,
                            missing_indicators,
                        )
                    with self.subTest("EWRMSE"):
                        # all obs minus 1 column
                        ew_rmse_true = sqrt((diffs[2] ** 2 + diffs[3] ** 2) / nsamples)
                        self.assertAlmostEqual(
                            ew_rmse_true,
                            self.rmse_elwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.rmse_elwise, error_df, tensor_df, missing_indicators
                        )
                with self.subTest("MAAPE"):
                    scale = 2 / pi
                    with self.subTest("CWMAAPE"):
                        self.assertAlmostEqual(
                            (  # no obs values ignored in this col, 2 errors
                                (
                                    np.arctan(
                                        abs(diffs[2] / (true_at_err[2] + EPSILON))
                                    )
                                    + np.arctan(
                                        abs(diffs[3] / (true_at_err[3] + EPSILON))
                                    )
                                )
                                / nsamples
                            ).item()
                            / 1  # average over the 1 missing col
                            * scale,  # scale
                            self.maape_colwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.maape_colwise,
                            error_df,
                            tensor_df,
                            missing_indicators,
                        )
                    with self.subTest("EWMAAPE"):
                        self.assertAlmostEqual(
                            (
                                np.arctan(abs(diffs[2] / (true_at_err[2] + EPSILON)))
                                + np.arctan(abs(diffs[3] / (true_at_err[3] + EPSILON)))
                            ).item()
                            / nsamples  # all columns observed except one
                            * scale,  # scale
                            self.maape_elwise(
                                error_df, tensor_df, missing_indicators
                            ).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            self.maape_elwise,
                            error_df,
                            tensor_df,
                            missing_indicators,
                        )
                for metric in self.torch_metrics:
                    metric.reset()


class TestStaticMulticatCategoricalMetrics(unittest.TestCase):
    def batched_input_equals(self, metric, *inputs):
        batched_input_equals(self, metric, *inputs)

    def setup_metrics(self, df: pd.DataFrame) -> None:
        self.err_elwise = CategoricalErrorMetric(
            list_to_tensor(hypothesis["cat_cols_idx"]), []
        )
        self.err_colwise = CategoricalErrorMetric(
            list_to_tensor(hypothesis["cat_cols_idx"]), [], True
        )
        self.torch_metrics = [self.err_elwise, self.err_colwise]

    def hypothesis_assumptions(self, df: pd.DataFrame):
        assume(
            (not df.isna().any().any())
            and (len(df) > 1)
            and (np.isinf(df).values.sum() == 0)
        )

    def test_no_categorical_features(self):
        err_colwise = CategoricalErrorMetric(
            list_to_tensor([]), list_to_tensor([]), True
        )
        err_elwise = CategoricalErrorMetric(list_to_tensor([]), list_to_tensor([]))
        torch_metrics = [err_elwise, err_colwise]

        N = 100
        F = 10
        tensor_df = torch.randn((N, F))
        true_df = torch.randn((N, F))

        # Error should be 0 since there's no columns to compute on even if it's EW
        for metric in torch_metrics:
            self.assertAlmostEqual(0, metric(tensor_df, true_df).item(), places=WITHIN)
            self.batched_input_equals(metric, tensor_df, true_df)
            metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_no_mask(self, df):
        self.hypothesis_assumptions(df)
        self.setup_metrics(df)
        tensor_df = torch.tensor(df.values)
        bin_col_idx = hypothesis["cat_cols_idx"][0]

        with self.subTest("All Equal"):
            for metric in self.torch_metrics:
                self.assertAlmostEqual(
                    0, metric(tensor_df, tensor_df).item(), places=WITHIN
                )
                self.batched_input_equals(metric, tensor_df, tensor_df)
                metric.reset()

        N = len(df)
        F = len(hypothesis["cat_cols_idx"])
        with self.subTest("Not Equal 1 Error in 1 Column"):
            # Create an error in one of the places
            error_df = df.copy()
            # flip a binary col
            error_df.iloc[0, bin_col_idx] = 1 - df.iloc[0, bin_col_idx]
            error_df = torch.tensor(error_df.values)
            # these are all the same actually
            with self.subTest("Colwise Error"):
                self.assertAlmostEqual(  # 1 error for 1 feature out of F
                    (1 / N) / F,
                    self.err_colwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(self.err_colwise, error_df, tensor_df)
            with self.subTest("Elwise Error"):
                self.assertAlmostEqual(  # 1 error out of all the cells
                    1 / (N * F),
                    self.err_elwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(self.err_elwise, error_df, tensor_df)
            for metric in self.torch_metrics:
                metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_mask(self, df):
        self.hypothesis_assumptions(df)
        self.setup_metrics(df)
        tensor_df = torch.tensor(df.values)
        bin_col_idx = hypothesis["cat_cols_idx"][0]

        missing_indicators = torch.ones_like(tensor_df).to(bool)
        missing_indicators[0, bin_col_idx] = False

        with self.subTest("All Equal"):
            for metric in self.torch_metrics:
                self.assertAlmostEqual(
                    0,
                    metric(tensor_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    metric, tensor_df, tensor_df, missing_indicators
                )
                metric.reset()

        with self.subTest("Not Equal inside Mask"):
            # Create an error in one of the places
            error_df = df.copy()
            # flip a binary col
            error_df.iloc[0, bin_col_idx] = 1 - df.iloc[0, bin_col_idx]
            error_df = torch.tensor(error_df.values)
            for metric in self.torch_metrics:
                self.assertAlmostEqual(
                    0,
                    metric(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    metric, error_df, tensor_df, missing_indicators
                )
                metric.reset()

        with self.subTest("Not Equal outside mask"):
            """
            . = value we don't care about becaues they're the same
            [[..., e1(obs), ..., e3, ...],
             [..., e2,      ..., e4, ...],
             [..., .,       ..., .,  ...]]
            Column stats: (covering different test cases)
                # ignored vals: 1 0 0(...)
                # missing vals with error: 1 2 0(...)
            """
            err_locs = [
                [0, hypothesis["bin_cols_idx"][0]],  # same col (obs/ignored)
                [1, hypothesis["bin_cols_idx"][0]],  # same col (missing)
                [0, hypothesis["bin_cols_idx"][1]],  # different col (missing)
                [1, hypothesis["bin_cols_idx"][1]],  # different col (missing)
            ]
            error_df = df.copy()
            for err_loc in err_locs:
                # flip a binary col
                error_df.iloc[err_loc[0], err_loc[1]] = (
                    1 - error_df.iloc[err_loc[0], err_loc[1]]
                )
            error_df = torch.tensor(error_df.values)
            N = len(df)
            F = len(hypothesis["cat_cols_idx"])
            self.assertAlmostEqual(
                # nsamples in col with obs is N-1, and then 1 error
                # nsamples is the other col is all N, but 2 errors
                # (thats 2 cols) For the remaining cols err is 0 (F - 2)
                ((1 / (N - 1)) + (2 / N)) / F,
                self.err_colwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_colwise, error_df, tensor_df, missing_indicators
            )
            self.assertAlmostEqual(
                # 1 wrong and ignored in first col, 2 wrong in second, and the remaining F-2 cols have all right
                (1 + 2) / (N * F - 1),  # 1 ignored so 1 cell less
                self.err_elwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_elwise, error_df, tensor_df, missing_indicators
            )
            for metric in self.torch_metrics:
                metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_mask_nomissing(self, df):
        """
        When we pass a mask with no missing values
            1. everywhere (EW edge case)
            2. some columns have no missing values (CW edge case)
        """
        self.hypothesis_assumptions(df)
        self.setup_metrics(df)
        tensor_df = torch.tensor(df.values)

        """
        . = value we don't care about becaues they're the same
        [[..., y1(obs), ..., y3, ...],
        [..., y2,      ..., y4, ...],
        [..., .,       ..., .,  ...]]
        Column stats: (covering different test cases)
            # ignored vals: 1 0 0(...)
            # missing vals with error: 1 2 0(...)
        """
        err_locs = [
            [0, hypothesis["bin_cols_idx"][0]],
            [1, hypothesis["bin_cols_idx"][0]],
            [0, hypothesis["bin_cols_idx"][1]],
            [1, hypothesis["bin_cols_idx"][1]],
        ]
        error_df = df.copy()
        for err_loc in err_locs:
            # flip a binary col
            error_df.iloc[err_loc[0], err_loc[1]] = (
                1 - error_df.iloc[err_loc[0], err_loc[1]]
            )
        error_df = torch.tensor(error_df.values)
        N = len(df)
        F = len(hypothesis["cat_cols_idx"])

        with self.subTest("No Missing"):
            missing_indicators = torch.zeros_like(tensor_df).to(bool)
            for metric in self.torch_metrics:
                # because there are no missing values at all the errors for missingonly should be 0
                self.assertAlmostEqual(
                    0,
                    metric(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    metric, error_df, tensor_df, missing_indicators
                )
                metric.reset()

        with self.subTest("Columns with no Missingness (1 Col with Missing Values)"):
            missing_indicators = torch.zeros_like(tensor_df).to(bool)
            # make the first bin column to be missing
            missing_indicators[:, hypothesis["bin_cols_idx"][0]] = True
            self.assertAlmostEqual(
                # nsamples is the missing col is all N, but 2 errors
                # For the remaining F - 1, we don't care bc they're observed
                2 / N,
                self.err_colwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_colwise, error_df, tensor_df, missing_indicators
            )
            self.assertAlmostEqual(
                # 2 wrong in missing ocl, and the remaining F-1 cols have all right
                2 / N,  # all but 1 col ignored, only N missing samples
                self.err_elwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_elwise, error_df, tensor_df, missing_indicators
            )
            for metric in self.torch_metrics:
                metric.reset()


class TestStaticOnehotCategoricalMetrics(unittest.TestCase):
    def batched_input_equals(self, metric, *inputs):
        batched_input_equals(self, metric, *inputs)

    def setup_metrics(self, df: pd.DataFrame) -> None:
        self.err_colwise = CategoricalErrorMetric(
            list_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
            list_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
            True,
        )
        self.err_elwise = CategoricalErrorMetric(
            list_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
            list_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
        )
        self.torch_metrics = [self.err_elwise, self.err_colwise]

    def setup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace({np.nan: 400, np.inf: 500})
        df = build_onehot_from_hypothesis(df, hypothesis["onehot_prefixes"])
        return df

    def hypothesis_assumptions(self, df: pd.DataFrame):
        # don't need to test continuous
        # Ensure all categories/cols present for testing
        assume(
            np.array_equal(
                df.nunique()[hypothesis["onehot_prefixes"]].values, np.array([4, 3])
            )
        )

    def make_onehot_error(
        self,
        df: pd.DataFrame,
        error_df: pd.DataFrame,
        sample_idx: int,
        onehot_group: List[int],
    ) -> pd.DataFrame:
        C = len(onehot_group)  # num categories
        # change one of the onehots
        old_cat = np.argmax(df.iloc[sample_idx, onehot_group])
        new_cat = C - old_cat
        error_df.iloc[sample_idx, onehot_group] = np.eye(C)[
            new_cat - 1 if (new_cat == old_cat or old_cat == 0) else new_cat
        ]
        return error_df

    def test_no_categorical_features(self):
        err_colwise = CategoricalErrorMetric(
            list_to_tensor([]), list_to_tensor([]), True
        )
        err_elwise = CategoricalErrorMetric(list_to_tensor([]), list_to_tensor([]))
        torch_metrics = [err_elwise, err_colwise]

        N = 100
        F = 10
        tensor_df = torch.randn((N, F))
        true_df = torch.randn((N, F))

        # Error should be 0 since there's no columns to compute on even if it's EW
        for metric in torch_metrics:
            self.assertAlmostEqual(0, metric(tensor_df, true_df).item(), places=WITHIN)
            self.batched_input_equals(metric, tensor_df, true_df)
            metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_no_mask(self, df):
        self.hypothesis_assumptions(df)
        self.setup_metrics(df)
        df = self.setup_data(df)

        tensor_df = torch.tensor(df.values)

        onehot_group = hypothesis["onehot"]["onehot_cols_idx"][0]
        N = len(df)  # num samples
        F = len(hypothesis["cat_cols_idx"])  # num features

        with self.subTest("All Equal"):
            for metric in self.torch_metrics:
                self.assertAlmostEqual(
                    0,
                    metric(tensor_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(metric, tensor_df, tensor_df)
                metric.reset()

        with self.subTest("Not Equal (1 Error in 1 Column)"):
            # Create an error in one of the onehots
            error_df = df.copy()
            error_df = self.make_onehot_error(
                df, error_df, sample_idx=0, onehot_group=onehot_group
            )
            error_df = torch.tensor(error_df.values)

            # these are all the same
            self.assertAlmostEqual(  # 1 error for 1 feature out of F
                (1 / N) / F,
                self.err_colwise(error_df, tensor_df).item(),
                places=WITHIN,
            )
            self.batched_input_equals(self.err_colwise, error_df, tensor_df)
            self.assertAlmostEqual(  # one error out of all the cells
                1 / (N * F),
                self.err_elwise(error_df, tensor_df).item(),
                places=WITHIN,
            )
            self.batched_input_equals(self.err_elwise, error_df, tensor_df)
            for metric in self.torch_metrics:
                metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_mask(self, df):
        self.hypothesis_assumptions(df)
        self.setup_metrics(df)
        df = self.setup_data(df)

        N = len(df)  # num samples
        F = len(hypothesis["cat_cols_idx"])  # num features

        tensor_df = torch.tensor(df.values)

        onehot_group = hypothesis["onehot"]["onehot_cols_idx"][0]
        missing_indicators = torch.ones_like(tensor_df).to(bool)
        missing_indicators[0, onehot_group] = False
        with self.subTest("All Equal"):
            for metric in self.torch_metrics:
                self.assertAlmostEqual(
                    0,
                    metric(tensor_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    metric, tensor_df, tensor_df, missing_indicators
                )
                metric.reset()

        with self.subTest("Observed Not Equal"):
            error_df = df.copy()
            error_df = self.make_onehot_error(
                df, error_df, sample_idx=0, onehot_group=onehot_group
            )
            error_df = torch.tensor(error_df.values)
            for metric in self.torch_metrics:
                self.assertAlmostEqual(
                    0,
                    metric(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    metric, error_df, tensor_df, missing_indicators
                )
                metric.reset()

        with self.subTest("Observed and Missing Not Equal"):
            error_df = df.copy()
            onehot_groups = hypothesis["onehot"]["onehot_cols_idx"][:2]
            """
            onehot group 1: has 2 errors, but 1 is observed and 1 is missing
            onehot group 2: has 2 errors, but both are missing.
            """
            for onehot_group in onehot_groups:
                error_df = self.make_onehot_error(
                    df, error_df, sample_idx=0, onehot_group=onehot_group
                )
                error_df = self.make_onehot_error(
                    df, error_df, sample_idx=1, onehot_group=onehot_group
                )
            error_df = torch.tensor(error_df.values)
            # 2 error, but ignoring 1 position
            self.assertAlmostEqual(
                # nsamples in col with obs is N-1, and then 1 error
                # nsamples is the other col is all N, but 2 errors
                # (thats 2 cols) For the remaining cols err is 0 (F - 2)
                ((1 / (N - 1)) + (2 / N)) / F,
                self.err_colwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_colwise, error_df, tensor_df, missing_indicators
            )
            self.assertAlmostEqual(
                # 1 wrong and ignored in first col, 2 wrong in second, and the remaining F-2 cols have all right
                (1 + 2) / (N * F - 1),  # 1 ignored so 1 cell less
                self.err_elwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_elwise, error_df, tensor_df, missing_indicators
            )
            for metric in self.torch_metrics:
                metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_mask_nomissing(self, df):
        self.hypothesis_assumptions(df)
        self.setup_metrics(df)
        df = self.setup_data(df)
        tensor_df = torch.tensor(df.values)
        N = len(df)  # num samples
        F = len(hypothesis["cat_cols_idx"])  # num features

        """
        onehot group 1: has 2 errors, but 1 is observed and 1 is missing
        onehot group 2: has 2 errors, but both are missing.
        """
        onehot_groups = hypothesis["onehot"]["onehot_cols_idx"][:2]
        error_df = df.copy()
        for onehot_group in onehot_groups:
            error_df = self.make_onehot_error(
                df, error_df, sample_idx=0, onehot_group=onehot_group
            )
            error_df = self.make_onehot_error(
                df, error_df, sample_idx=1, onehot_group=onehot_group
            )
        error_df = torch.tensor(error_df.values)

        with self.subTest("No missing"):
            missing_indicators = torch.zeros_like(tensor_df).to(bool)
            for metric in self.torch_metrics:
                # because there are no missing values at all the errors for missingonly should be 0
                self.assertAlmostEqual(
                    0,
                    metric(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    metric, error_df, tensor_df, missing_indicators
                )
                metric.reset()

        with self.subTest("Columns with no Missingness (1 Col with Missing Values)"):
            missing_indicators = torch.zeros_like(tensor_df).to(bool)
            # make the first onehot groups of columns to be missing
            missing_indicators[:, hypothesis["onehot"]["onehot_cols_idx"][0]] = True
            self.assertAlmostEqual(
                # nsamples is the missing col is all N, but 2 errors
                # For the remaining F - 1, they're all observed so we don't care
                2 / N,
                self.err_colwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_colwise, error_df, tensor_df, missing_indicators
            )
            self.assertAlmostEqual(
                # 2 wrong in missing ocl, and the remaining F-1 cols have all right
                2 / N,  # all but 1 col ignored, only N missing samples
                self.err_elwise(error_df, tensor_df, missing_indicators).item(),
                places=WITHIN,
            )
            self.batched_input_equals(
                self.err_elwise, error_df, tensor_df, missing_indicators
            )
            for metric in self.torch_metrics:
                metric.reset()


if __name__ == "__main__":
    unittest.main()
