from math import pi, sqrt
import unittest
import numpy as np
import pandas as pd
import torch
from hypothesis import assume, given, HealthCheck, settings, strategies as st
from hypothesis.extra.pandas import data_frames
from torchmetrics import Metric

from autopopulus.utils.impute_metrics import (
    EPSILON,
    AccuracyMetric,
    MAAPEMetric,
    RMSEMetric,
    universal_metric,
)
from autopopulus.test.common_mock_data import hypothesis
from autopopulus.test.utils import build_onehot_from_hypothesis
from autopopulus.data.transforms import list_to_tensor


# when subtracting, tensors add in a little margin of error that accumulates, so we want to get close within WITHIN decimal places.
WITHIN = 6


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

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
            self.assertAlmostEqual(
                batch_metric.compute().item(), metric(*inputs).item()
            )
            metric.reset()

    @settings(
        suppress_health_check=[HealthCheck(3), HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(data_frames(columns=hypothesis["columns"]))
    def test_static_multicat_continuous(self, df):
        assume(
            (not df.isna().any().any())
            and (np.isinf(df).values.sum() == 0)
            # large values in float aren't properly represented and I will get the wrong results
            # numbers are represented with: sign, exponent, fraction
            # so even if a float is just an integer, 1003 => 1.003e3
            # the last digit is at most risk of being messed up/unreliable.
            and ((df.values > 1e10).sum() == 0)
        )
        # apparently hypothesis can't satisfy this assumption along with the others
        # I can't return early or else the whole test dies
        if len(df) > 1:
            tensor_df = torch.tensor(df.values)

            maape_elwise = MAAPEMetric()
            rmse_elwise = RMSEMetric()
            rmse_colwise = RMSEMetric(columnwise=True, nfeatures=df.shape[1])
            maape_colwise = MAAPEMetric(columnwise=True, nfeatures=df.shape[1])
            torch_metrics = [rmse_colwise, rmse_elwise, maape_elwise, maape_colwise]
            metrics_list = torch_metrics + [
                universal_metric(metric.clone()) for metric in torch_metrics
            ]

            with self.subTest("No Mask - Basic"):
                for metric in metrics_list:
                    self.assertAlmostEqual(
                        0, metric(tensor_df, tensor_df).item(), places=WITHIN
                    )
                    self.batched_input_equals(metric, tensor_df, tensor_df)
                for metric in torch_metrics:
                    metric.reset()

                ctn_col_idx = hypothesis["ctn_cols_idx"][0]

                # Now if they dont exactly equal each other
                with self.subTest("Not Equal 1 Error in 1 Column"):
                    # Create an error in one of the places
                    error_df = df.copy()
                    diff = 6
                    # subtraction happens in np cuz with torch i was getting the wrong values
                    error_df.iloc[0, ctn_col_idx] = df.iloc[0, ctn_col_idx] - diff
                    error_df = torch.tensor(error_df.values)

                    with self.subTest("RMSE"):
                        self.assertAlmostEqual(
                            ((diff**2 / len(df)) ** 0.5) / df.shape[1],
                            rmse_colwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(rmse_colwise, error_df, tensor_df)
                        ew_rmse_true = ((diff**2) / len(df) / df.shape[1]) ** 0.5
                        self.assertAlmostEqual(
                            ew_rmse_true,
                            rmse_elwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(rmse_elwise, error_df, tensor_df)
                    with self.subTest("MAAPE"):
                        maape_true = (
                            np.arctan(
                                abs(diff / tensor_df[0, ctn_col_idx] + EPSILON)
                            ).item()
                            / (len(df) * df.shape[1])
                            * 2
                            / pi
                        )
                        self.assertAlmostEqual(
                            maape_true,
                            maape_elwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(maape_elwise, error_df, tensor_df)
                        # cw = ew when there's no mask
                        self.assertAlmostEqual(
                            maape_true,
                            maape_colwise(error_df, tensor_df).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(maape_colwise, error_df, tensor_df)
                    for metric in torch_metrics:
                        metric.reset()

            with self.subTest("Mask"):
                missing_indicators = torch.ones_like(tensor_df).to(bool)
                # value at [0,0] will be ignored b/c it is "observed"
                missing_indicators[0, ctn_col_idx] = False
                for metric in metrics_list:
                    self.assertAlmostEqual(
                        0,
                        metric(tensor_df, tensor_df, missing_indicators).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(
                        metric, tensor_df, tensor_df, missing_indicators
                    )
                for metric in torch_metrics:
                    metric.reset()

                with self.subTest("Observed Value Not Equal"):
                    error_df = df.copy()
                    error_df.iloc[0, ctn_col_idx] = df.iloc[0, ctn_col_idx] - diff
                    error_df = torch.tensor(error_df.values)

                    for metric in metrics_list:
                        self.assertAlmostEqual(
                            0,
                            metric(error_df, tensor_df, missing_indicators).item(),
                            places=WITHIN,
                        )
                        self.batched_input_equals(
                            metric, error_df, tensor_df, missing_indicators
                        )
                    for metric in torch_metrics:
                        metric.reset()

                with self.subTest("Observed NEQ and Missing NEQ (Same Col)"):
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
                                            ((diffs[2] ** 2) + (diffs[3] ** 2))
                                            / nsamples
                                        )
                                    )  # all other columns the same / 0 error
                                    / nfeatures  # average over all columns
                                ),
                                rmse_colwise(
                                    error_df, tensor_df, missing_indicators
                                ).item(),
                                places=WITHIN,
                            )
                            self.batched_input_equals(
                                rmse_colwise, error_df, tensor_df, missing_indicators
                            )
                        with self.subTest("EWRMSE"):
                            # all elements but 1 are observed
                            ew_rmse_true = sqrt(
                                (diffs[1] ** 2 + diffs[2] ** 2 + diffs[3] ** 2)
                                / (nsamples * nfeatures - 1)
                            )
                            self.assertAlmostEqual(
                                ew_rmse_true,
                                rmse_elwise(
                                    error_df, tensor_df, missing_indicators
                                ).item(),
                                places=WITHIN,
                            )
                            self.batched_input_equals(
                                rmse_elwise, error_df, tensor_df, missing_indicators
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
                                                abs(
                                                    diffs[2]
                                                    / (true_at_err[2] + EPSILON)
                                                )
                                            )
                                            + np.arctan(
                                                abs(
                                                    diffs[3]
                                                    / (true_at_err[3] + EPSILON)
                                                )
                                            )
                                        )
                                        / nsamples
                                    )
                                ).item()
                                / nfeatures  # average over al cols
                                * scale,  # scale
                                maape_colwise(
                                    error_df, tensor_df, missing_indicators
                                ).item(),
                                places=WITHIN,
                            )
                            self.batched_input_equals(
                                maape_colwise, error_df, tensor_df, missing_indicators
                            )
                        with self.subTest("EWMAAPE"):
                            self.assertAlmostEqual(
                                (
                                    np.arctan(
                                        abs(diffs[1] / (true_at_err[1] + EPSILON))
                                    )
                                    + np.arctan(
                                        abs(diffs[2] / (true_at_err[2] + EPSILON))
                                    )
                                    + np.arctan(
                                        abs(diffs[3] / (true_at_err[3] + EPSILON))
                                    )
                                )  # 1 obs value ignored in entire df
                                / (nsamples * nfeatures - 1)
                                * scale,  # scale
                                maape_elwise(
                                    error_df, tensor_df, missing_indicators
                                ).item(),
                                places=WITHIN,
                            )
                            self.batched_input_equals(
                                maape_elwise, error_df, tensor_df, missing_indicators
                            )
                    for metric in torch_metrics:
                        metric.reset()

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

        accuracy_elwise = AccuracyMetric(list_to_tensor(hypothesis["cat_cols_idx"]), [])
        accuracy_colwise = AccuracyMetric(
            list_to_tensor(hypothesis["cat_cols_idx"]), [], True
        )
        torch_metrics = [accuracy_elwise, accuracy_colwise]

        with self.subTest("No Mask"):
            with self.subTest("All Equal"):
                for metric in torch_metrics:
                    self.assertAlmostEqual(
                        1, metric(tensor_df, tensor_df).item(), places=WITHIN
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
                with self.subTest("Colwise Accuracy"):
                    self.assertAlmostEqual(  # 1 error for 1 feature out of F
                        (((N - 1) / N) + F - 1) / F,
                        accuracy_colwise(error_df, tensor_df).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(accuracy_colwise, error_df, tensor_df)
                with self.subTest("Elwise Accuracy"):
                    self.assertAlmostEqual(  # 1 error out of all the cells
                        (N * F - 1) / (N * F),
                        accuracy_elwise(error_df, tensor_df).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(accuracy_elwise, error_df, tensor_df)
                for metric in torch_metrics:
                    metric.reset()

        with self.subTest("Mask"):
            missing_indicators = torch.ones_like(tensor_df).to(bool)
            missing_indicators[0, bin_col_idx] = False
            with self.subTest("Not Equal inside Mask"):
                for metric in torch_metrics:
                    self.assertAlmostEqual(
                        1,
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
                [[..., y1(obs), ..., y3, ...],
                [..., y2,      ..., y4, ...],
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
                self.assertAlmostEqual(
                    # nsamples in col with obs is N-1, and then 1 error
                    # nsamples is the other col is all N, but 2 errors
                    # (thats 2 cols) For the remaining cols acc is 1 (F - 2)
                    (((N - 1 - 1) / (N - 1)) + ((N - 2) / N) + (F - 2)) / F,
                    accuracy_colwise(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    accuracy_colwise, error_df, tensor_df, missing_indicators
                )
                self.assertAlmostEqual(
                    # 1 wrong and ignored in first col, 2 wrong in second, and the remaining F-2 cols have all right
                    ((N - 2) + (N - 2) + N * (F - 2))
                    / (N * F - 1),  # 1 ignored so 1 cell less
                    accuracy_elwise(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    accuracy_elwise, error_df, tensor_df, missing_indicators
                )
                for metric in torch_metrics:
                    metric.reset()

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
        accuracy_colwise = AccuracyMetric(
            list_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
            list_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
            True,
        )
        accuracy_elwise = AccuracyMetric(
            list_to_tensor(hypothesis["onehot"]["bin_cols_idx"]),
            list_to_tensor(hypothesis["onehot"]["onehot_cols_idx"]),
        )
        torch_metrics = [accuracy_elwise, accuracy_colwise]

        with self.subTest("No Mask"):
            with self.subTest("All Equal"):
                for metric in torch_metrics:
                    self.assertAlmostEqual(
                        1,
                        metric(tensor_df, tensor_df).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(metric, tensor_df, tensor_df)
                    metric.reset()

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

                # these are all the same
                error_df = torch.tensor(error_df.values)
                self.assertAlmostEqual(  # 1 error for 1 feature out of F
                    (((N - 1) / N) + F - 1) / F,
                    accuracy_colwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(accuracy_colwise, error_df, tensor_df)
                self.assertAlmostEqual(  # one error out of all the cells
                    (N * F - 1) / (N * F),
                    accuracy_elwise(error_df, tensor_df).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(accuracy_elwise, error_df, tensor_df)
                for metric in torch_metrics:
                    metric.reset()

        with self.subTest("Mask"):
            missing_indicators = torch.ones_like(tensor_df).to(bool)
            missing_indicators[0, onehot_group] = False
            with self.subTest("Not Equal inside Mask"):
                for metric in torch_metrics:
                    self.assertAlmostEqual(
                        1,
                        metric(error_df, tensor_df, missing_indicators).item(),
                        places=WITHIN,
                    )
                    self.batched_input_equals(
                        metric, error_df, tensor_df, missing_indicators
                    )
                    metric.reset()

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
                    accuracy_colwise(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    accuracy_colwise, error_df, tensor_df, missing_indicators
                )
                self.assertAlmostEqual(
                    ((N * F - 1) - 1)
                    / (
                        N * F - 1
                    ),  # error in 1 non-masked cells when theres 1 cell less
                    accuracy_elwise(error_df, tensor_df, missing_indicators).item(),
                    places=WITHIN,
                )
                self.batched_input_equals(
                    accuracy_elwise, error_df, tensor_df, missing_indicators
                )
                for metric in torch_metrics:
                    metric.reset()


if __name__ == "__main__":
    unittest.main()
