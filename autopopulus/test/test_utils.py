import unittest
from unittest.mock import ANY, MagicMock, call, patch

import pandas as pd
import numpy as np
from numpy.random import default_rng
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# validation for mocking _score and _search for _fit_and_score
from sklearn.model_selection import _validation, _search
from sklearn.pipeline import FunctionTransformer

from autopopulus.models.evaluation import (
    bootstrap_confidence_interval,
    confidence_interval,
)
from autopopulus.models.sklearn_model_utils import TransformScorer, TunableEstimator
from autopopulus.task_logic.baseline_static_imputation import (
    BASELINE_IMPUTER_MODEL_PARAM_GRID,
)
from autopopulus.test.common_mock_data import X, y, seed, splits
from autopopulus.data.dataset_classes import CommonDataModule
from autopopulus.test.utils import get_dataset_loader
from autopopulus.utils.impute_metrics import MAAPEMetric, RMSEMetric, universal_metric
from autopopulus.models.prediction_models import PREDICTOR_MODEL_METADATA


class TestUtils(unittest.TestCase):
    def test_bootstrap_confidence_interval(self):
        # following example 2 from https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        data = [45, 55, 67, 45, 68, 79, 98, 87, 84, 82]
        np.testing.assert_allclose(
            confidence_interval(data, 0.98), 16.22075, rtol=0.001
        )


class TestTransformScorer(unittest.TestCase):
    def test_basic(self):
        """Example taken from test_metrics."""
        true = X["nomissing"]
        pred = X["nomissing"].copy()
        diff = 6
        pred.iloc[0, 0] = pred.iloc[0, 0] - diff

        # return the "messed up" version
        input_data = true
        estimator = FunctionTransformer(lambda data: pred)
        # the tuning tests use MAAPE so we'll try RMSE here
        scorer = TransformScorer(universal_metric(RMSEMetric()), higher_is_better=False)
        ew_rmse_true = ((diff**2) / len(true) / true.shape[1]) ** 0.5
        # multiply by -1 because sign will be -1 for the score (higher is not better)
        self.assertEqual(scorer(estimator, input_data, true), -1 * ew_rmse_true)

    def test_missing_only(self):
        """Example taken from test_metrics."""
        true = X["nomissing"]
        input_data = X["nomissing"].copy()
        input_data.iloc[0, 1] = np.nan
        # the tuning tests use MAAPE so we'll try RMSE here
        scorer = TransformScorer(
            universal_metric(RMSEMetric()), higher_is_better=False, missingonly=True
        )
        with self.subTest("Error On Observed Value"):
            # mess up the index to make sure it's set properly later
            pred = X["nomissing"].copy().reset_index(drop=True)
            diff = 6
            pred.iloc[0, 0] = pred.iloc[0, 0] - diff
            # return the "messed up" version
            estimator = FunctionTransformer(lambda data: pred)
            self.assertEqual(scorer(estimator, input_data, true), 0)

        with self.subTest("Error On Missing Value"):
            # mess up the index to make sure it's set properly later
            pred = X["nomissing"].copy().reset_index(drop=True)
            diff = 6
            pred.iloc[0, 1] = pred.iloc[0, 1] - diff
            # return the "messed up" version
            estimator = FunctionTransformer(lambda data: pred)
            # only 1 value missing
            ew_rmse_true = ((diff**2) / 1) ** 0.5
            self.assertEqual(scorer(estimator, input_data, true), -1 * ew_rmse_true)

    def test_ground_truth_missing(self):
        X_true = X["X"]  # has missing values
        input_data = X["X"]
        X_pred = X["nomissing"].copy()
        diff = 6
        X_pred.iloc[0, 0] = X_pred.iloc[0, 0] - diff
        estimator = FunctionTransformer(lambda data: X_pred)
        # the tuning tests use MAAPE so we'll try RMSE here
        scorer = TransformScorer(
            universal_metric(RMSEMetric()), higher_is_better=False, missingonly=True
        )
        # When GT is missing it should fill with pred and the error should be 0
        self.assertEqual(scorer(estimator, input_data, X_true), 0)


class TestTunableEstimator(unittest.TestCase):
    # _fit_and_score calls _score in _validation
    @patch("autopopulus.data.dataset_classes.train_test_split")
    @patch.object(_search, "_fit_and_score", wraps=_search._fit_and_score)
    def test_transformer(self, mock_score, mock_split):
        mock_split.side_effect = [
            (splits["train"] + splits["val"], splits["test"]),
            (splits["train"], splits["val"]),
        ]
        data_settings = {
            # allow missing for imputer transformer
            "dataset_loader": get_dataset_loader(X["X"], y),
            "seed": seed,
            "test_size": 0.5,
            "val_size": 0.5,
            "batch_size": 2,
        }
        datamodule = CommonDataModule(**data_settings, scale=True)
        datamodule.setup("fit")

        imputer = TunableEstimator(
            KNNImputer(),
            BASELINE_IMPUTER_MODEL_PARAM_GRID["knn"],
            # higher is not better because it's an error
            score_fn=TransformScorer(
                universal_metric(MAAPEMetric()), higher_is_better=False
            ),
        )

        imputer.fit(
            datamodule.splits["data"],
            # TODO: this should be datamodule.splits["ground_truth"]
            # TODO: i need to be able to deal with when ground_truth has nans in it for baselines like I have for AE imputation
            {
                split: X["nomissing"].iloc[splits[split]]
                for split in ["train", "val", "test"]
            },
        )
        # make sure the test set is correctly the val set
        for call_args in mock_score.call_args_list:
            true_call = call(
                ANY,
                X["X"].iloc[splits["train"] + splits["val"]],
                X["nomissing"].iloc[splits["train"] + splits["val"]],
                train=splits["train"],
                test=splits["val"],
                parameters=ANY,
                split_progress=ANY,
                candidate_progress=ANY,
            )
            for arg, true_arg in zip(call_args[0], true_call.args):
                if isinstance(arg, pd.DataFrame):
                    pd.testing.assert_frame_equal(arg, true_arg)
                else:
                    self.assertEqual(arg, true_arg)
                    # np.testing.assert_allclose(arg, true_arg)
            for kwarg, true_kwarg in zip(call_args[1], true_call.kwargs):
                self.assertEqual(kwarg, true_kwarg)

        # should not allow predict since KNN imputer is a transformer
        with self.assertRaises(AttributeError):
            imputer.predict(datamodule.splits["data"]["train"])
        with self.assertRaises(AttributeError):
            imputer.predict_proba(datamodule.splits["data"]["train"])

        # make sure this runs no errors
        imputer.transform(datamodule.splits["data"]["train"])

    @patch("autopopulus.data.dataset_classes.train_test_split")
    # _fit_and_score calls _score in _validation
    @patch.object(_search, "_fit_and_score", wraps=_search._fit_and_score)
    def test_predictor(self, mock_score, mock_split):
        mock_split.side_effect = [
            (splits["train"] + splits["val"], splits["test"]),
            (splits["train"], splits["val"]),
        ]
        data_settings = {
            # nomissing here so predictions don't complain about nan
            "dataset_loader": get_dataset_loader(X["nomissing"], y),
            "seed": seed,
            "test_size": 0.5,
            "val_size": 0.5,
            "batch_size": 2,
            "scale": True,
        }
        datamodule = CommonDataModule(**data_settings)
        datamodule.setup("fit")

        imputer = TunableEstimator(
            LogisticRegression(),
            PREDICTOR_MODEL_METADATA["lr"]["tune_kwargs"],
            score_fn=f1_score,
        )

        imputer.fit(
            datamodule.splits["data"],
            # this needs the y's
            {split: y.iloc[splits[split]] for split in ["train", "val", "test"]},
        )
        # make sure the test set is correctly the val set
        for call_args in mock_score.call_args_list:
            true_call = call(
                ANY,
                X["nomissing"].iloc[splits["train"] + splits["val"]],
                y.iloc[splits["train"] + splits["val"]],
                train=splits["train"],
                test=splits["val"],
                parameters=ANY,
                split_progress=ANY,
                candidate_progress=ANY,
            )
            for arg, true_arg in zip(call_args[0], true_call.args):
                if isinstance(arg, pd.DataFrame):
                    pd.testing.assert_frame_equal(arg, true_arg)
                elif isinstance(arg, pd.Series):
                    pd.testing.assert_series_equal(arg, true_arg)
                else:
                    self.assertEqual(arg, true_arg)
                    # np.testing.assert_allclose(arg, true_arg)
            for kwarg, true_kwarg in zip(call_args[1], true_call.kwargs):
                self.assertEqual(kwarg, true_kwarg)

        # should not transform since KNN imputer is a predictor
        with self.assertRaises(AttributeError):
            imputer.transform(datamodule.splits["data"]["train"])

        # make sure these run no error
        imputer.predict(datamodule.splits["data"]["train"])
        imputer.predict_proba(datamodule.splits["data"]["train"])
