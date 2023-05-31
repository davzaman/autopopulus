from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re

from timeit import default_timer as timer
from lightning_utilities import apply_to_collection
from tqdm import tqdm
from numpy import isnan, ndarray
from numpy.random import default_rng
from pandas import DataFrame, concat

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection._search import _check_refit
from sklearn.multiclass import available_if
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics._scorer import _cached_call, _BaseScorer

from autopopulus.utils.utils import rank_zero_print
from data.types import DataT


def _estimator_has(attr):
    # simplified version of sklearn fn that works for my wrapped clas
    def check(self):
        _check_refit(self.cv, attr)
        if hasattr(self.cv, "best_estimator_"):
            # raise an AttributeError if `attr` does not exist
            getattr(self.cv.best_estimator_, attr)
            return True
        # raise an AttributeError if `attr` does not exist
        getattr(self.cv.estimator, attr)
        return True

    return check


class TransformScorer(_BaseScorer):
    """
    Modified PredictScorer, calls transform instead of predict.
    Expects score function to take in order: [preds, target, missing indicators]
    This is only meant to be combined with my own local metrics.
    """

    def __init__(
        self,
        score_func: Callable[..., Any],
        higher_is_better: bool,
        missingonly: bool = False,
        score_func_expects_pred_first: bool = True,
        **kwargs,
    ) -> None:
        sign = 1 if higher_is_better else -1
        self.missingonly = missingonly
        self.score_func_expects_pred_first = score_func_expects_pred_first
        super().__init__(score_func, sign, kwargs)

    # Adjusted from PredictScorer to evaluate impute/transform accuracy
    def _score(self, method_caller, estimator, X, X_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.transform.

        X_true : {array-like, sparse matrix}
            Golden standard for transformed data.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        X_pred = method_caller(estimator, "transform", X)
        # numpy method works on both np and pd
        where_data_are_missing = isnan(X) if self.missingonly else None
        arg_order = (
            (X_pred, X_true, where_data_are_missing)
            if self.score_func_expects_pred_first
            else (X_true, X_pred, where_data_are_missing)
        )
        if sample_weight is not None:
            return self._sign * self._score_func(
                *arg_order, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(*arg_order, **self._kwargs)


class TunableEstimator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        estimator_params: Dict[str, Any],
        # BaseScorer for predict estimator, transform scorer for impute estimator
        score_fn: Union[_BaseScorer, TransformScorer],
    ) -> None:
        # make_scorer expects a y along with X which fails for imputation
        self.scorer = score_fn
        self.estimator = estimator
        # just run on defaults if no score_func (ground truth has nans)
        if isinstance(self.scorer, _BaseScorer) and self.scorer._score_func is None:
            self.estimator_params = {}
        else:
            self.estimator_params = estimator_params

    def fit(self, X: Dict[str, DataFrame], y: Dict[str, DataFrame]):
        # Need to concat train and val for hold-out validation with sklearn
        X_tune = concat([X["train"], X["val"]], axis=0)
        y_tune = concat([y["train"], y["val"]], axis=0)

        # Set GridSearch with a hold-out validation instead of CV (via predefinedsplit)
        # indicate indices: -1 @ indices for train, 0 for evaluation
        # This should work for both static and longitudinal
        holdout_validation_split = PredefinedSplit(
            test_fold=[-1] * X["train"].groupby(level=0).ngroups
            + [0] * X["val"].groupby(level=0).ngroups
        )
        self.cv = GridSearchCV(
            self.estimator,
            self.estimator_params,
            scoring=self.scorer,
            cv=holdout_validation_split,
            n_jobs=-1,  # because im using a lamba fn universal_metric this will be unpickelable and the parralelization requires serialization which requires pickling
        )
        rank_zero_print(f"Starting fit of {self.estimator}")
        start = timer()
        args = [X_tune, y_tune]
        try:
            self.cv.fit(*args)
        except ValueError:  # lightgbm has a weird issue with column names.
            # https://github.com/autogluon/autogluon/issues/399#issuecomment-623326629
            args = apply_to_collection(
                args,
                DataFrame,
                lambda df: df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x)),
            )
            self.cv.fit(*args)
        rank_zero_print(f"Fit took {timer() - start} seconds.")

    @available_if(_estimator_has("transform"))
    def transform(self, X: DataFrame) -> ndarray:
        return self.cv.transform(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X: DataFrame) -> ndarray:
        return self.cv.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X: DataFrame) -> ndarray:
        return self.cv.predict_proba(X)


# TODO: WIP. ideally i'd have a BaselineImputer model that is wrapped by this class
@DeprecationWarning
class BootstrapEstimator(BaseEstimator):
    kosher_bootstrap_methods: Set[Optional[str]] = ("out_of_bag", "test_only", None)

    def __init__(
        self,
        estimator: BaseEstimator,
        bootstrap_method: Optional[str] = None,
        num_bootstrap_samples: int = 0,
    ) -> None:
        """Expects estimator to have a `fit()` and `evaluate()` method."""
        assert (
            num_bootstrap_samples > 0 or bootstrap_method is None
        ), "Need to specify num_bootstrap_samples if you'd like to bootstrap evaluate."
        assert (
            bootstrap_method in self.kosher_bootstrap_methods,
            f"The bootstrap method can only be one of the following: {self.kosher_bootstrap_methods}",
        )
        assert hasattr(
            estimator, "evaluate"
        ), "Estimator should have an `evaluate()` method that returns Dict[str, float], a mapping of metric name to value."
        self.estimator = estimator
        self.bootstrap_method = bootstrap_method
        self.num_bootstrap_samples = num_bootstrap_samples
        self.is_fitted = False

    def fit(self, X: Dict[str, DataFrame], y: Dict[str, DataFrame]):
        self.data = (X, y)
        if self.bootstrap_method == "out_of_bag":  # Do nothing.
            return self
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def test(self) -> Dict[str, float]:
        if self.num_bootstrap_samples:
            if self.bootstrap_method == "out_of_bag":
                # don't require fit
                return self.bootstrap_out_of_bag()
            else:  # test-only
                check_is_fitted(self)
                return self.bootstrap_test()
        else:
            X, y = self.data
            return self.estimator.evaluate(X["test"], y["test"])

    def bootstrap_test(self) -> Dict[str, List[float]]:
        # metric name -> list of values per bootstrap
        bootstrap_metrics: Dict[str, List[float]] = {}
        X_test, y_test = self.data[0]["test"], self.data[1]["test"]

        # Create N seeds using the original seed for each bootstrap
        gen = default_rng(self.seed)
        bootstrap_seeds = gen.integers(0, 10000, self.num_bootstrap_samples)
        for b in tqdm(range(self.num_bootstrap_samples)):
            X_boot_test, y_boot_test = resample(
                X_test, y_test, stratify=y_test, random_state=bootstrap_seeds[b]
            )

            metric_vals = self.estimator.evaluate(X_boot_test, y_boot_test)
            for metric_name, metric_val in metric_vals.items():
                # create/get key with metric name, and either create an empty list with 1 new metric_val item, or append to an existing list.
                bootstrap_metrics[metric_name] = bootstrap_metrics.setdefault(
                    metric_name, []
                ) + [metric_val]
        return bootstrap_metrics

    @DeprecationWarning  # TODO: wip
    def bootstrap_out_of_bag(self) -> Dict[str, List[float]]:
        return
        # Create N seeds using the original seed for each bootstrap
        gen = default_rng(self.seed)
        bootstrap_seeds = gen.integers(0, 10000, self.num_bootstrap_samples)
        for b in tqdm(range(self.num_bootstrap_samples)):
            pass
            # X_boot, y_boot = resample(
            #     X_train, y_train, stratify=y_train, random_state=bootstrap_seeds[b]
            # )

    # for check_is_fitted
    def __sklearn_is_fitted(self) -> bool:
        return self.is_fitted
