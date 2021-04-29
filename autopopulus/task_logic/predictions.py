"""
All code involved with the prediction part of the pipeline.
i.e. taking the imputed data and running it through a logistic regression and random forest.
"""
from typing import List, Dict, Tuple, Any, Optional
from argparse import Namespace

import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

## Tune ##
from ray.tune import track

#### Sk-learn ####
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.utils import resample

#### Scipy ####
import scipy.stats as st

#### Local Module ####
from models.ops import create_models, get_performance
from utils.SklearnModelTuner import SklearnModelTuner
from utils.log_utils import add_scalars, get_logger


def run_predictions(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    args: Namespace,
    feature_names: Optional[List[str]] = None,
) -> None:
    models = create_models(args, X_train.shape[1])
    for modelp in models:
        log = get_logger(args, modelp["name"])
        _run_predictions(
            args,
            modelp,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args.num_bootstraps,
            args.confidence_level,
            log=log,
        )
        if log:
            log.close()


def _run_predictions_no_bootstrap(
    args: Namespace,
    modelp: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, bool]]:
    X_train_boot, y_train_boot = X_train, y_train

    model = SklearnModelTuner(
        pipeline=modelp["pipeline"],
        parameters=modelp["parameters"],
        X_train=X_train_boot,
        y_train=y_train_boot.astype(int),
        X_valid=X_val,
        y_valid=y_val.astype(int),
        # eval_metric=roc_auc_score,
        eval_metric=average_precision_score,
    )
    X_evaluate = X_val if args.val else X_test
    y_evaluate = y_val if args.val else y_test
    predictions = model.best_model.predict(X_evaluate)
    predictions_proba = model.best_model.predict_proba(X_evaluate)[:, 1]
    performance = get_performance(args, y_evaluate, predictions, predictions_proba)
    if args.verbose:
        print(performance)

    return (performance, {}, {})


def _run_predictions(
    args: Namespace,
    modelp: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    num_bootstraps: int,
    confidence_level: float,
    log: Optional[SummaryWriter] = None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, bool]]:
    # Bootstrap sample num_bootstraps times
    bootstrap_performance = {}
    np.random.seed(args.seed)
    bootstrap_seeds = np.random.randint(0, 10000, num_bootstraps)
    for b in range(num_bootstraps):
        # Resample training data with replacement
        X_train_boot, y_train_boot = resample(
            X_train, y_train, stratify=y_train, random_state=bootstrap_seeds[b]
        )

        model = SklearnModelTuner(
            pipeline=modelp["pipeline"],
            parameters=modelp["parameters"],
            X_train=X_train_boot,
            y_train=y_train_boot.astype(int),
            X_valid=X_val,
            y_valid=y_val.astype(int),
            # eval_metric=roc_auc_score,
            eval_metric=average_precision_score,
        )

        X_evaluate = X_val if not args.runtest else X_test
        y_evaluate = y_val if not args.runtest else y_test
        # If using xgb you need to pass in np array not dataframe or else the feature names will not match
        # enforce np
        X_evaluate = (
            X_evaluate.values if isinstance(X_evaluate, pd.DataFrame) else X_evaluate
        )
        predictions = model.best_model.predict(X_evaluate)
        predictions_proba = model.best_model.predict_proba(X_evaluate)[:, 1]
        performance = get_performance(args, y_evaluate, predictions, predictions_proba)
        if args.verbose:
            print(performance)

        # save performance across bootstrap samples to form CI
        for metric_name in performance.keys():
            # if exists, append performance, else start a list
            if metric_name in bootstrap_performance:
                bootstrap_performance[metric_name].append(performance[metric_name])
            else:
                bootstrap_performance[metric_name] = [performance[metric_name]]

        # plot across all bootstraps
        add_scalars(log, performance, b, prefix="predict")
    _log_performance_statistics(args, bootstrap_performance, confidence_interval, log)


def _log_performance_statistics(
    args: Namespace,
    bootstrap_performance: Dict[str, List[float]],
    confidence_interval: float,
    log: Optional[SummaryWriter] = None,
) -> None:
    """Compute Mean, CI, normality test across bootstrap samples"""
    mean_performance = {
        f"{k}-mean": np.mean(v) for k, v in bootstrap_performance.items()
    }
    normality = {
        f"{k}-isnormal": shapiro_wilk_test(v) for k, v in bootstrap_performance.items()
    }
    ci_lower, ci_upper = {}, {}
    for k, v in bootstrap_performance.items():
        lower, upper = bootstrap_confidence_interval(v)
        ci_lower[f"{k}-lower"] = lower
        ci_upper[f"{k}-upper"] = upper

    #### LOGGING ####
    prefix = "predict-aggregate"
    add_scalars(log, mean_performance, prefix=prefix)
    add_scalars(log, ci_lower, prefix=prefix)
    add_scalars(log, ci_upper, prefix=prefix)
    add_scalars(log, normality, prefix=prefix)
    if hasattr(args, "tune"):
        track.log(**mean_performance)


def bootstrap_confidence_interval(
    metric: List[float], confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Returns confidence interval at given confidence level for statistical
    distributions established with bootstrap sampling.
    """
    alpha = 1 - confidence_level
    lower = alpha / 2 * 100
    upper = (alpha / 2 + confidence_level) * 100
    return np.percentile(metric, lower), np.percentile(metric, upper)


def confidence_interval(
    metric: List[float], confidence_level: float = 0.95
) -> float:  # ) -> Tuple[float, float]:
    """Returns confidence interval at given confidence level for data on metric.
    Assumes normality and will produce symmetric bounds.
    Note if sem is 0 st.t.interval will throw a runtime error.
    Ref: https://stackoverflow.com/questions/15033511
    """
    """
    # for # bootstrap samples < 100 it is better to look up in t dist table
    # if we want to use z use st.norm.interval instead.
    return st.t.interval(confidence_level, len(metric) - 1,
                         loc=np.mean(metric), scale=st.sem(metric))
    """
    # mean = np.mean(metric)
    half_interval = st.sem(metric) * st.t.ppf(
        (1 + confidence_level) / 2.0, len(metric) - 1
    )
    return half_interval
    # return (mean - half_interval, mean + half_interval)


def shapiro_wilk_test(metric: List[float], confidence_level: float = 0.95) -> bool:
    """Outputs W,p. We want p > (alpha = 1 - CI = .05 usually)
    Ref: https://machinelearningmastery.com/
        a-gentle-introduction-to-normality-tests-in-python/
    p > alpha: Sample looks Gaussian (fail to reject H0)
         else: Sample does not look Gaussian (reject H0)
    """
    return st.shapiro(metric)[1] > (1 - confidence_level)
