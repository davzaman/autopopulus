from functools import reduce
import inspect
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pandas import DataFrame, MultiIndex, Series, concat
from numpy import ndarray, logspace, mean
from numpy.random import seed, randint
from tensorboardX import SummaryWriter
from pytorch_lightning.utilities import rank_zero_info

#### sklearn ####
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import resample

# metrics
from sklearn.metrics import (
    f1_score,
    make_scorer,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.contrib.vector_classifiers._continuous_interval_tree import (
    ContinuousIntervalTree,
)
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor


#### Local ####
from autopopulus.data.constants import PATIENT_ID
from autopopulus.data.types import DataTypeTimeDim
from autopopulus.utils.cli_arg_utils import YAMLStringListToList
from autopopulus.utils.log_utils import add_scalars, get_logger
from autopopulus.models.evaluation import (
    bootstrap_confidence_interval,
    confidence_interval,
    shapiro_wilk_test,
)


PREDICTOR_MODEL_METADATA = {
    "lr": {
        "cls": LogisticRegression,
        "full_name": "logistic_regression",
        "pipeline_name": "LR",
        "model_kwargs": {
            "solver": "lbfgs",
            "max_iter": 5000,  # convergence warning
            "class_weight": "balanced",
        },
        "tune_kwargs": {"penalty": ["l2"], "C": logspace(4, 1, 20, base=1.0)},
        "has_seed": True,
        "has_n_jobs": True,
        "can_handle_longitudinal": False,
    },
    "rf": {
        "cls": RandomForestClassifier,
        "full_name": "random_forest",
        "pipeline_name": "RF",
        "model_kwargs": {"class_weight": "balanced"},
        "tune_kwargs": {"n_estimators": range(5, 35, 5), "max_depth": range(3, 11)},
        "has_seed": True,
        "has_n_jobs": True,
        "can_handle_longitudinal": False,
    },
    "xgb": {
        "cls": XGBClassifier,
        "full_name": "xgboost",
        "pipeline_name": "XGB",
        "model_kwargs": {
            "scale_pos_weight": 72,
            "use_label_encoder": False,
            "eval_metric": "logloss",  # Get rid of warning
            "use_label_encoder": False,  # get rid of warning
        },
        "tune_kwargs": {
            "n_estimators": range(5, 35, 5),
            "max_depth": range(3, 11),
            "learning_rate": [1e-5, 1e-3, 1e-1],
        },
        "has_seed": True,
        "has_n_jobs": True,
        "can_handle_longitudinal": False,
    },
    "shapelettsc": {
        "cls": ShapeletTransformClassifier,
        "full_name": "shapelettsc",
        "pipeline_name": "ShapeletTSC",
        "model_kwargs": {},
        "tune_kwargs": {"estimator": [RotationForest(), ContinuousIntervalTree()]},
        "has_seed": True,
        "has_n_jobs": True,
        "can_handle_longitudinal": True,
    },
    "knntsc": {
        "cls": KNeighborsTimeSeriesClassifier,
        "full_name": "knntsc",
        "pipeline_name": "KNNTSC",
        "model_kwargs": {},
        "tune_kwargs": {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]},
        "has_seed": False,
        "has_n_jobs": True,
        "can_handle_longitudinal": True,
    },
}


class Predictor(TransformerMixin):
    """
    Predictor class which will run prediction models (after tuning them) and evaluate them on the data.
    If the data is longitudinal and a longitudinal model was specified:
        - Data will first be padded
    If the data is longitudinal but a static model was specified:
        - We use tsfresh to generate static features based on the longitudinal ones.

    """

    def __init__(
        self,
        seed: int,
        predictors: List[str],
        num_bootstraps: int,
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        logdir: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.seed = seed
        self.num_bootstraps = num_bootstraps
        self.data_type_time_dim = data_type_time_dim
        self.logdir = logdir
        self.verbose = verbose
        self.model_names = predictors
        self.model = self.build_models()

    def fit(self, X: Dict[str, DataFrame], y: Dict[str, DataFrame]):
        bootstrap_metrics = {}
        # Need to concat train and val for hold-out validation with sklearn
        X_tune = concat([X["train"], X["val"]], axis=0)
        y_tune = concat([y["train"], y["val"]], axis=0)
        X_tune, y_tune = self._align_Xy(X_tune, y_tune)

        for modeln, model in zip(self.model_names, self.model):
            log = get_logger(self.logdir, modeln)
            # Create Pipeline steps
            steps = []
            if self.data_type_time_dim.is_longitudinal():
                if PREDICTOR_MODEL_METADATA[modeln]["can_handle_longitudinal"]:
                    steps.append(("pad", PaddingTransformer()))
                else:
                    # To be able to use static predictors with longitudinal data. Requires data is passed as param in skmodeltuner.
                    # X_tune = self._extract_static_features_from_longitudinal(
                    # X_tune, y_tune
                    # )
                    X_tune = TSFreshFeatureExtractor(
                        default_fc_parameters="efficient", show_warnings=False
                    ).fit_transform(X_tune)

            pipeline_name = PREDICTOR_MODEL_METADATA[modeln]["pipeline_name"]
            steps.append((pipeline_name, model))

            # Set GridSearch with a hold-out validation instead of CV (via predefinedsplit)
            # indicate indices: -1 @ indices for train, 0 for evaluation
            holdout_validation_split = PredefinedSplit(
                test_fold=[-1] * len(X["train"]) + [0] * len(X["val"])
            )
            cv = GridSearchCV(
                Pipeline(steps),
                self._get_model_param_grid(modeln),
                make_scorer(average_precision_score),
                cv=holdout_validation_split,
            )

            # Create N seeds using the original seed for each bootstrap
            seed(self.seed)
            bootstrap_seeds = randint(0, 10000, self.num_bootstraps)
            for b in tqdm(range(self.num_bootstraps)):
                X_boot, y_boot = resample(
                    X_tune, y_tune, stratify=y_tune, random_state=bootstrap_seeds[b]
                )
                # Run GridSearch on concatenated train+val
                # steps["tsfresh-features"].set_timeseries_container(X_boot)
                cv.fit(X_boot, y_boot)
                # Evaluate on best model
                # self.evaluate(y["test"], cv.predict(X["test"]), cv.predict_proba(X["test"])[:,1])
                metric_results = self.evaluate(
                    y["test"], cv.predict(X["test"]), cv.predict_proba(X["test"])
                )
                # plot across all bootstraps
                add_scalars(log, metric_results, b, prefix="predict")
                # save performance across bootstrap samples to form CI
                bootstrap_metrics.append(metric_results)
            if log:
                log.close()
        # Roll up results for logging, assuming they all have the same metric keys
        bootstrap_metrics_rolled = reduce(
            lambda metrics1, metrics2: {
                metricn: metrics1[metricn] + metrics2[metricn]
                for metricn in metrics1.keys()
            },
            bootstrap_metrics,
        )
        self._log_performance_statistics(
            bootstrap_metrics_rolled, confidence_interval, log
        )
        return self

    def transform(self, X: Union[ndarray, DataFrame]) -> ndarray:
        """Applies trained model to given data X."""
        raise NotImplementedError  # For now we're jsut doing evaluation, so there's no single saved trained model at the end
        return self.model.transform(X)

    def evaluate(
        self, true: ndarray, pred: ndarray, pred_proba: ndarray
    ) -> Dict[str, float]:
        """Prints metrics, returns a dictionary of performance metrics.

        Just calls sklearn metric functions. Refer to sklearn documentation.

        Note that sklearn is expecting "array-like" inputs, we are approximating
        with ndarray. There is no type-hint for array-like
        (ref: stackoverflow.com/questions/35673895)
        """
        conf_matrix = confusion_matrix(y_true=true.astype(int), y_pred=pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        true = true.astype(int)
        performance = {
            "F1-score": f1_score(
                y_true=true,
                y_pred=pred,
                average="macro",
                labels=[1],
            ),
            "Recall-score": recall_score(
                y_true=true,
                y_pred=pred,
                average="macro",
                labels=[1],
            ),
            "Precision-score": precision_score(
                y_true=true,
                y_pred=pred,
                average="macro",
                labels=[1],
            ),
            "ROC-AUC": roc_auc_score(true, pred_proba, average="macro"),
            "PR-AUC": average_precision_score(true, pred_proba, average="macro"),
            "Brier-score": brier_score_loss(true, pred_proba),
            # cannot log a whole confusion matrix as a metric in MLFlow, instead
            # Log the individual components of the CM (assuming binary labels)
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }

        if self.verbose:
            # print redundant with TP/etc information but it's formatted better
            rank_zero_info(f"CM: {conf_matrix}")
            rank_zero_info(performance)
        return performance

    def build_models(self, n_jobs: int = -1) -> List[BaseEstimator]:
        """Create models to be trained and evaluated.
        Returns the newly initialized but untrained models.
        """
        models = []
        for predictor_name in self.model_names:
            # Get metadata
            model_metadata = PREDICTOR_MODEL_METADATA[predictor_name]
            # Get kwargs from metadata
            model_kwargs = model_metadata["model_kwargs"]
            # set seed and njobs if it requires it
            if model_metadata["has_seed"]:
                model_kwargs["random_state"] = self.seed
            if model_metadata["has_n_jobs"]:
                model_kwargs["n_jobs"] = n_jobs
            # Get model class and initialize
            model_cls = model_metadata["cls"]
            models.append(model_cls(**model_kwargs))
        return models

    ###################
    #     HELPERS     #
    ###################
    def _get_model_param_grid(self, modeln: str) -> Dict[str, Any]:
        """Given the grid for the tune kwargs for the supported model, create a dictionary to use with sklearn GridSearch for tuning."""
        # Set grid for model
        model_pipe_name = PREDICTOR_MODEL_METADATA[modeln]["pipeline_name"]
        tune_kwargs = PREDICTOR_MODEL_METADATA[modeln]["tune_kwargs"]
        return {
            f"{model_pipe_name}__{param_name}": param_values
            for param_name, param_values in tune_kwargs.items()
        }

    @staticmethod
    def _align_Xy(X: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        """Limit labels to index of X incase debugging/not running on whole dataset.
        Cannot be pipeline step because of the .fit() validation step."""
        if X.shape[0] != y.shape[0]:
            index = X.index
            if isinstance(X.index, MultiIndex):
                index = index.get_level_values(PATIENT_ID)
            return (X, y[index])
        return (X, y)

    # def _extract_static_features_from_longitudinal(
    #     self, X: DataFrame, y: Series
    # ) -> DataFrame:
    #     """Uses tsfresh to extract static features from longitudinal data."""
    #     if PATIENT_ID not in X.columns:
    #         X = X.reset_index()
    #     return select_features(
    #         extract_features(
    #             timeseries_container=X,
    #             column_id=PATIENT_ID,
    #             column_sort="time",
    #             n_jobs=4,
    #         ),
    #         y,
    #     )

    def _log_performance_statistics(
        self,
        bootstrap_performance: Dict[str, List[float]],
        log: Optional[SummaryWriter] = None,
    ) -> None:
        """Compute Mean, CI, normality test across bootstrap samples"""
        mean_performance = {
            f"{k}-mean": mean(v) for k, v in bootstrap_performance.items()
        }
        normality = {
            f"{k}-isnormal": shapiro_wilk_test(v)
            for k, v in bootstrap_performance.items()
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

    ###################
    #    INTERFACE    #
    ###################
    def add_prediction_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--predictors",
            default="lr",
            action=YAMLStringListToList(choices=PREDICTOR_MODEL_METADATA.keys()),
            help="Which ml classifiers to use for downstream static and longitudinal prediction",
        )
        p.add_argument(
            "--num-bootstraps",
            type=int,
            # required=True,
            default=99,
            help="We do the predictive steps num_boostraps number of times to create a confidence interval on the metrics on the prediction performance.",
        )
        p.add_argument(
            "--confidence-level",
            type=float,
            # required=True,
            default=-1.95,
            help="For the bootstrap confidence intervals, what is the confidence level we want to use.",
        )
        return p

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs
    ) -> "Predictor":
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)
