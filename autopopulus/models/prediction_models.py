from timeit import default_timer as timer
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pandas import DataFrame, MultiIndex, Series, concat
from numpy import ndarray, logspace, mean
from numpy.random import Generator, default_rng
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import rank_zero_info
import re
from lightgbm.basic import LightGBMError

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
from lightgbm import LGBMClassifier

from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.sklearn import RotationForest, ContinuousIntervalTree
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.datatypes._panel._convert import from_multi_index_to_nested
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor


#### Local ####
from autopopulus.data.constants import PATIENT_ID
from autopopulus.data.types import DataTypeTimeDim
from autopopulus.utils.cli_arg_utils import YAMLStringListToList
from autopopulus.utils.log_utils import BasicLogger
from autopopulus.models.evaluation import (
    bootstrap_confidence_interval,
    shapiro_wilk_test,
)
from autopopulus.utils.utils import CLIInitialized, rank_zero_print


PREDICTOR_MODEL_METADATA = {
    "lr": {
        "cls": LogisticRegression,
        "full_name": "logistic_regression",
        "pipeline_name": "LR",
        "model_kwargs": {
            "solver": "sag",
            # "max_iter": 5000,  # convergence warning
            "class_weight": "balanced",
        },
        "tune_kwargs": {"penalty": ["l2"], "C": logspace(4, 1, 20, base=1.0)},
        "has_seed": True,
        "has_n_jobs": True,
        "longitudinal": False,
    },
    "rf": {
        "cls": RandomForestClassifier,
        "full_name": "random_forest",
        "pipeline_name": "RF",
        "model_kwargs": {"class_weight": "balanced"},
        "tune_kwargs": {"n_estimators": range(5, 35, 5), "max_depth": range(3, 11)},
        "has_seed": True,
        "has_n_jobs": True,
        "longitudinal": False,
    },
    "xgb": {
        "cls": XGBClassifier,
        "full_name": "xgboost",
        "pipeline_name": "XGB",
        "model_kwargs": {
            "scale_pos_weight": 72,
            "use_label_encoder": False,
            "eval_metric": "logloss",  # Get rid of warning
            # "use_label_encoder": False,  # get rid of warning
        },
        "tune_kwargs": {
            "n_estimators": range(5, 35, 5),
            "max_depth": range(3, 11),
            "learning_rate": [1e-5, 1e-3, 1e-1],
        },
        "has_seed": True,
        "has_n_jobs": True,
        "longitudinal": False,
    },
    "lgbm": {
        "cls": LGBMClassifier,
        "full_name": "lgbm",
        "pipeline_name": "LGBM",
        "model_kwargs": {
            "boosting_type": "goss",
        },
        "tune_kwargs": {
            "n_estimators": range(5, 35, 5),
            "max_depth": range(3, 11),
            "learning_rate": [1e-5, 1e-3, 1e-1],
        },
        "has_seed": True,
        "has_n_jobs": True,
        "longitudinal": False,
    },
    # Time series
    "shapelettsc": {
        "cls": ShapeletTransformClassifier,
        "full_name": "shapelettsc",
        "pipeline_name": "ShapeletTSC",
        "model_kwargs": {},
        "tune_kwargs": {"estimator": [RotationForest(), ContinuousIntervalTree()]},
        "has_seed": True,
        "has_n_jobs": True,
        "longitudinal": True,
    },
    "knntsc": {
        "cls": KNeighborsTimeSeriesClassifier,
        "full_name": "knntsc",
        "pipeline_name": "KNNTSC",
        "model_kwargs": {},
        "tune_kwargs": {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]},
        "has_seed": False,
        "has_n_jobs": True,
        "longitudinal": True,
    },
}


class Predictor(TransformerMixin, CLIInitialized):
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
        base_logger_context: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
        parent_run_hash: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.seed = seed
        self.num_bootstraps = num_bootstraps
        self.data_type_time_dim = data_type_time_dim
        self.base_logger_context = base_logger_context
        self.experiment_name = experiment_name
        self.parent_run_hash = parent_run_hash
        self.verbose = verbose
        self.model_names = predictors
        self.model = self.build_models()

    def fit(self, X: Dict[str, DataFrame], y: Dict[str, DataFrame]):
        bootstrap_metrics = []
        # Need to concat train and val for hold-out validation with sklearn
        X_train_val = concat([X["train"], X["val"]], axis=0)
        y_train_val = concat([y["train"], y["val"]], axis=0)

        # only do this once if we see the need since it's $$$
        self._set_X_transforms(X["train"], y["train"])

        pbar = tqdm(zip(self.model_names, self.model), total=len(self.model))
        for modeln, model in pbar:
            pbar.set_description(f"Processing {modeln}")
            # Reset incase changes are made in the following logic
            X_tune, y_tune = self._preproc_predictor_data(
                X_train_val, y_train_val, self.X_transforms[modeln]
            )
            X_eval, y_eval = self._preproc_predictor_data(
                X["test"], y["test"], self.X_transforms[modeln]
            )

            # We can't initialize earlier because we need predictive model in the logdir for tensorboard
            logger = BasicLogger(
                run_hash=self.parent_run_hash,
                experiment_name=self.experiment_name,
                base_context=self.base_logger_context,
                predictive_model=modeln,
            )

            # Set GridSearch with a hold-out validation instead of CV (via predefinedsplit)
            # indicate indices: -1 @ indices for train, 0 for evaluation
            # This should work for both static and longitudinal
            holdout_validation_split = PredefinedSplit(
                test_fold=[-1] * X["train"].groupby(level=0).ngroups
                + [0] * X["val"].groupby(level=0).ngroups
            )
            pipeline_name = PREDICTOR_MODEL_METADATA[modeln]["pipeline_name"]
            cv = GridSearchCV(
                Pipeline([(pipeline_name, model)]),
                self._get_model_param_grid(modeln),
                scoring=make_scorer(average_precision_score),
                cv=holdout_validation_split,
                n_jobs=-1,
            )

            # Create N seeds using the original seed for each bootstrap
            gen = default_rng(self.seed)
            bootstrap_seeds = gen.integers(0, 10000, self.num_bootstraps)
            for b in tqdm(range(self.num_bootstraps)):
                X_boot, y_boot = resample(
                    X_tune, y_tune, stratify=y_tune, random_state=bootstrap_seeds[b]
                )

                # Run GridSearch on concatenated train+val
                rank_zero_print(f"Starting fit of {model}")
                start = timer()
                try:
                    cv.fit(X_boot, y_boot)
                except ValueError:  # lightgbm has a weird issue with column names.
                    # https://github.com/autogluon/autogluon/issues/399#issuecomment-623326629
                    cv.fit(
                        X_boot.rename(
                            columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x)
                        ),
                        y_boot,
                    )
                rank_zero_print(f"Fit took {timer() - start} seconds.")
                # Evaluate on best model
                metric_results = self.evaluate(
                    y_eval, cv.predict(X_eval), cv.predict_proba(X_eval)[:, 1]
                )
                # plot across all bootstraps
                logger.add_scalars(
                    metric_results,
                    global_step=b,
                    context={"step": "predict", "predictive_model": modeln},
                    tb_name_format="{predictive_model}/{step}/{name}",
                )
                # save performance across bootstrap samples to form CI
                bootstrap_metrics.append(metric_results)

            # Roll up results for logging, assuming they all have the same metric keys
            # self._log_performance_statistics(bootstrap_metrics, logger, modeln)
            logger.close()
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
            rank_zero_print(f"CM: {conf_matrix}")
            rank_zero_print(performance)
        return performance

    def build_models(self, n_jobs: int = -1) -> List[BaseEstimator]:
        """Create models to be trained and evaluated.
        Returns the newly initialized but untrained models.
        """
        models = []
        for predictor_name in self.model_names:
            # Get metadata
            model_metadata = PREDICTOR_MODEL_METADATA[predictor_name]
            # Validate the model asked-for
            if not self.data_type_time_dim.is_longitudinal():
                assert (
                    model_metadata["longitudinal"] == False
                ), f"One of the predictors ({predictor_name}) is longitudinal-only, but the data was indicated to be static."
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

    def _set_X_transforms(self, X_train: DataFrame, y_train: Series) -> None:
        """
        Train the transformers just once.
        However, we might need multiple different transformers depending on the requested predictor models.
        Creates a map from modeln to the corresponding tranforms,
            and only creates/trains that transform once.
        """
        X_train, y_train = self._align_Xy(X_train, y_train)

        # Figure out which transforms we'll need
        transforms = {}
        self.X_transforms = {}
        for modeln in self.model_names:
            if self.data_type_time_dim.is_longitudinal():
                if PREDICTOR_MODEL_METADATA[modeln]["longitudinal"]:
                    transforms["long_on_long"] = None
                    self.X_transforms[modeln] = "long_on_long"
                else:  # use static predictors with longitudinal data.
                    transforms["long_on_static"] = None
                    self.X_transforms[modeln] = "long_on_static"
            else:
                transforms["static_on_static"] = None
                self.X_transforms[modeln] = "static_on_static"

        # Create and train that transform just once
        if "long_on_long" in transforms:
            pipe = Pipeline(
                [
                    (
                        "sktime-format",
                        FunctionTransformer(from_multi_index_to_nested),
                    ),
                    ("pad", PaddingTransformer()),
                ]
            )
            pipe.fit(X_train, y_train)
            transforms["long_on_long"] = pipe.transform
        elif "long_on_static" in transforms:
            pipe = Pipeline(
                [
                    (
                        "tsfresh_extractor",
                        TSFreshFeatureExtractor(
                            default_fc_parameters="efficient",
                            show_warnings=False,
                            n_jobs=-1,
                        ),
                    )
                ]
            )
            pipe.fit(X_train, y_train)
            transforms["long_on_static"] = pipe.transform
        else:
            transforms["static_on_static"] = None

        # map transform name to now (once-trained) transform function
        self.X_transforms = {
            modeln: transforms[transform_name]
            for modeln, transform_name in self.X_transforms.items()
        }

    def _preproc_predictor_data(
        self,
        X: DataFrame,
        y: Series,
        X_transform: Optional[Callable[[DataFrame], DataFrame]] = None,
    ) -> Tuple[DataFrame, Series]:
        if X_transform:
            X = X_transform(X)
        if self.data_type_time_dim.is_longitudinal():
            # y_tune and eval needs to be squashed to be per-patient
            y = y.groupby(level=0).first()

        X, y = self._align_Xy(X, y)
        return (X, y)

    def _log_performance_statistics(
        self,
        bootstrap_metrics: List[Dict[str, float]],
        logger: BasicLogger,
        modeln: str,
    ) -> None:
        """Compute Mean, CI, normality test across bootstrap samples"""
        bootstrap_performance: Dict[str, List[float]] = {
            metricn: [metrics[metricn] for metrics in bootstrap_metrics]
            for metricn in bootstrap_metrics[0]
        }
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
        for metrics in [mean_performance, ci_lower, ci_upper, normality]:
            logger.add_scalars(
                metrics,
                context={"step": "predict-aggregate", "predictive_model": modeln},
                tb_name_format="{predictive_model}/{step}/{name}",
            )

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
