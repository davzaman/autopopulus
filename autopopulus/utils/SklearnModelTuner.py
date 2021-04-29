from sklearn.pipeline import Pipeline
from typing import List, Dict, Any, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost.sklearn import XGBClassifier

from models.dnn import DNNClassifier


class SklearnModelTuner:
    def __init__(
        self,
        pipeline: Pipeline,
        parameters: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        eval_metric: Callable[[pd.Series, pd.Series], float],
    ) -> None:
        self.pipeline = pipeline
        self.parameters = parameters
        additional_fit_params = {}
        # Assumes last step in pipeline is model
        if isinstance(pipeline[-1], XGBClassifier):
            # Get rid of warning when using xgboost about changing default eval metric
            additional_fit_params["XGB__eval_metric"] = "logloss"
            # XGB doesn't work with pandas dataframes
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(X_valid, pd.DataFrame):
                X_valid = X_valid.values
        if isinstance(pipeline[-1], DNNClassifier):
            additional_fit_params["DNN__X_val"] = X_valid
            additional_fit_params["DNN__y_val"] = y_valid
        self.predictions = [
            pipeline.set_params(**params)
            .fit(X_train, y_train, **additional_fit_params)
            .predict_proba(X_valid)[:, 1]
            for params in tqdm(parameters)
        ]
        self.performance = [
            eval_metric(y_valid, prediction) for prediction in self.predictions
        ]
        self.best_params = self.parameters[np.argmax(self.performance)]
        self.best_model = pipeline.set_params(**self.best_params).fit(
            X_train, y_train, **additional_fit_params
        )
        self.best_performance = np.max(self.performance)
