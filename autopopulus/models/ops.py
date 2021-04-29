from typing import List, Dict, Any
from argparse import Namespace
import numpy as np

#### sklearn ####
from sklearn.pipeline import Pipeline

# metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#### Local ####
from models.dnn import DNNClassifier


def create_models(args: Namespace, input_size: int) -> List[Dict[str, Any]]:
    """Create models to be trained and evaluated.

    Returns the models as a dictionary:
        - name
        - sklearn pipeline
        - list of possible parameters to be used by SklearnModelTuner
    """
    predictors = args.predictors
    models = []

    #### Logistic Regression ####
    if "lr" in predictors:
        lrmodel = LogisticRegression(
            random_state=args.seed,
            solver="lbfgs",
            max_iter=5000,  # convergence warning
            class_weight="balanced",
        )

        lr = {
            "name": "logistic_regression",
            "pipeline": Pipeline([("LR", lrmodel)]),
            "parameters": [
                {"LR__penalty": penalty, "LR__C": c, "LR__solver": "liblinear"}
                # for penalty in ['l1', 'l2']
                for penalty in ["l2"]
                for c in np.logspace(4, 1, 20)
            ],
        }
        models.append(lr)

    #### Random Forest ####
    if "rf" in predictors:
        rfmodel = RandomForestClassifier(
            class_weight="balanced", random_state=args.seed
        )

        rf = {
            "name": "random_forest",
            "pipeline": Pipeline([("RF", rfmodel)]),
            "parameters": [
                {
                    "RF__n_estimators": n_estimators,
                    # 'RF__max_features': max_features,
                    "RF__max_depth": max_depth,
                }
                for n_estimators in range(5, 35, 5)
                # for max_features in range()
                for max_depth in range(3, 11)
            ],
        }
        models.append(rf)

    #### XGBoost ####
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
    if "xgb" in predictors:
        xgbmodel = XGBClassifier(
            scale_pos_weight=72, random_state=args.seed, use_label_encoder=False
        )

        xgb = {
            "name": "xgboost",
            "pipeline": Pipeline([("XGB", xgbmodel)]),
            "parameters": [
                {
                    "XGB__n_estimators": n_estimators,
                    # 'RF__max_features': max_features,
                    "XGB__max_depth": max_depth,
                    "XGB__learning_rate": learning_rate,
                }
                for n_estimators in range(5, 35, 5)
                # for max_features in range()
                for max_depth in range(3, 11)
                for learning_rate in [1e-5, 1e-3, 1e-1]
            ],
        }
        models.append(xgb)

    #### Deep Neural Network ####
    # use underlying pytorch model (False for use_keras)
    if "dnn" in predictors:
        dnnmodel = DNNClassifier(
            False,
            input_size,
            20,
            args.seed,
            args.batch_size,
            args.num_gpus * 4,
        )
        dnn = {
            "name": "deep_nn",
            "pipeline": Pipeline([("DNN", dnnmodel)]),
            "parameters": [
                {"DNN__lr": lr, "DNN__l2_penalty": l2_penalty, "DNN__dropout": dropout}
                for lr in [1e-5, 1e-3, 1e-1]
                for l2_penalty in [1e-4, 1e-1, 0]
                for dropout in [0, 0.2, 0.5]
            ],
        }
        models.append(dnn)

    return models


def get_performance(
    args: Namespace,
    y_valid: np.ndarray,
    predictions: np.ndarray,
    predictions_proba: np.ndarray,
) -> Dict[str, float]:
    """Prints metrics, returns a dictionary of performance metrics.

    Just calls sklearn metric functions. Refer to sklearn documentation.

    Note that sklearn is expecting "array-like" inputs, we are approximating
    with ndarray. There is no type-hint for array-like
    (ref: stackoverflow.com/questions/35673895)
    """
    conf_matrix = confusion_matrix(y_true=y_valid.astype(int), y_pred=predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    performance = {
        "F1-score": f1_score(
            y_true=y_valid.astype(int), y_pred=predictions, average="macro", labels=[1]
        ),
        "Recall-score": recall_score(
            y_true=y_valid.astype(int), y_pred=predictions, average="macro", labels=[1]
        ),
        "Precision-score": precision_score(
            y_true=y_valid.astype(int), y_pred=predictions, average="macro", labels=[1]
        ),
        "ROC-AUC": roc_auc_score(
            y_valid.astype(int), predictions_proba, average="macro"
        ),
        "PR-AUC": average_precision_score(
            y_valid.astype(int), predictions_proba, average="macro"
        ),
        "Brier-score": brier_score_loss(y_valid.astype(int), predictions_proba),
        # cannot log a whole confusion matrix as a metric in MLFlow, instead
        # Log the individual components of the CM (assuming binary labels)
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

    # print redundant with TP/etc information but it's formatted better
    if args.verbose:
        print("CM: ", conf_matrix)
    return performance
