from logging import info
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import Logger

#### Pytorch ####
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from sklearn.base import BaseEstimator, ClassifierMixin

from autopopulus.models.utils import ResetSeed

## Debugging/Running dnn.py separately ##
"""
import sys, os

sys.path.insert(
    1, os.path.join(sys.path[0], "/home/davina/Private/rapid_decline/autopopulus")
)
sys.path.insert(1, os.path.join(sys.path[0], "/home/davina/Private/rapid_decline/"))
"""
from autopopulus.data.utils import get_dataloader

METRICS_TF = [
    "accuracy",
    # "AUC",
    # "Precision",
    # "Recall",
    # tf.keras.metrics.AUC(curve="PR", name="PR_AUC"),
]


class DNNClassifier(ClassifierMixin, BaseEstimator):
    """Classifier compatible with sklearn, uses DNN to do BINARY classification.
    Implements, fit, predict_proba, and predict.
    Underlying DNN can be in keras or pytorch_lightning.
    """

    def __init__(
        self,
        use_keras: bool,
        input_dim: int,
        max_epochs: int,
        seed: int,
        batch_size: int,
        num_gpus: int,
        lr: float = 1e-3,
        l2_penalty: float = 1e-4,
        dropout: float = 0.5,
        logger: Optional[Logger] = None,
    ):
        self.input_dim = input_dim
        self.max_epochs = max_epochs
        # Hyperparams, can tune with scikitlearn
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.use_keras = use_keras
        self.seed = seed
        self.dnn = DNNLightning(self.lr, self.l2_penalty, self.dropout, input_dim, seed)
        callbacks = [EarlyStopping(monitor="DNN/val-loss", patience=5)]
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            deterministic=True,
            # gpus=self.num_gpus,
            # accelerator="ddp" if self.num_gpus > 1 else None,
            # profiler="simple",  # or "advanced" which is more granular
            checkpoint_callback=False,
            callbacks=callbacks,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ):
        if self.use_keras:
            self.dnn.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            torch.manual_seed(self.seed)
            train_loader = get_dataloader(
                X_train, y_train, self.batch_size, self.num_gpus, self.num_gpus
            )
            val_loader = (
                get_dataloader(
                    X_val, y_val, self.batch_size, self.num_gpus, self.num_gpus
                )
                if X_val is not None
                else None
            )
            self.trainer.fit(self.dnn, train_loader, val_dataloaders=val_loader)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int).squeeze()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """The returned estimates for all classes are ordered by the label of classes.

        Since it's binary classification it will be col0 = 0 label, col1= 1 label. This is meant to match sklearn's output for other classifiers.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        logit = self.dnn(torch.Tensor(X))
        positive_class_probability = torch.sigmoid(logit).detach().numpy()
        negative_class_probability = 1 - positive_class_probability
        return np.hstack((negative_class_probability, positive_class_probability))


class DNNLightning(pl.LightningModule):
    """Pytorch-based DNN model. Returns logits (numerical stability)."""

    def __init__(
        self,
        lr: float,
        l2_penalty: float,
        dropout: float,
        input_dim: int,
        seed: int,
        output_bias: Optional[int] = None,
    ):
        super().__init__()
        # self.loss = nn.BCELoss()
        # self.loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.metrics = {
            "accuracy": pl.metrics.Accuracy(),
            "precision": pl.metrics.Precision(),
            "recall": pl.metrics.Recall(),
            "rocauc": pl.metrics.functional.classification.auroc,
            "prauc": self._PRAUC,
        }

        self.layer_dims = [input_dim, 64, 64, 32, 16, 1]
        layers = []
        # -2: exclude the last layer (-1), and also account i,i+1 (-1)
        for i in range(len(self.layer_dims) - 2):
            layers += [
                nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]),
                nn.ReLU(inplace=True),
                ResetSeed(seed),
                nn.Dropout(self.dropout),
            ]

        final_fc = nn.Linear(self.layer_dims[-2], self.layer_dims[-1])
        if output_bias:
            final_fc.bias.data.fill_(output_bias)

        # layers += [final_fc, nn.Sigmoid()]  # BCE
        layers += [final_fc]  # BCE With logits loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch)
        self.shared_logging_step_end(outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch)
        self.shared_logging_step_end(outputs, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch)
        self.shared_logging_step_end(outputs, "test")
        return loss

    def shared_step(self, batch) -> Dict[str, float]:
        X, y_true = batch
        y_est = self(X)
        loss = self.loss(y_est, y_true.unsqueeze(1))
        return loss, {"loss": loss, "preds": y_est.detach(), "target": y_true.detach()}

    def shared_logging_step_end(self, outputs: Dict[str, float], step_type: str):
        """Log metrics + loss at end of step.
        Compatible with dp mode: https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#classification-metrics.
        """
        # Log loss
        self.info(
            f"DNN/{step_type}-loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Log all metrics
        logit = outputs["preds"]
        positive_class_probability = torch.sigmoid(logit)
        preds = (positive_class_probability > 0.5).int().flatten()
        for name, metricfn in self.metrics.items():
            self.info(
                f"DNN/{step_type}-{name}",
                metricfn(preds, outputs["target"].int()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.l2_penalty
        )

    def _PRAUC(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prcurve = pl.metrics.classification.PrecisionRecallCurve(
            num_classes=1, pos_label=1
        )
        prec, recall, _ = prcurve(pred, target)
        return pl.metrics.functional.classification.auc(recall, prec)


if __name__ == "__main__":
    from sklearn import datasets

    # TODO[LOW]: update this with new code
    from sklearn.model_selection import train_test_split
    from utils.get_set_cli_args import init_cli_args

    args = init_cli_args()
    # seed for np, torch, python.random, pythonhashseed
    pl.seed_everything(args.seed)

    bc = datasets.load_breast_cancer()
    X, y = bc["data"], bc["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.seed)

    max_epochs = 5  # or args.max_epochs (but that's a large #)
    pt = DNNClassifier(
        False,
        X_train.shape[1],
        max_epochs,
        args.seed,
        args.batch_size,
        args.num_gpus,
    )

    pt.fit(X_train, y_train, X_test, y_test)
    info(repr(pt.dnn))

    pt_res = pt.predict(X_test)
    # info(pt.predict_proba(iris_df))
