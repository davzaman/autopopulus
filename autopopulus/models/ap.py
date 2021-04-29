import inspect
import sys
from typing import List, Optional, Union
from argparse import ArgumentParser, Namespace
from warnings import warn

import pandas as pd
import numpy as np

#### Pytorch ####
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#### Experiment Tracking ####
from tensorboardX import SummaryWriter
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# For Imputer Class
from sklearn.base import TransformerMixin, BaseEstimator

# Local
from models.ae import ACTIVATION_CHOICES, AEDitto, LOSS_CHOICES, OPTIM_CHOICES
from utils.log_utils import (
    MyLogger,
    get_serialized_model_path,
)
from utils.utils import YAMLStringListToList, str2bool
from data.utils import CommonDataModule
from data.transforms import simple_impute_tensor


class AEImputer(TransformerMixin, BaseEstimator):
    """Imputer compatible with sklearn, uses autoencoder to do imputation on tabular data.
    Implements fit and transform.
    Underlying autoencoder can be different flavors.
    """

    def __init__(
        self,
        n_features: int,
        max_epochs: int,
        seed: int,
        patience: int,
        hidden_layers: List[Union[int, float]],
        learning_rate: float,
        l2_penalty: float,
        lossn: str,
        optimn: str,
        activation: str,
        mvec: bool,
        vae: bool,
        undiscretize_data: bool = False,
        replace_nan_with: Optional[Union[int, str]] = None,
        dropout: Optional[float] = None,
        dropout_corruption: Optional[float] = None,
        batchswap_corruption: Optional[float] = None,
        summarywriter: Optional[SummaryWriter] = None,
        runtune: bool = False,
        runtest: bool = False,
        num_gpus: int = 1,
    ):
        self.runtune = runtune
        self.seed = seed
        self.n_features = n_features
        self.num_gpus = num_gpus
        self.runtest = runtest
        self.summarywriter = summarywriter
        # Set hparams
        self.max_epochs = max_epochs
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.patience = patience
        self.lossn = lossn
        self.optimn = optimn
        self.activation = activation
        self.mvec = mvec
        self.vae = vae
        self.dropout_corruption = dropout_corruption
        self.batchswap_corruption = batchswap_corruption
        self.undiscretize_data = undiscretize_data
        self.replace_nan_with = replace_nan_with

        self.ae = AEDitto(
            input_dim=n_features,
            hidden_layers=self.hidden_layers,
            lr=self.learning_rate,
            dropout=self.dropout,
            seed=seed,
            l2_penalty=self.l2_penalty,
            lossn=self.lossn,
            optimn=self.optimn,
            activation=self.activation,
            mvec=self.mvec,
            vae=self.vae,
            undiscretize_data=self.undiscretize_data,
            replace_nan_with=self.replace_nan_with,
            dropout_corruption=self.dropout_corruption,
            batchswap_corruption=self.batchswap_corruption,
        )

        callbacks = [EarlyStopping(monitor="AE/val-loss", patience=self.patience)]
        if self.runtune:
            callbacks.append(
                TuneReportCallback(
                    [
                        "AE/val-loss",
                        "impute/val-RMSE",
                        "impute/val-RMSE-missingonly",
                        "impute/val-MAAPE",
                        "impute/val-MAAPE-missingonly",
                    ],
                    on="validation_end",
                )
            )

        logger = MyLogger(summarywriter)
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            deterministic=True,
            gpus=self.num_gpus,
            accelerator="ddp" if self.num_gpus > 1 else None,
            # NOTE: CANNOT use "precision=16" speedup with any of the other paper methods.
            checkpoint_callback=False,
            callbacks=callbacks,
            profiler="simple",  # or "advanced" which is more granular
            weights_summary="full",
        )

    def fit(self, data: CommonDataModule):
        """Trains the autoencoder for imputation."""
        pl.seed_everything(self.seed)
        self.data = data
        self.ae.columns = self.data.columns
        self.ae.ctn_columns = self.data.ctn_columns
        if hasattr(self.data, "columns_disc"):
            self.ae.discrete_columns = self.data.columns_disc.to_list()

        if (
            self.replace_nan_with is None
            and not self.undiscretize_data
            and self.data.X_train.isna().any()
        ):
            warn(
                "WARNING: You did not indicate what value to replace nans with and are not undiscretizing the data, but NaNs were detected in the input. Please indicate what value you'd like to replace nans with."
            )

        self.trainer.fit(self.ae, datamodule=self.data)
        torch.save(self.ae.state_dict(), get_serialized_model_path("AEDitto", "pt"))

        if self.runtest and not self.runtune:
            self.trainer.test(self.ae, self.data.test_dataloader())

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        undiscretized_X: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Applies trained autoencoder to given data X."""
        assert (
            not self.undiscretize_data or undiscretized_X is not None
        ), "Indicated data will be undiscretized, but undiscretized data not passed in."
        undiscretized_X = (
            torch.tensor(
                undiscretized_X.values,
                device=self.ae.device,
                dtype=torch.float,
            )
            if self.undiscretize_data
            else None
        )

        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, device=self.ae.device, dtype=torch.float)
        else:
            X = torch.tensor(X, device=self.ae.device, dtype=torch.float)

        if self.replace_nan_with is not None:
            if self.replace_nan_with == "simple":  # simple impute warm start
                X = simple_impute_tensor(X, self.ae.ctn_cols_idx, self.ae.cat_cols_idx)
            else:  # Replace nans with a single value provided
                X = torch.nan_to_num(X, nan=self.replace_nan_with)

        if self.vae:
            logit, mu, var = self.ae(X)
        else:
            logit = self.ae(X)

        logit = logit.detach()

        imputed, _, _ = self.ae.get_imputed_tensor_from_model_output(
            X, logit, X, (~torch.isnan(X)).bool(), undiscretized_X, undiscretized_X
        )
        return imputed.detach().cpu().numpy()

    def load_autoencoder(self, serialized_model_path: str) -> None:
        """Loads the underlying autoencoder state dict from path."""
        self.ae.load_state_dict(torch.load(serialized_model_path))

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
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
        imputer_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        imputer_kwargs.update(**kwargs)

        return cls(**imputer_kwargs)

    @staticmethod
    def add_imputer_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--learning-rate",
            type=float,
            required="--method=ap" in sys.argv,
            help="When using the Autopopulus method, set the learning rate for the underlying autoencoder when training.",
        )
        p.add_argument(
            "--max-epochs",
            type=int,
            required="--method=ap" in sys.argv,
            default=100,
            help="When using the Autopopulus method, set the maximum number of epochs allowed for training the underlying autoencoder.",
        )
        p.add_argument(
            "--patience",
            type=int,
            required="--method=ap" in sys.argv,
            default=5,
            help="Using early stopping when training the underlying autoencoder for Autopopulus, set the patience for early stopping.",
        )
        p.add_argument(
            "--hidden-layers",
            type=str,
            required="--method=ap" in sys.argv,
            action=YAMLStringListToList(convert=float),
            help="A comma separated list of integers or float point numbers (with no spaces) that represent the size of each hidden layer. Float point will compute relative size to input.",
        )
        p.add_argument(
            "--l2-penalty",
            type=float,
            default=0,
            help="When training the autoencoder, what weight decay or l2 penalty to apply to the optimizer.",
        )
        p.add_argument(
            "--dropout",
            type=Optional[float],
            default=None,
            help="When training the autoencoder, what dropout to use (if at all) between layers.",
        )
        p.add_argument(
            "--lossn",
            type=str,
            choices=LOSS_CHOICES,
            default="BCE",
            help="When training the autoencoder, what type of loss to use.",
        )
        p.add_argument(
            "--optimn",
            type=str,
            choices=OPTIM_CHOICES,
            default="Adam",
            help="When training the autoencoder, what optimizer to use.",
        )
        p.add_argument(
            "--activation",
            type=str,
            choices=ACTIVATION_CHOICES,
            default="ReLU",
            help="When training the autoencoder, what activation function to use between each layer.",
        )
        p.add_argument(
            "--mvec",
            type=str2bool,
            default=False,
            help="When training the autoencoder, ignore missing values in the loss.",
        )
        p.add_argument(
            "--vae",
            type=str2bool,
            default=False,
            help="Use a variational autoencoder.",
        )
        p.add_argument(
            "--dropout-corruption",
            type=Optional[float],
            default=None,
            help="If implementing a denoising autoencoder, what percentage of corruption at the input using dropout (noise is 0's).",
        )
        p.add_argument(
            "--batchswap-corruption",
            type=Optional[float],
            default=None,
            help="If implementing a denoising autoencoder, what percentage of corruption at the input, swapping out values as noise.",
        )
        # Tuning
        p.add_argument(
            "--experiment-name",
            type=str,
            default="myexperiment",
            help="When running tuning, what experiment name to set. The guild file also shares this name.",
        )
        p.add_argument(
            "--tune-n-samples",
            type=int,
            default=1,
            help="When defining the distributions/choices to go over during hyperparameter tuning, how many samples to take.",
        )
        p.add_argument(
            "--runtune",
            type=str2bool,
            default=False,
            help="Whether or not to run tuning instead of single training.",
        )
        return p
