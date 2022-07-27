import inspect
import cloudpickle
import sys
from typing import Optional, Union
from argparse import ArgumentParser, Namespace

# import numpy as np
import pandas as pd

#### Pytorch ####
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

#### Experiment Tracking ####
from tensorboardX import SummaryWriter
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# For Imputer Class
from sklearn.base import TransformerMixin, BaseEstimator

# Local
from autopopulus.models.ae import AEDitto
from autopopulus.utils.log_utils import (
    MyLogger,
    get_serialized_model_path,
)
from autopopulus.utils.cli_arg_utils import str2bool
from autopopulus.data import CommonDataModule
from autopopulus.data.types import DataTypeTimeDim
from data.constants import PATIENT_ID


class AEImputer(TransformerMixin, BaseEstimator):
    """Imputer compatible with sklearn, uses autoencoder to do imputation on tabular data.
    Implements fit and transform.
    Wraps AEDitto which is a flexible Pytorch Lightning style Autoencoder.
    """

    def __init__(
        self,
        max_epochs: int = 100,
        patience: int = 7,
        summarywriter: Optional[SummaryWriter] = None,
        runtune: bool = False,
        runtest: bool = False,
        fast_dev_run: int = None,
        num_gpus: int = 1,
        num_nodes: int = 1,
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        *args,  # For inner AEDitto
        **kwargs,
    ):
        self.runtune = runtune
        self.fast_dev_run = fast_dev_run
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.runtest = runtest
        self.patience = patience
        self.max_epochs = max_epochs
        self.data_type_time_dim = data_type_time_dim
        # This is a convenience
        self.longitudinal = self.data_type_time_dim.is_longitudinal()
        logger = MyLogger(summarywriter)

        # Create AE and trainer
        self.ae = AEDitto(
            *args,
            **kwargs,
            longitudinal=self.longitudinal,
            data_type_time_dim=self.data_type_time_dim,
        )
        self.trainer = self.create_trainer(
            logger,
            self.data_type_time_dim.name,
            self.patience,
            self.max_epochs,
            self.num_nodes,
            self.num_gpus,
            self.fast_dev_run,
            self.runtune,
        )

    @staticmethod
    def create_trainer(
        logger: MyLogger,
        data_type_time_dim_name: str,
        patience: int,
        max_epochs: int,
        num_nodes: Optional[int] = None,
        num_gpus: Optional[int] = None,
        fast_dev_run: Optional[bool] = None,
        tune: Optional[bool] = None,
    ) -> pl.Trainer:
        callbacks = [
            # ModelSummary(max_depth=3),
            EarlyStopping(
                check_on_train_epoch_end=False,
                monitor=f"AE/{data_type_time_dim_name}/val-loss",
                patience=patience,
            ),
        ]
        if tune:
            callbacks.append(
                TuneReportCallback(
                    [
                        f"AE/{data_type_time_dim_name}/val-loss",
                        f"impute/{data_type_time_dim_name}/val-RMSE",
                        f"impute/{data_type_time_dim_name}/val-RMSE-missingonly",
                        f"impute/{data_type_time_dim_name}/val-MAAPE",
                        f"impute/{data_type_time_dim_name}/val-MAAPE-missingonly",
                    ],
                    on="validation_end",
                )
            )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            # deterministic=True,  # TODO im running into th lstm determinism error but their suggested fixes aren't working...
            num_nodes=num_nodes,
            # use 1 processes if on cpu
            devices=num_gpus if num_gpus else 1,
            accelerator="gpu" if num_gpus else "cpu",
            # strategy="ddp" if self.num_gpus > 1 else None,
            strategy=DDPPlugin(find_unused_parameters=False) if num_gpus > 1 else None,
            # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761#discussioncomment-1152286
            # plugins=DDPPlugin(find_unused_parameters=False)
            # if self.num_gpus > 1
            # else None,
            # NOTE: CANNOT use "precision=16" speedup with any of the other paper methods.
            enable_checkpointing=False,
            callbacks=callbacks,
            profiler="simple",  # or "advanced" which is more granular
            fast_dev_run=fast_dev_run,  # For debugging
        )
        return trainer

    def fit(self, data: CommonDataModule):
        """Trains the autoencoder for imputation."""
        self._fit(data)

        if self.runtest:  # Serialize test dataloader to run in separate script
            # Serialize data with torch but serialize model with plightning
            torch.save(
                data.test_dataloader(),
                get_serialized_model_path("static_test_dataloader", "pt"),
                pickle_module=cloudpickle,
            )
        return self

    def _fit(self, data: CommonDataModule):
        """Trains the autoencoder for imputation."""
        # set the data so the plmodule has a reference to it to use it after it's been setup to build the model dynamically in ae.setup()
        # can't instantiate the model here bc we need *args and **kwargs from init
        self.ae.datamodule = data
        self.trainer.fit(self.ae, datamodule=data)
        self.trainer.save_checkpoint(
            get_serialized_model_path(f"AEDitto_{self.data_type_time_dim}", "pt"),
        )

    def transform(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Applies trained autoencoder to given data X.
        Calls predict on pl model and then recovers columns and indices.
        """
        preds_list = self.trainer.predict(self.ae, dataloader)
        # stack the list of preds from dataloader
        preds = torch.vstack(preds_list).cpu().numpy()
        columns = dataloader.dataset.split["data"]["normal"].columns

        # Recover IDs, we use only indices used by the batcher (so if we limit to debug, this still works, even if it's shuffled)
        ids = (
            dataloader.dataset.split_ids["data"]["normal"][
                : self.fast_dev_run * dataloader.batch_size
            ]
            if self.fast_dev_run
            else dataloader.dataset.split_ids["data"]["normal"]
        )
        # (n samples, t padded time points, f features) -> 2d pd df
        if self.longitudinal:
            index = pd.MultiIndex.from_product(
                [range(s) for s in preds.shape], names=[PATIENT_ID, "time", "feature"]
            )
            index = index.set_levels(level=PATIENT_ID, levels=ids)
            preds_df = pd.DataFrame(preds.flatten(), index=index).unstack(
                level="feature"
            )
            preds_df.columns = dataloader.dataset.split["data"]["normal"].columns
            return preds_df
        return pd.DataFrame(preds, columns=columns, index=ids)

    @classmethod
    def from_checkpoint(cls, args: Namespace) -> "AEImputer":
        ae_imputer = cls.from_argparse_args(args, runtune=False)
        ae_imputer.ae = ae_imputer.load_autoencoder(args.ae_from_checkpoint)
        return ae_imputer

    @staticmethod
    def load_autoencoder(serialized_model_path: str) -> None:
        """Loads the underlying autoencoder state dict from path."""
        # Ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        autoencoder = AEDitto.load_from_checkpoint(serialized_model_path)
        # put into eval mode for inference
        autoencoder.eval()
        return autoencoder

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
        # returns a immutable dict MappingProxyType, want to combine so copy
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        # Update with stuff required for AEDitto
        valid_kwargs.update(inspect.signature(AEDitto.__init__).parameters.copy())
        imputer_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        imputer_kwargs.update(**kwargs)

        return cls(**imputer_kwargs)

    @staticmethod
    def add_imputer_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--num-nodes",
            type=int,
            required="--metho=ap" in sys.argv,
            default=1,
            help="Number of nodes in pytorch lightning distributed cluster.",
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
            "--fast-dev-run",
            type=int,
            default=0,
            help="Debugging: limits number of batches for train/val/test on deep learning model.",
        )
        p.add_argument(
            "--runtune",
            type=str2bool,
            default=False,
            help="Whether or not to run tuning instead of single training.",
        )
        return p
