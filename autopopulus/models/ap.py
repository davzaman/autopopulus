import cloudpickle
from typing import Optional, Union
from argparse import ArgumentParser, Namespace
import pandas as pd
import warnings
from sklearn.base import TransformerMixin, BaseEstimator

#### Pytorch ####
import torch
from torch.utils.data import DataLoader

## Lightning ##
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profiler import Profiler
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.loggers.base import LightningLoggerBase

warnings.filterwarnings(action="ignore", category=LightningDeprecationWarning)

# Local
from autopopulus.models.ae import AEDitto
from autopopulus.utils.log_utils import (
    AutoencoderLogger,
    get_serialized_model_path,
)
from autopopulus.data import CommonDataModule
from autopopulus.data.types import DataTypeTimeDim
from autopopulus.data.constants import PATIENT_ID, TIME_LEVEL
from autopopulus.utils.utils import CLIInitialized
from autopopulus.data.dataset_classes import (
    CommonDatasetWithTransform,
    CommonTransformedDataset,
)


class AEImputer(TransformerMixin, BaseEstimator, CLIInitialized):
    """Imputer compatible with sklearn, uses autoencoder to do imputation on tabular data.
    Implements fit and transform.
    Wraps AEDitto which is a flexible Pytorch Lightning style Autoencoder.
    """

    def __init__(
        self,
        max_epochs: int = 100,
        patience: int = 7,
        logger: Optional[LightningLoggerBase] = None,
        tune_callback: Optional[Callback] = None,
        strategy: Optional[Strategy] = None,
        trial_num: Optional[int] = None,
        runtest: bool = False,
        fast_dev_run: int = None,
        profiler: Optional[Union[str, Profiler]] = None,
        num_gpus: int = 1,
        num_nodes: int = 1,
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        *args,  # For inner AEDitto
        **kwargs,
    ):
        self.profiler = profiler
        self.trial_num = trial_num
        self.fast_dev_run = fast_dev_run
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.runtest = runtest
        self.patience = patience
        self.max_epochs = max_epochs
        self.data_type_time_dim = data_type_time_dim
        # This is a convenience
        self.longitudinal = self.data_type_time_dim.is_longitudinal()

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
            tune_callback,
            strategy,
            profiler=self.profiler,
        )
        self.inference_trainer = self.create_trainer(
            logger,
            self.data_type_time_dim,
            self.patience,
            self.max_epochs,
            self.num_nodes,
            # Ensure run on 1 GPU if we want to use GPUs for correctness
            # Also bc synchronizing outputs on inference/predict is not supported for ddp
            # Ref: https://github.com/PyTorchLightning/pytorch-lightning/discussions/12906
            num_gpus=1 if self.num_gpus else 0,
            fast_dev_run=self.fast_dev_run,
            tune_callback=None,  # No tuning since we're testing
            strategy=None,
            profiler=self.profiler,
        )

    @staticmethod
    def create_trainer(
        logger: LightningLoggerBase,
        data_type_time_dim_name: str,
        patience: int,
        max_epochs: int,
        num_nodes: Optional[int] = None,
        num_gpus: Optional[int] = None,
        fast_dev_run: Optional[bool] = None,
        tune_callback: Optional[Callback] = None,
        strategy: Optional[Strategy] = None,
        profiler: Optional[Union[str, Profiler]] = None,
    ) -> pl.Trainer:
        callbacks = [
            # ModelSummary(max_depth=3),
            EarlyStopping(
                check_on_train_epoch_end=False,
                monitor=f"AE/{data_type_time_dim_name}/val-loss",
                patience=patience,
            ),
        ]

        # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761#discussioncomment-1152286
        if strategy is None:
            # Use DDP if there's more than 1 GPU, otherwise, it's not necessary.
            strategy = "ddp_find_unused_parameters_false" if num_gpus > 1 else None
            accelerator = "gpu" if num_gpus else "cpu"
        else:
            accelerator = None

        if tune_callback is not None:
            callbacks.append(tune_callback)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            deterministic=True,
            num_nodes=num_nodes,
            devices=num_gpus if num_gpus else "auto",
            accelerator=accelerator,
            strategy=strategy,
            # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html#distributed-and-16-bit-precision
            precision=16,
            enable_checkpointing=False,
            callbacks=callbacks,
            profiler=profiler,
            fast_dev_run=fast_dev_run,  # For debugging
        )
        return trainer

    def fit(self, data: CommonDataModule):
        """Trains the autoencoder for imputation."""
        self._fit(data)
        self._save_test_data(data)
        return self

    @rank_zero_only
    def _save_test_data(self, data: CommonDataModule):
        if self.runtest:  # Serialize test dataloader to run in separate script
            # Serialize data with torch but serialize model with plightning
            torch.save(
                data.test_dataloader(),
                get_serialized_model_path(
                    f"{self.data_type_time_dim.name}_test_dataloader",
                    "pt",
                    self.trial_num,
                ),
                pickle_module=cloudpickle,
            )

    def _fit(self, data: CommonDataModule):
        """Trains the autoencoder for imputation."""
        # set the data so the plmodule has a reference to it to use it after it's been setup to build the model dynamically in ae.setup()
        # can't instantiate the model here bc we need *args and **kwargs from init
        self.ae.datamodule = data
        self.trainer.fit(self.ae, datamodule=data)
        self.trainer.save_checkpoint(
            get_serialized_model_path(
                f"AEDitto_{self.data_type_time_dim.name}", "pt", self.trial_num
            ),
        )

    def transform(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Applies trained autoencoder to given data X.
        Calls predict on pl model and then recovers columns and indices.
        """
        preds_list = self.inference_trainer.predict(self.ae, dataloader)
        # stack the list of preds from dataloader
        preds = torch.vstack(preds_list).cpu().numpy()
        if isinstance(dataloader.dataset, CommonDatasetWithTransform):
            columns = dataloader.dataset.split["data"].columns
            reference_ids = dataloader.dataset.split_ids["data"]
        elif isinstance(dataloader.dataset, CommonTransformedDataset):
            columns = dataloader.dataset.transformed_data["original"]["data"].columns
            reference_ids = dataloader.dataset.split_ids

        # Recover IDs, we use only indices used by the batcher (so if we limit to debug, this still works, even if it's shuffled)
        ids = (
            reference_ids[: self.fast_dev_run * dataloader.batch_size]
            if self.fast_dev_run
            else reference_ids
        )
        # (n samples, t padded time points, f features) -> 2d pd df
        if self.longitudinal:
            index = pd.MultiIndex.from_product(
                [range(s) for s in preds.shape],
                names=[PATIENT_ID, TIME_LEVEL, "feature"],
            )
            index = index.set_levels(level=PATIENT_ID, levels=ids)
            preds_df = pd.DataFrame(preds.flatten(), index=index).unstack(
                level="feature"
            )
            preds_df.columns = columns
            return preds_df
        return pd.DataFrame(preds, columns=columns, index=ids)

    @classmethod
    def from_checkpoint(
        cls, args: Namespace, ae_from_checkpoint: Optional[str] = None, **kwargs
    ) -> "AEImputer":
        checkpoint = (
            ae_from_checkpoint
            if ae_from_checkpoint is not None
            else args.ae_from_checkpoint
        )
        logger = AutoencoderLogger(args)
        kwargs["tune_callback"] = None
        ae_imputer = cls.from_argparse_args(args, logger=logger, **kwargs)
        ae_imputer.ae = ae_imputer.load_autoencoder(checkpoint)
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
        return super().from_argparse_args(args, [AEDitto], **kwargs)

    @staticmethod
    def add_imputer_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument(
            "--num-nodes",
            type=int,
            default=1,
            help="Number of nodes in pytorch lightning distributed cluster.",
        )
        p.add_argument(
            "--max-epochs",
            type=int,
            default=100,
            help="When using the Autopopulus method, set the maximum number of epochs allowed for training the underlying autoencoder.",
        )
        p.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Using early stopping when training the underlying autoencoder for Autopopulus, set the patience for early stopping.",
        )
        # Tuning

        p.add_argument(
            "--tune-n-samples",
            type=int,
            default=0,
            help="When defining the distributions/choices to go over during hyperparameter tuning, how many samples to take.",
        )
        p.add_argument(
            "--n-gpus-per-trial",
            type=int,
            default=1,
            help="How many GPUs to use per hyperparameter trial when tuning.",
        )
        p.add_argument(
            "--num-gpus",
            type=int,
            default=4,
            help="Number of gpus for the pytorch dataset used in passing batches to the autoencoder.",
        )
        p.add_argument(
            "--total-cpus-on-machine",
            type=int,
            default=0,
            help="How many CPUs are on the machine running experiments.",
        )
        p.add_argument(
            "--total-gpus-on-machine",
            type=int,
            default=0,
            help="How many GPUs are on the machine running experiments.",
        )
        p.add_argument(
            "--fast-dev-run",
            type=int,
            default=0,
            help="Debugging: limits number of batches for train/val/test on deep learning model.",
        )
        return p
