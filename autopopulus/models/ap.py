import cloudpickle
from typing import Any, Dict, List, Optional, Tuple, Union
from argparse import ArgumentParser, Namespace
import pandas as pd
import warnings
from sklearn.base import TransformerMixin, BaseEstimator

#### Pytorch ####
import torch
from torch.utils.data import DataLoader

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    LightningTrainer,
    LightningConfigBuilder,
    LightningCheckpoint,
)

## Lightning ##
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.loggers import Logger

from autopopulus.task_logic.utils import ImputerT, get_tune_metric

# from pl_bolts.callbacks import ModuleDataMonitor, BatchGradientVerificationCallback

warnings.filterwarnings(action="ignore", category=LightningDeprecationWarning)

# Local
from autopopulus.models.ae import AEDitto
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    TUNE_LOG_DIR,
    AutoencoderLogger,
    copy_artifacts_from_tune,
    get_serialized_model_path,
)
from autopopulus.models.callbacks import EpochTimerCallback, VisualizeModelCallback
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
        patience: int = 3,
        logger: Optional[Logger] = None,
        tune_callback: Optional[Callback] = None,
        strategy: Strategy = "auto",
        trial_num: Optional[int] = None,
        runtest: bool = False,
        fast_dev_run: int = None,
        model_monitoring: bool = False,
        early_stopping: bool = True,
        profiler: Optional[Union[str, Profiler]] = None,
        num_gpus: int = 1,
        num_nodes: int = 1,
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        *args,  # For inner AEDitto
        **kwargs,
    ):
        self.max_epochs = max_epochs
        self.patience = patience
        self.logger = logger
        self.tune_callback = tune_callback
        self.strategy = strategy
        self.trial_num = trial_num
        self.runtest = runtest
        self.fast_dev_run = fast_dev_run
        self.model_monitoring = model_monitoring
        self.early_stopping = early_stopping
        self.profiler = profiler
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.data_type_time_dim = data_type_time_dim
        self.ae_args = args
        self.ae_kwargs = kwargs
        # This is a convenience
        self.longitudinal = self.data_type_time_dim.is_longitudinal()

        # train trainer needs to know the data feature space so its created on fit
        # inference trainer doesn't need this so we can create on ini
        self.inference_trainer = pl.Trainer(
            **self._get_trainer_args(
                data=None,
                trainer_overrides={
                    # Ensure run on 1 GPU if we want to use GPUs for correctness
                    # Also bc synchronizing outputs on inference/predict is not supported for ddp
                    # Ref: https://github.com/PyTorchLightning/pytorch-lightning/discussions/12906
                    "devices": 1 if self.num_gpus else 0,
                    "num_nodes": 1,
                },
                callback_overrides={  # don't care about early stopping during eval
                    "loss_feature_space": None,
                    "early_stopping": False,
                    "tune_callback": None,  # No tuning since we're testing
                },
            )
        )

    def fit(self, data: CommonDataModule):
        """Trains the autoencoder for imputation."""
        data.setup("fit")

        args, kwargs = self._get_model_args(data)
        self.ae = AEDitto(*args, **kwargs)

        self.trainer = pl.Trainer(**self._get_trainer_args(data))
        self.trainer.fit(self.ae, datamodule=data)
        self.trainer.save_checkpoint(
            get_serialized_model_path(
                f"AEDitto_{self.data_type_time_dim.name}", "pt", self.trial_num
            ),
        )
        self._save_test_data(data)
        return self

    def tune(
        self,
        experiment_name: str,
        tune_n_samples: int,
        total_cpus_on_machine: int,
        total_gpus_on_machine: int,
        n_gpus_per_trial: int,
        data: CommonDataModule,
    ):
        """
        Set to use Ray 2.4.0.
        Ref: https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html
        Ref: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
        """
        data.setup("fit")
        # this is matching get_args_from_data.
        # TODO[LOW]: This should be changed to be more easily in sync with get_args_from_data
        data_feature_space = "mapped" if "mapped" in data.groupby else "original"
        tune_metric: str = get_tune_metric(ImputerT.AE, data, data_feature_space)
        #### Setup LightningTrainer ####
        run_config = RunConfig(
            name=experiment_name,
            # TODO: if i'm doing keep=1 do i need this?
            local_dir=TUNE_LOG_DIR,
            # define AIR CheckpointConfig to properly save checkpoints in AIR format.
            checkpoint_config=CheckpointConfig(
                # this needs to basically mirror .checkpointing(...) above
                num_to_keep=1,
                # deciding how to kick out old checkpoints
                checkpoint_score_attribute=tune_metric,
                checkpoint_score_order="min",
            ),
        )

        # Uses the total hardware on the machine and the gpus per trial requested
        # to determine how many CPUs per trial.
        # More workers means more parallel dataloading, but more overhead (slower start).
        ncpu_per_gpu = total_cpus_on_machine // total_gpus_on_machine
        scaling_config = ScalingConfig(
            num_workers=min(4, ncpu_per_gpu),
            use_gpu=self.num_gpus > 0,
            # TODO: should cpu be 1?
            resources_per_worker={
                "CPU": int(0.8 * ncpu_per_gpu),
                "GPU": min(n_gpus_per_trial, self.num_gpus),
            },
        )
        # set up the model without the param grid params
        ae_args, ae_kwargs = self._get_model_args(data)
        lightning_config = (
            LightningConfigBuilder()
            .module(cls=AEDitto, *ae_args, **ae_kwargs)
            .ddp_strategy(find_unused_parameters=False)
            .trainer(
                **self._get_trainer_args(
                    data, trainer_overrides={"enable_checkpointing": True}
                )
            )
            .fit_params(datamodule=data)
            # LightningTrainer: freq of metric reporting == freq of checkpointing
            .checkpointing(monitor=tune_metric, save_top_k=1, mode="min")
            .build()
        )
        # separately passing constant configs to Trainer and searchable configs to Tuner (one for search one for the model itself)
        lightning_trainer = LightningTrainer(
            lightning_config=lightning_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        #### Setup Tuner ####
        # config for tuner
        search_lightning_config = (
            LightningConfigBuilder().module(**self._get_tune_grid(data)).build()
        )
        tuner = tune.Tuner(
            lightning_trainer,
            param_space={"lightning_config": search_lightning_config},
            tune_config=tune.TuneConfig(
                metric=tune_metric,
                mode="min",
                scheduler=ASHAScheduler(),
                num_samples=tune_n_samples,
                trial_name_creator=lambda trial: trial.trial_id,
            ),
            # run_config=air.RunConfig(),  # different from ray.air.config.RunConfig
        )
        result = tuner.fit()
        best_result: air.Result = result.get_best_result(metric=tune_metric, mode="min")
        self._save_artifacts_from_tune(best_result)
        self._save_test_data(data)
        return self

    def transform(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Applies trained autoencoder to given data X.
        Calls predict on pl model and then recovers columns and indices.
        """
        assert hasattr(self, "ae"), "You need to call tune or fit first!"
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

    ######################
    #     Model Args     #
    ######################
    def _get_model_args(self, data: CommonDataModule) -> Tuple[Any, Any]:
        return (
            self.ae_args,
            {
                "longitudinal": self.longitudinal,
                "data_type_time_dim": self.data_type_time_dim,
                **self.ae_kwargs,
                **self._get_args_from_data(data),
            },
        )

    def _get_args_from_data(self, data: CommonDataModule):
        """
        Set model info that we can only dynamically get from the data after it's been setup(). Normally this happens in trainer.fit()
        However, I need to build the model in __init__ so I need this info upfront.
        Any attribute set here should be serialized in `on_save_checkpoint`.
        NOTE: If there's onehot in col_idxs_by_type, it will be PADDED .
        Anything that uses it will need to account for that.
        """
        return {
            "nfeatures": data.nfeatures,
            "columns": data.columns,
            "discretizations": data.discretizations,
            "inverse_target_encode_map": data.inverse_target_encode_map,
            "feature_map": data.feature_map,
            "data_feature_space": "mapped" if "mapped" in data.groupby else "original",
            # We still need this if we're loading the ae from a file and not calling fit
            "col_idxs_by_type": data.col_idxs_by_type,
            "semi_observed_training": data.semi_observed_training,
            "evaluate_on_remaining_semi_observed": data.evaluate_on_remaining_semi_observed,
        }

    def _get_trainer_args(
        self,
        data: Optional[CommonDataModule] = None,
        trainer_overrides: Optional[Dict[str, Any]] = None,
        callback_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """This MUST get called after data.setup("fit")!!!!"""

        trainer_args = {
            "logger": self.logger,
            "max_epochs": self.max_epochs,
            "deterministic": True,
            "num_nodes": self.num_nodes,
            "devices": self.num_gpus,
            # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html#distributed-and-16-bit-precision
            "precision": 16,
            "enable_checkpointing": False,
            "strategy": self.strategy,
            "profiler": self.profiler,
            "fast_dev_run": self.fast_dev_run,
        }
        if data is not None:
            trainer_args["callbacks"] = self._get_callbacks(data, callback_overrides)

        if trainer_overrides is not None:
            assert all(
                [k in trainer_args for k in trainer_overrides]
            ), "Provided a trainer override for a key that doesn't exist."
            trainer_args.update(trainer_overrides)

        # update these after potentially overriding trainer args
        # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761#discussioncomment-1152286
        if trainer_args["strategy"] == "auto":
            # Use DDP if there's more than 1 GPU, otherwise, it's not necessary.
            trainer_args["strategy"] = (
                "ddp_find_unused_parameters_false"
                if trainer_args["devices"] > 1
                else None
            )
            trainer_args["accelerator"] = "gpu" if trainer_args["devices"] else "cpu"
        else:
            trainer_args["accelerator"] = None
        # this should come after setting strategy and accelerator
        if trainer_args["devices"] == 0:
            trainer_args["devices"] = "auto"

        return trainer_args

    def _get_callback_args(self, data: CommonDataModule) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "loss_feature_space": "mapped" if "mapped" in data.groupby else "original",
            "model_monitoring": self.model_monitoring,
            "early_stopping": self.early_stopping,
            "tune_callback": self.tune_callback,
        }

    def _get_callbacks(
        self,
        data: CommonDataModule,
        callback_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Callback]:
        """To pass to pl.Trainer."""
        callback_args = self._get_callback_args(data)
        if callback_overrides is not None:
            assert all(
                [k in callback_args for k in callback_overrides]
            ), "Provided a callback override for a key that doesn't exist."
            callback_args.update(callback_overrides)
        return self._create_callbacks_from_args(**callback_args)

    @staticmethod
    def _create_callbacks_from_args(
        patience: int,
        loss_feature_space: str,
        model_monitoring: bool,
        early_stopping: bool,
        tune_callback: Callback,
    ) -> List[Callback]:
        """To pass to pl.Trainer."""
        callbacks = []  # ModelSummary(max_depth=3),
        callbacks.append(EpochTimerCallback())
        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    check_on_train_epoch_end=False,
                    monitor=IMPUTE_METRIC_TAG_FORMAT.format(
                        name="loss",
                        feature_space=loss_feature_space,
                        filter_subgroup="all",
                        reduction="NA",
                        split="val",
                        feature_type="mixed",
                    ),
                    patience=patience,
                )
            )
        if model_monitoring:
            callbacks.append(VisualizeModelCallback())
        if tune_callback is not None:
            callbacks.append(tune_callback)

        return callbacks

    ######################
    #    Tune Helpers    #
    ######################
    def _save_artifacts_from_tune(self, best_result: air.Result):
        # Assign best model, and then copy the metrics and model to the right dir for logging
        checkpoint: LightningCheckpoint = best_result.checkpoint
        best_model: pl.LightningModule = checkpoint.get_model(AEDitto)
        self.ae = best_model
        model_log_path = get_serialized_model_path(
            f"AEDitto_{self.data_type_time_dim.name}", "pt", self.trial_num
        )
        copy_artifacts_from_tune(
            best_result, model_path=model_log_path, metric_path=self.logger.save_dir
        )

    def _get_tune_grid(self, data: CommonDataModule) -> Dict[str, Any]:
        if self.fast_dev_run or data.limit_data:
            # Will have to set cuda_visible devices before running the python code if true
            # ray.init(local_mode=True)  # Local debugging for tuning
            # tune will try everything in the grid, so just do 1
            hidden_layers_grid = [[0.5]]
            batchnorm = [False]
        else:
            hidden_layers_grid = [
                [0.5, 0.25, 0.5],
                [0.5],
                [1.0, 0.5, 1.0],
                [1.5],
                [1.0, 1.5, 1.0],
            ]
            batchnorm = [True, False]

        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "l2_penalty": tune.loguniform(1e-5, 1),
            "batchnorm": tune.choice(batchnorm),
            # "patience": tune.choice([3, 5, 10]),
            # assume discretized, so num inputs on the dataset is 56
            "hidden_layers": tune.choice(hidden_layers_grid),
        }
        return config

    ######################
    #    Checkpointing   #
    ######################
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

    @classmethod
    def from_checkpoint(
        cls, args: Namespace, ae_from_checkpoint: Optional[str] = None, **kwargs
    ) -> "AEImputer":
        checkpoint = (
            ae_from_checkpoint
            if ae_from_checkpoint is not None
            else args.ae_from_checkpoint
        )
        kwargs["tune_callback"] = None
        ae_imputer = cls.from_argparse_args(
            args, logger=AutoencoderLogger(args), **kwargs
        )
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

    ######################
    #        Args        #
    ######################
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
            default=3,
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
        p.add_argument(
            "--model-monitoring",
            type=bool,
            default=False,
            help="Adds callbacks to the trainer to monitor the gradient and data passed to the model amongst other checks.",
        )
        return p
