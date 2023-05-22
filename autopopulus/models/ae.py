from argparse import ArgumentParser
from math import ceil
import sys
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from pandas import DataFrame, Index
from torchmetrics import Metric
from numpy import array

#### Pytorch ####
from torch import (
    exp,
    isnan,
    nan_to_num,
    randn_like,
    Tensor,
)
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

## Lightning ##
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from lightning_utilities.core.apply_func import apply_to_collection

from autopopulus.models.torch_model_utils import (
    CtnCatLoss,
    BatchSwapNoise,
    ReconstructionKLDivergenceLoss,
    ResetSeed,
    binary_column_threshold,
    detach_tensor,
    onehot_column_threshold,
)
from autopopulus.data.transforms import (
    get_invert_discretize_tensor_args,
    get_invert_target_encode_tensor_args,
    invert_discretize_tensor_gpu,
    invert_target_encoding_tensor_gpu,
    list_to_tensor,
    simple_impute_tensor,
)
from autopopulus.utils.impute_metrics import AccuracyMetric, MAAPEMetric, RMSEMetric
from autopopulus.utils.cli_arg_utils import YAMLStringListToList, StringOrInt, str2bool
from autopopulus.utils.utils import rank_zero_print
from autopopulus.utils.log_utils import IMPUTE_METRIC_TAG_FORMAT
from autopopulus.data.types import DataTypeTimeDim

HiddenAndCellState = Tuple[Tensor, Tensor]
MuLogVar = Tuple[Tensor, Tensor]

LOSS_CHOICES = ["BCE", "MSE", "CEMSE", "CEMAAPE"]
OPTIM_CHOICES = ["Adam", "SGD"]

COL_IDXS_BY_TYPE_FORMAT = "idx_{data_feature_space}_{feature_type}"


class AEDitto(LightningModule):
    """
                  67?7I?.777+.779
        6777:.,77.?IIIIIIIIIIII?7
      6777IIII???IIIIIIIIIIIIIII?7
      677IIIIII  II:IIIII  I??I?.??/77779
     6777=?????IIIIIII+=?==+???III~777779
      6777IIII? IIIIIIIIIII III?II?I.779
        q7=?????           IIIIIIIIII??,
         77I???IIIII????????????IIIIIIIII9
         7=??II???IIIII????IIIIIIIIIIIII.
         7+????????????????I??IIIII???I79
        7++?IIIIIIII????III???IIIIIIII77
      7?+++IIIIIIII?????IIIIIIIIIIIII?779
    77+++++??II???IIIIIIIIIIIII?II???++7
    7++++++++++++++++IIIII??+++++++++++++9
      7+++++++++++++++++++++++++++++++++++
          7+++++++++++++++++++++++++++++7
               7++++++++++++++7
    Pytorch-based Autoencoder model. Ditto because it can be any flavor.
    Layers can be set directly or proportionally.
    Flavors:
        - {overcomplete, undercomplete}
            - depends on hidden_layers
        - {Denoising, Vanilla, Variational}
            - Denoising via {dropout (noise is value 0), batchswap}_corruption
            - variational via variational flag.
        - {static, longitudinal}
        - {dropout, no dropout}
        - warm start {0, simple (mean/mode)}
        - losses {mvec, all} x {BCE, MSE, BCE (CAT) + MSE (CTN), reconstruction+kldivergence, accuracy (discrete version only)}
            - BCE is always with logits.
            - reconstruction+kldivergence is always when variational flag is chosen.
            - mvec ensures loss is only computed on originally non missing data
    """

    def __init__(
        self,
        seed: int,
        # Data hps # TODO: type hint data hps better
        nfeatures: Dict[str, int],
        # groupby: Dict,  # todo i dont think i need this
        columns: Dict[str, Dict[str, Index]],
        discretizations: Dict,
        inverse_target_encode_map: Dict[
            str, Dict[str, DataFrame]
        ],  # map/ordinal map -> bincol -> index=category names, vals=numerical encodings
        feature_map: str,
        data_feature_space: str,  # original/mapped
        col_idxs_by_type: Dict[str, Dict[str, List]],  # orig/map-> {cat/ctn -> idxs}
        # model hps
        hidden_layers: List[Union[int, float]],
        learning_rate: float,
        l2_penalty: float = 0,
        lossn: str = "CEMAAPE",
        optimn: str = "Adam",
        activation: str = "ReLU",
        metrics: Optional[  # SEPARATE FROM LOSS, only for evaluation
            List[
                Dict[  # NOTE: Any should be ... (kwargs) but not supported yet
                    str,
                    Union[str, Union[Metric, Callable[[Tensor, Tensor, Any], Tensor]]],
                ]
            ]
        ] = None,
        replace_nan_with: Optional[Union[int, str]] = None,  # warm start
        mvec: bool = False,
        batchnorm: bool = False,
        variational: bool = False,
        longitudinal: bool = False,  # Convenience on top of data type time dim
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        dropout: Optional[float] = None,
        dropout_corruption: Optional[float] = None,
        batchswap_corruption: Optional[float] = None,
    ):
        super().__init__()
        # keep col_idxs_by_type for validation of args even though we set up buffers
        self.save_hyperparameters(ignore=["metrics"])  # Required for serialization

        # everythign assigned here should be modified in on_load/save_checkpoint
        self.hidden_and_cell_state = {}
        self.set_feature_map_inversion(self.hparams.feature_map)
        # everythign below here should create attributes that will go into the state dict
        self.validate_args()
        # this needs to come before init metrics
        self.set_col_idxs_by_type_as_buffers(col_idxs_by_type)
        self.init_metrics(metrics)
        self._model_creation()

    #######################
    #    Forward Logic    #
    #######################
    def forward(
        self,
        split: str,
        X: Tensor,
        seq_len: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, MuLogVar]]:
        """
        What happens when you pass data through an object of this class: encode then decode.
        We need to know which split so if using an RNN we can use the correct H,C.
        If the last batch in training is < batch_size, the dims of H,C will be incorrect for validation.
        Also H,C from train should not contribute to H,C for validation, etc.
        """
        if self.hparams.variational:
            # If RNN pass through (h, c) and get the new one back
            code, (mu, logvar) = self.encode(split, X, seq_len)
            return self.decode(split, code, seq_len), (mu, logvar)

        code = self.encode(split, X, seq_len)
        return self.decode(split, code, seq_len)

    def encode(
        self,
        split: str,
        X: Tensor,
        orig_seq_lens: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, MuLogVar]]:
        """Returns code for AE/DAE, mu/var for VAE."""
        # if using rnn we need to know at which depth/layer we are
        # since the layers are mixed i can't just increment without conditional logic
        self.curr_rnn_depth = None

        for layer in self.encoder:
            X = self.apply_layer(split, layer, X, orig_seq_lens)

        if self.hparams.variational:
            mu, logvar = self.fc_mu(X), self.fc_var(X)
            return self.reparameterize(mu, logvar), (mu, logvar)

        return X

    def decode(
        self,
        split: str,
        X: Tensor,
        orig_seq_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, HiddenAndCellState]:
        self.curr_rnn_depth = None

        for layer in self.decoder:
            X = self.apply_layer(split, layer, X, orig_seq_lens)
        return X

    def apply_layer(
        self,
        split: str,
        layer: nn.Module,
        X: Union[Tensor, PackedSequence],
        orig_seq_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, HiddenAndCellState]:
        """Delegates logic for applying a layer of the AE based on its type."""
        if isinstance(layer, nn.RNNBase) and orig_seq_lens is not None:
            if self.curr_rnn_depth is None:
                self.curr_rnn_depth = 0
            else:
                self.curr_rnn_depth += 1
            return self.apply_rnn_layer(split, layer, X, orig_seq_lens)
        if isinstance(X, PackedSequence):
            X, _ = pad_packed_sequence(X, batch_first=True)
        return layer(X)

    def apply_rnn_layer(
        self,
        split: str,
        rnn_layer: nn.RNNBase,
        X: Union[Tensor, PackedSequence],
        orig_seq_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, HiddenAndCellState]:
        """Deals with special output of RNNBased layers."""
        X_packed = (
            pack_padded_sequence(
                X, orig_seq_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            if isinstance(X, Tensor)
            else X  # already packed
        )
        # pass data to the rnn, pick H,C for the current split (and update it)
        # (
        # rnn_output_packed,
        # self.hidden_and_cell_state[split][self.curr_rnn_depth],
        # ) = rnn_layer(X_packed, self.hidden_and_cell_state[split][self.curr_rnn_depth])
        rnn_output_packed, _ = rnn_layer(X_packed)
        # undo the packing operation
        # lstm_output, lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # grab last hidden state
        # return hidden[-1]

        # input to next layer is collective hidden states for all time points of this layer
        return rnn_output_packed

    #######################
    # Training/Eval Logic #
    #######################
    # Ref: https://pytorch-lightning.readthedocs.io/en/stable/advanced/sequences.html
    # For (h,c) passthrough (currently not using, since validation doesn't accept hiddens)
    def training_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch, "train")
        self.shared_logging_step_end(outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch, "val")
        self.shared_logging_step_end(outputs, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self.shared_step(batch, "test")
        self.shared_logging_step_end(outputs, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch, "predict")

    def shared_step(self, batch, split: str) -> Tuple[float, Dict[str, float]]:
        ### Unpack ###
        data, ground_truth = (
            batch[self.hparams.data_feature_space]["data"],
            batch[self.hparams.data_feature_space]["ground_truth"],
        )
        seq_len = (
            batch[self.hparams.data_feature_space]["seq_len"]
            if "seq_len" in batch[self.hparams.data_feature_space]
            else None
        )
        # set this before filling in data with replacement (if doing so)
        where_data_are_observed = ~(isnan(data)).bool()

        ### Warm Start ###
        # replace nan in ground truth too if its missing any
        if self.hparams.replace_nan_with is not None:
            data, ground_truth = self._warm_start_imputation(
                split, data, ground_truth, where_data_are_observed
            )

        ### Model ###
        # pass through the sequence length (if they're none, nothing happens)
        reconstruct_batch = self(split, data, seq_len)
        if self.hparams.variational:  # unpack when vae
            reconstruct_batch, (mu, logvar) = reconstruct_batch

        # On Predict stop here, no loss/evaluation
        if split == "predict":
            return pred

        ### Loss ###
        """
        Note: mvec will treat missing values as 0 (ignore in loss during training)
        data * mask: normalize by # all features (missing and observed).
        data[mask]: normalize by only # observed features. (Want)
        """
        if self.hparams.mvec:  # and self.training
            eval_pred = reconstruct_batch[where_data_are_observed]
            eval_true = ground_truth[where_data_are_observed]
        else:
            eval_pred = reconstruct_batch
            eval_true = ground_truth

        if self.hparams.variational:
            loss = self.loss(eval_pred, eval_true, mu, logvar)
        else:
            # NOTE: if no mvec and no vae for some reason it says the loss is modifying the output in place, the clone is a quick hack
            # loss = self.loss(eval_pred.clone(), eval_true)
            loss = self.loss(eval_pred, eval_true)

        #### Evaluations ####
        # detach so it metric computation doesn't go into the computational graph
        # cpu since feature space inversion atm can't go on the GPU
        # float because sigmoid on half precision isn't implemented for CPU
        detached_data = apply_to_collection(
            (data, reconstruct_batch, ground_truth, where_data_are_observed),
            Tensor,
            detach_tensor,
            to_cpu=False,
        )
        (
            pred,
            ground_truth,
            where_data_are_observed,
        ) = self.get_imputed_tensor_from_model_output(
            *detached_data,
            detach_tensor(batch["original"]["data"]),
            detach_tensor(batch["original"]["ground_truth"]),
            "original",
        )

        # evaluate in mapped feature space
        if "mapped" in batch:  # test not empty or None
            # assumes detached_data has not been changed
            self.metric_logging_step(
                *self.get_imputed_tensor_from_model_output(
                    *detached_data, None, None, "mapped"
                ),
                split,
                "mapped",
            )
        return (
            loss,
            {
                "loss": loss.item(),
                "pred": pred,
                "ground_truth": ground_truth,
                "where_data_are_observed": where_data_are_observed,
            },
        )

    def _warm_start_imputation(
        self,
        split: str,
        data: Tensor,
        ground_truth: Tensor,
        where_data_are_observed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.hparams.replace_nan_with == "simple":  # simple impute warm start
            # TODO[LOW]: this fails if the whole column is accidentally nan as part of the amputation process
            fn = simple_impute_tensor
            kwargs = {
                "where_data_are_observed": where_data_are_observed,
                "ctn_col_idxs": self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="continuous",
                ),
                "bin_col_idxs": self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="binary",
                ),
                "onehot_group_idxs": self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="onehot",
                ),
            }
        else:  # Replace nans with a single value provided
            fn = nan_to_num
            kwargs = {"nan": self.hparams.replace_nan_with}

        # apply
        data = fn(data, **kwargs)
        if split != "predict":
            ground_truth = fn(ground_truth, **kwargs)
        return (data, ground_truth)

    def shared_logging_step_end(self, outputs: Dict[str, float], split: str):
        """Log metrics + loss at end of step.
        Compatible with dp mode: https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#classification-metrics.
        """
        # Log loss
        self.log(
            IMPUTE_METRIC_TAG_FORMAT.format(
                name="loss",
                feature_space=self.hparams.data_feature_space,
                filter_subgroup="all",
                reduction="NA",
                split=split,
            ),
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )

        self.metric_logging_step(
            outputs["pred"],
            outputs["ground_truth"],
            outputs["where_data_are_observed"],
            split,
            "original",
        )

    def metric_logging_step(
        self,
        pred: Tensor,
        true: Tensor,
        where_data_are_observed: Tensor,
        split: str,
        data_feature_space: str,
    ):
        """
        Log all metrics.
        Order of keys:
        split -> filter_subgroup -> reduction -> (? feature space) -> name -> fn
        https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html#logging-torchmetrics
        # NOTE: if you add too many metrics it will mess up the progress bar
        """
        for filter_subgroup, split_moduledict in self.metrics[
            f"{split}_metrics"
        ].items():
            for reduction, moduledict in split_moduledict.items():
                # EW is reduction -> metric
                # CW is reduction -> feature space -> metric
                if data_feature_space in moduledict:
                    moduledict = moduledict[data_feature_space]
                for name, metric in moduledict.items():
                    context = {  # listed in order of metric dict keys for reference
                        "split": split,
                        "filter_subgroup": filter_subgroup,
                        "feature_space": data_feature_space,
                        "reduction": reduction,
                    }

                    log_settings = {
                        "on_step": False,
                        "on_epoch": True,
                        "prog_bar": False,
                        "logger": True,
                        "rank_zero_only": True,
                    }
                    # I need to compute the metric in a separate line from logging
                    # Ref: https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html#common-pitfalls
                    if (
                        filter_subgroup == "missingonly"
                        and (
                            where_data_are_missing := ~(where_data_are_observed.bool())
                        ).any()
                    ):  # Compute metrics for missing only data
                        metric_val = metric(pred, true, where_data_are_missing)
                    else:
                        metric_val = metric(pred, true)

                    self.log(
                        IMPUTE_METRIC_TAG_FORMAT.format(name=name, **context),
                        metric_val,
                        **log_settings,
                    )

    #######################
    #   Initialization    #
    #######################
    def init_metrics(self, metrics):
        """
        https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-and-devices
        https://lightning.ai/forums/t/lightningmodule-init-vs-setup-method/147
        * torchmetrics and models themselves need to be initialized inside __init__ and inside moduledict/list if i want the internal states to be placed on the correct device
        * with plightning, metric.reset() is called at the end of an epoch for me
        * I need a separate metric per dataset split (https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html#logging-torchmetrics)
        """
        if metrics is None:
            # separate metrics
            # split -> filter subgroup -> reduction -> (?feature space) -> name -> fn
            self.metrics = nn.ModuleDict(
                {
                    split: nn.ModuleDict(
                        {
                            filter_subgroup: self.get_reduction_metrics()
                            for filter_subgroup in ["all", "missingonly"]
                        }
                    )
                    for split in ["train_metrics", "val_metrics", "test_metrics"]
                }
            )

        else:  # WARNING: if the metrics aren't initialized inside __init__ they may not be put on the correct device
            self.metrics = metrics

    def get_reduction_metrics(self) -> nn.ModuleDict:
        """RMSE, MAAPE, (?Accuracy) x (?{original, mapped}) x {cw, ew}"""
        cwmetric_names = [("RMSE", RMSEMetric), ("MAAPE", MAAPEMetric)]
        # we need separate metrics for each feature map for cw metrics
        # since they depend on the number of features (which differs bc mapping)
        feature_spaces = (
            ["original", "mapped"]
            if self.hparams.data_feature_space == "mapped"
            else ["original"]
        )

        cwmetrics = nn.ModuleDict(
            {
                feature_space: nn.ModuleDict(
                    {
                        name: metric(
                            columnwise=True,
                            nfeatures=self.hparams.nfeatures[feature_space],
                        )
                        for name, metric in cwmetric_names
                    }
                )
                for feature_space in feature_spaces
            }
        )
        # Add accuracy for number of bins correctly imputed if everything is discretized
        # we don't care about element-wise categorical Accuracy
        if self.hparams.feature_map == "discretize_continuous":
            for feature_space in feature_spaces:
                cwmetrics[feature_space].update(
                    nn.ModuleDict(
                        {
                            "Accuracy": AccuracyMetric(
                                self.get_col_idxs_by_type(
                                    data_feature_space=feature_space,
                                    feature_type="binary",
                                ),
                                self.get_col_idxs_by_type(
                                    data_feature_space=feature_space,
                                    feature_type="onehot",
                                ),
                                columnwise=True,
                            ),
                        }
                    )
                )
        ewmetrics = nn.ModuleDict({"RMSE": RMSEMetric(), "MAAPE": MAAPEMetric()})
        # again, i can't even have ANY normal dicts here, it all has to be nn.moduledict/list, even wrappers
        # EW: reduction -> moduledict(name -> fn)
        # CW: reduction -> moduledict(feature_space -> moduledict(name -> fun))
        return nn.ModuleDict({"CW": cwmetrics, "EW": ewmetrics})

    def set_col_idxs_by_type_as_buffers(
        self, col_idxs_by_type: Dict[str, Dict[str, List]]
    ):
        # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#transferring-tensors-to-device
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/8
        # Register_buffer will save to state_dict, construct tensor directly on device
        # but they're not trainable params
        for data_feature_space, col_idxs in col_idxs_by_type.items():
            for feature_type, idxs in col_idxs.items():
                self.register_buffer(
                    COL_IDXS_BY_TYPE_FORMAT.format(
                        data_feature_space=data_feature_space, feature_type=feature_type
                    ),
                    list_to_tensor(idxs),
                )
                # original/mapped -> {cat/ctn -> indices}

    def get_col_idxs_by_type(
        self,
        data_feature_space: str,
        feature_type: str,
        default_action: Optional[Callable] = None,
    ) -> Tensor:
        if default_action is not None:
            return getattr(
                self,
                COL_IDXS_BY_TYPE_FORMAT.format(
                    data_feature_space=data_feature_space, feature_type=feature_type
                ),
                default_action,
            )
        return getattr(
            self,
            COL_IDXS_BY_TYPE_FORMAT.format(
                data_feature_space=data_feature_space, feature_type=feature_type
            ),
        )

    def setup(self, stage: str):
        """
        Anything init/built here might not be put on the correct device.
        Logic is here since datamodule.setup() will be called in trainer.fit() before self.setup().
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        return

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["hidden_and_cell_state"] = self.hidden_and_cell_state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.hidden_and_cell_state = checkpoint["hidden_and_cell_state"]
        # grab inversion function again since we can only pickle the name
        self.set_feature_map_inversion(self.hparams.feature_map)

    def _model_creation(self) -> None:
        """Configure loss and build model."""
        # Requires lossn is set and if col indices set for CEMSE
        # if it's in one of them it should be in all of them
        # we dont need this in shared step bc we can tell from the batch, but in the setup steps we need to know
        self.loss = self.configure_loss()
        self.set_layer_dims()
        # number of layers will always be even because it's symmetric
        self.code_index = len(self.hparams.layer_dims) // 2
        self.build_encoder()
        self.build_decoder()

    def set_feature_map_inversion(self, feature_map: str):
        """Lambdas are not pickle-able for checkpointing, so dynamically set."""
        # Safe to assume "mapped" key exists for these feature maps
        if feature_map == "discretize_continuous":
            self.feature_map_inversion = (
                lambda data_tensor: invert_discretize_tensor_gpu(
                    data_tensor,
                    **get_invert_discretize_tensor_args(
                        self.hparams.discretizations["data"],
                        self.hparams.columns["original"],
                        self.device,
                    ),
                )
            )
        elif feature_map == "target_encode_categorical":
            self.feature_map_inversion = (
                lambda data_tensor: invert_target_encoding_tensor_gpu(
                    data_tensor,
                    **get_invert_target_encode_tensor_args(
                        self.hparams.inverse_target_encode_map["mapping"],
                        self.hparams.inverse_target_encode_map["ordinal_mapping"],
                        self.hparams.columns["mapped"],
                        self.hparams.columns["original"],
                        self.device,
                    ),
                )
            )
        else:
            self.feature_map_inversion = None

    def configure_optimizers(self):
        """Pick optimizer. PL Function called after model.setup()"""
        optim_choices = {
            "Adam": optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.l2_penalty,
            ),
            "SGD": optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.l2_penalty,
            ),
        }
        return optim_choices[self.hparams.optimn]

    def configure_loss(self):
        """
        Pick loss from options.
        All losses expect logits, do not sigmoid/softmax in the architecture.
        """
        loss_choices = {
            "BCE": nn.BCEWithLogitsLoss(),
            "MSE": nn.MSELoss(),
            "CEMSE": CtnCatLoss(
                self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="continuous",
                ),
                self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="binary",
                ),
                self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="onehot",
                ),
            ),
            "CEMAAPE": CtnCatLoss(
                self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="continuous",
                ),
                self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="binary",
                ),
                self.get_col_idxs_by_type(
                    data_feature_space=self.hparams.data_feature_space,
                    feature_type="onehot",
                ),
                loss_ctn=MAAPEMetric(),
            ),
        }
        assert (
            self.hparams.lossn in loss_choices
        ), f"Passed invalid loss name. Please choose among: {list(loss_choices.keys())}"

        loss = loss_choices[self.hparams.lossn]
        # Add KL Divergence for VAE Loss
        if self.hparams.variational:
            # Not supporting this for now...I don't think it makes sense?
            # cat_cols_idx = self.cat_cols_idx if self.hparams.lossn == "CEMSE" else None
            loss = ReconstructionKLDivergenceLoss(loss, None)
        return loss

    def set_layer_dims(self):
        # Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        assert len(self.hparams.hidden_layers) > -1, "Passed no hidden layers."
        assert (
            array(self.hparams.hidden_layers[: len(self.hparams.hidden_layers) // 2])
            == array(
                self.hparams.hidden_layers[len(self.hparams.hidden_layers) // 2 + 1 :][
                    ::-1
                ]
            )
        ).all(), "Hidden Layers must be symmetric."
        # if isinstance(hidden_layers[-1], int) or hidden_layers[0].is_integer():
        if isinstance(self.hparams.hidden_layers[-1], int):
            self.hparams.layer_dims = (
                [self.hparams.nfeatures[self.hparams.data_feature_space]]
                + [int(dim) for dim in self.hparams.hidden_layers]
                + [self.hparams.nfeatures[self.hparams.data_feature_space]]
            )
        else:  # assuming float, compute relative size of input
            self.hparams.layer_dims = (
                [self.hparams.nfeatures[self.hparams.data_feature_space]]
                # ceil: e.g.: input_dim = 4,  rel_size: 0.3 and 0.2 when rounded down give 0, so we always round up to the nearest integer (to at least 1).
                + [
                    ceil(
                        rel_size
                        * self.hparams.nfeatures[self.hparams.data_feature_space]
                    )
                    for rel_size in self.hparams.hidden_layers
                ]
                + [self.hparams.nfeatures[self.hparams.data_feature_space]]
            )

    def validate_args(self):
        #### Assertions ####
        if self.hparams.lossn == "CEMSE" or self.hparams.lossn == "CEMAAPE":
            assert self.hparams.col_idxs_by_type[
                self.hparams.data_feature_space
            ], "Failed to get indices of continuous and categorical columns. Likely failed to pass list of columns and list of continuous columns. This is required for CEMSE and CEMAAPE loss."
            assert (
                self.hparams.feature_map != "discretize_continuous"
            ), "Passed a loss of CEMSE or CEMAAPE but indicated you discretized the data. These cannot both happen (since if all the data is discretized you want to use BCE loss instead)."
        # for now do not support vae with categorical features
        # TODO: create VAE prior/likelihood for fully categorical features, or beta-div instead.
        if self.hparams.variational:
            assert (
                self.hparams.feature_map != "discretize_continuous"
            ), "Indicated you discretized the data and also wanted to use a variational autoencoder. These cannot be used together."
            # cat_cols are only fine if you're target encoding
            assert len(
                self.hparams.col_idxs_by_type["original"].get("categorical", [])
            ) == 0 or (
                self.hparams.feature_map == "target_encode_categorical"
            ), "Indicated you wanted to use a variational autoencoder and also have categorical features, but these cannot be used togther."
            assert (
                self.hparams.lossn != "CEMSE" and self.hparams.lossn != "CEMAAPE"
            ), "Indicated CEMSE / CEMAAPE loss which refers to mixed data loss, but also indicated you want to use a variational autoencoder which only support continuous features."

        if self.hparams.feature_map == "discretize_continuous":
            assert (
                self.hparams.columns["original"] is not None
            ), "The data is discretized. To invert the discretization and maintain order, we need the original column order."

        assert hasattr(
            nn, self.hparams.activation
        ), "Failed to choose a valid activation function. Please choose one of the activation functions from torch.nn module."

        if type(self.hparams.replace_nan_with) is str:
            assert (
                self.hparams.replace_nan_with == "simple"
            ), "Gave invalid choice to replace nan."

    #######################
    #    Build Network    #
    #######################
    def build_encoder(self):
        """Builds just the encoder layers in the autoencoder (first half).
        Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        """
        encoder_layers = []

        # should come before dropout
        if self.hparams.batchswap_corruption:
            encoder_layers.append(BatchSwapNoise(self.hparams.batchswap_corruption))
        if self.hparams.dropout_corruption:
            # Dropout as binomial corruption
            encoder_layers += [
                ResetSeed(self.hparams.seed),
                nn.Dropout(self.hparams.dropout_corruption),
            ]

        stop_at = self.code_index
        if self.hparams.variational:  # for fc_mu and fc_var, stop before code
            stop_at -= 1

        for i in range(stop_at):
            encoder_layers.append(
                self.select_layer_type(
                    self.hparams.layer_dims[i],
                    self.hparams.layer_dims[i + 1],
                    set_layer_bias=not self.hparams.batchnorm,
                )
            )
            if self.hparams.batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self.hparams.layer_dims[i + 1]))
            encoder_layers.append(self.select_activation())
            if self.hparams.dropout:
                encoder_layers += [
                    ResetSeed(self.hparams.seed),  # ensures reprodudicibility
                    nn.Dropout(self.hparams.dropout),
                ]
        # self.encoder = nn.Sequential(*encoder_layers)
        self.encoder = nn.ModuleList(encoder_layers)
        if self.hparams.variational:
            self.fc_mu = nn.Linear(
                self.hparams.layer_dims[self.code_index - 1],
                self.hparams.layer_dims[self.code_index],
            )
            self.fc_var = nn.Linear(
                self.hparams.layer_dims[self.code_index - 1],
                self.hparams.layer_dims[self.code_index],
            )

    def build_decoder(self):
        """Builds just the decoder layers in the autoencoder (second half).
        Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        """
        decoder_layers = []
        # -2: exclude the last layer (-1), and also account i,i+1 (-1)
        for i in range(self.code_index, len(self.hparams.layer_dims) - 2):
            decoder_layers.append(
                self.select_layer_type(
                    self.hparams.layer_dims[i],
                    self.hparams.layer_dims[i + 1],
                    set_layer_bias=not self.hparams.batchnorm,
                )
            )
            if self.hparams.batchnorm:
                decoder_layers.append(nn.BatchNorm1d(self.hparams.layer_dims[i + 1]))
            decoder_layers.append(self.select_activation())
            if self.hparams.dropout:
                decoder_layers += [
                    ResetSeed(self.hparams.seed),  # ensures reprodudicibility
                    nn.Dropout(self.hparams.dropout),
                ]
        decoder_layers.append(
            nn.Linear(self.hparams.layer_dims[-2], self.hparams.layer_dims[-1])
        )

        # will will NOT sigmoid/softmax here since our loss expects logits
        self.decoder = nn.ModuleList(decoder_layers)

    def select_layer_type(
        self, dim1: int, dim2: int, set_layer_bias: bool
    ) -> nn.Module:
        """LSTM/RNN if longitudinal, else Linear."""
        if self.hparams.longitudinal:
            return nn.LSTM(dim1, dim2, batch_first=True, bias=set_layer_bias)
        return nn.Linear(dim1, dim2, bias=set_layer_bias)

    def select_activation(self) -> nn.Module:
        kwargs = {"inplace": True} if self.hparams.activation == "ReLU" else {}
        return getattr(nn, self.hparams.activation)(**kwargs)
        if self.hparams.activation == "ReLU":
            return nn.ReLU(inplace=True)
        elif self.hparams.activation == "sigmoid":
            return nn.Sigmoid()
        else:  # TanH
            return nn.Tanh()

    #######################
    #       Helpers       #
    #######################
    def reparameterize(self, mu, logvar):  # mean of N(mu, var)  # ln variance
        """Reparameterization trick for VAE.

        https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
        If we assume that the latent var z ~ p(z | x) = N(mu, std_dev^2)
        A valid reparameterization is z = mu + std_dev*epsilon
            where epsilon is noise: eps ~ N(0,1)
        """
        # since we're using log of variance e^.5ln(var) = var^.5
        std_dev = exp(0.5 * logvar)
        # noise following N(0,1)
        epsilon = randn_like(std_dev, device=self.device)

        # reparameterize z = mu + std_dev*epsilon
        return mu + (std_dev * epsilon)

    def get_imputed_tensor_from_model_output(
        self,
        data: Tensor,
        reconstruct_batch: Tensor,
        ground_truth: Tensor,
        data_are_observed: Tensor,
        original_data: Optional[Tensor],
        original_ground_truth: Optional[Tensor],
        data_feature_space: str,  # original/mapped
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Assumes all tensors passed in are detached.
        This function should not mutate the inputs.
        1. Invert any feature mapping (if passing original data)
        2. sigmoid/softmax categorical columns only + threshold
        3. keep original values where it's not missing.
        """
        # don't modify the original one passed in when inverting/etc
        pred = reconstruct_batch.clone()
        # get unmapped versions of everything
        if original_data is not None and original_ground_truth is not None:
            if self.feature_map_inversion is not None:
                pred = self.feature_map_inversion(pred)
            data = original_data
            ground_truth = original_ground_truth
            # re-compute locations where data is observed
            data_are_observed = (~isnan(data)).bool()

        # Sigmoid/softmax and threshold but in original space
        pred = binary_column_threshold(
            pred,
            self.get_col_idxs_by_type(
                data_feature_space=data_feature_space,
                feature_type="binary",
                default_action=lambda: [],  # empty list if dne
            ),
            0.5,
        )  # do nothing if no "binary" cols (empty list [])
        pred = onehot_column_threshold(
            pred,
            self.get_col_idxs_by_type(
                data_feature_space=data_feature_space,
                feature_type="onehot",
                default_action=lambda: [],  # empty list if dne
            ),
        )  # do nothing if no "binary" cols (empty list [])

        # Keep original where it's not missing, otherwise fill with pred
        imputed = data.where(data_are_observed, pred)
        # If the original dataset contains nans (no fully observed), we need to fill in ground_truth too for the metric computation
        # potentially nan in different places than data if amputing (should do nothing if originally fully observed/amputing)
        ground_truth_are_observed = (~isnan(ground_truth)).bool()
        # keep where ground_truth is observed, otherwise fill with pred
        ground_truth = ground_truth.where(ground_truth_are_observed, pred)

        if imputed.isnan().sum():
            rank_zero_warn("NaNs still found in imputed data.")
        return imputed, ground_truth, data_are_observed

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
            "--variational",
            type=str2bool,
            default=False,
            help="Use a variational autoencoder.",
        )
        p.add_argument(
            "--dropout-corruption",
            type=float,
            default=None,
            help="If implementing a denoising autoencoder, what percentage of corruption at the input using dropout (noise is 0's).",
        )
        p.add_argument(
            "--batchswap-corruption",
            type=float,
            default=None,
            help="If implementing a denoising autoencoder, what percentage of corruption at the input, swapping out values as noise.",
        )
        p.add_argument(
            "--replace-nan-with",
            default=0,
            action=StringOrInt(str_choices=["simple"]),
            help="How to do warm-start to autoencoder imputation. Simple imputation or fill with an integer value.",
        )
        return p
