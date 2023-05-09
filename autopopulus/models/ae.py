from argparse import ArgumentParser
from math import ceil
import sys
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from torchmetrics import Metric
from numpy import stack

#### Pytorch ####
from torch import long as torch_long
from torch import exp, isnan, nan_to_num, randn_like, tensor, Tensor, device
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

## Lightning ##
import pytorch_lightning as pl

from autopopulus.models.utils import (
    CtnCatLoss,
    BatchSwapNoise,
    ReconstructionKLDivergenceLoss,
    ResetSeed,
    binary_column_threshold,
    onehot_column_threshold,
)
from autopopulus.data.transforms import (
    invert_target_encoding_tensor,
    invert_discretize_tensor,
    simple_impute_tensor,
)
from autopopulus.utils.impute_metrics import (
    CWRMSE,
    CWMAAPE,
    MAAPEMetric,
    RMSEMetric,
    categorical_accuracy,
)
from autopopulus.utils.cli_arg_utils import YAMLStringListToList, StringOrInt, str2bool
from autopopulus.utils.utils import rank_zero_print
from autopopulus.utils.log_utils import IMPUTE_METRIC_TAG_FORMAT
from autopopulus.data import CommonDataModule
from autopopulus.data.types import DataTypeTimeDim
from autopopulus.data.constants import PAD_VALUE

HiddenAndCellState = Tuple[Tensor, Tensor]
MuLogVar = Tuple[Tensor, Tensor]

LOSS_CHOICES = ["BCE", "MSE", "CEMSE", "CEMAAPE"]
OPTIM_CHOICES = ["Adam", "SGD"]


class AEDitto(pl.LightningModule):
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
        hidden_layers: List[Union[int, float]],
        learning_rate: float,
        seed: int,
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
        variational: bool = False,
        longitudinal: bool = False,  # Convenience on top of data type time dim
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        dropout: Optional[float] = None,
        dropout_corruption: Optional[float] = None,
        batchswap_corruption: Optional[float] = None,
        datamodule: Optional[CommonDataModule] = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.lossn = lossn
        self.optimn = optimn
        self.seed = seed
        self.lr = learning_rate
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.activation = activation
        self.init_metrics(metrics)
        # Other options
        self.replace_nan_with = replace_nan_with
        self.mvec = mvec
        self.variational = variational
        self.longitudinal = longitudinal
        self.data_type_time_dim = data_type_time_dim
        self.dropout_corruption = dropout_corruption
        self.batchswap_corruption = batchswap_corruption

        self.hidden_layers = hidden_layers
        self.n_layers = len(hidden_layers) + 1  # fencing problem +1

        # Required for serialization
        self.save_hyperparameters(ignore=["datamodule"])

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
        if self.variational:
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

        if self.variational:
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
        data_version = "mapped" if "mapped" in batch else "original"
        data, ground_truth = (
            batch[data_version]["data"],
            batch[data_version]["ground_truth"],
        )
        seq_len = (
            batch[data_version]["seq_len"] if "seq_len" in batch[data_version] else None
        )
        # set this before filling in data with replacement (if doing so)
        non_missing_mask = ~(isnan(data)).bool()

        ### Warm Start ###
        # replace nan in ground truth too if its missing any
        if self.replace_nan_with is not None:
            if self.replace_nan_with == "simple":  # simple impute warm start
                # TODO[LOW]: this fails if the whole column is accidentally nan as part of the amputation process
                fn = simple_impute_tensor
                kwargs = {
                    "non_missing_mask": non_missing_mask,
                    "ctn_col_idxs": self.col_idxs_by_type[data_version]["continuous"],
                    "bin_col_idxs": self.col_idxs_by_type[data_version]["binary"],
                    "onehot_group_idxs": self.col_idxs_by_type[data_version]["onehot"],
                }
            else:  # Replace nans with a single value provided
                fn = nan_to_num
                kwargs = {"nan": self.replace_nan_with}

            # apply
            data = fn(data, **kwargs)
            if split != "predict":
                ground_truth = fn(ground_truth, **kwargs)

        ### Model ###
        # pass through the sequence length (if they're none, nothing happens)
        reconstruct_batch = self(split, data, seq_len)
        if self.variational:  # unpack when vae
            reconstruct_batch, (mu, logvar) = reconstruct_batch
        else:
            reconstruct_batch = reconstruct_batch

        ### Loss ###
        if split != "predict":
            """
            Note: mvec will treat missing values as 0 (ignore in loss during training)
            data * mask: normalize by # all features (missing and observed).
            data[mask]: normalize by only # observed features. (Want)
            """
            if self.mvec:  # and self.training
                eval_pred = reconstruct_batch[non_missing_mask]
                eval_true = ground_truth[non_missing_mask]
            else:
                eval_pred = reconstruct_batch
                eval_true = ground_truth

            if self.variational:
                loss = self.loss(eval_pred, eval_true, mu, logvar)
            else:
                # NOTE: if no mvec and no vae for some reason it says the loss is modifying the output in place, the clone is a quick hack
                loss = self.loss(eval_pred.clone(), eval_true)

            # TODO: everything to cpu here?
            # save mapped outputs for evaluation
            if "mapped" in batch:  # test not empty or None
                self.metric_logging_step(
                    reconstruct_batch.detach(),
                    ground_truth.detach(),
                    non_missing_mask.detach(),
                    split,
                    "mapped",
                )

        (
            pred,
            ground_truth,
            non_missing_mask,
        ) = self.get_imputed_tensor_from_model_output(
            data.detach().cpu().float(),
            reconstruct_batch.detach().cpu().float(),
            ground_truth.detach().cpu().float(),
            non_missing_mask.detach().cpu().float(),
            batch["original"]["data"].detach().cpu().float(),
            batch["original"]["ground_truth"].detach().cpu().float(),
        )

        return (
            (
                loss,
                {
                    "loss": loss.item(),
                    "pred": pred,
                    "ground_truth": ground_truth,
                    "non_missing_mask": non_missing_mask,
                },
            )
            if split != "predict"
            else pred  # Just get the predictions there's no loss/etc
        )

    def shared_logging_step_end(self, outputs: Dict[str, float], split: str):
        """Log metrics + loss at end of step.
        Compatible with dp mode: https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#classification-metrics.
        """
        # Log loss
        self.log(
            IMPUTE_METRIC_TAG_FORMAT.format(
                name="loss",
                feature_space="original",
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
            outputs["non_missing_mask"],
            split,
            "original",
        )

    def metric_logging_step(
        self,
        pred: Tensor,
        true: Tensor,
        non_missing_mask: Tensor,
        split: str,
        ismapped: str,
    ):
        """Log all metrics whether cat/ctn."""
        # NOTE: if you add too many metrics it will mess up the progress bar
        for metric in self.metrics:
            if metric["name"] == "Accuracy":
                metric["fn"] = metric["fn"](
                    self.col_idxs_by_type[ismapped]["binary"],
                    self.col_idxs_by_type[ismapped]["onehot"],
                )
            context = {
                "split": split,
                "feature_space": ismapped,
                "reduction": metric["reduction"],
            }

            log_settings = {
                "on_step": False,
                "on_epoch": True,
                "prog_bar": False,
                "logger": True,
                "rank_zero_only": True,
            }

            self.log(
                IMPUTE_METRIC_TAG_FORMAT.format(
                    name=metric["name"], filter_subgroup="all", **context
                ),
                metric["fn"](pred, true),
                **log_settings,
            )
            # Compute metrics for missing only data
            missing_only_mask = ~(non_missing_mask.bool())
            if missing_only_mask.any():
                self.log(
                    IMPUTE_METRIC_TAG_FORMAT.format(
                        name=metric["name"], filter_subgroup="missingonly", **context
                    ),
                    metric["fn"](pred, true, missing_only_mask),
                    **log_settings,
                )

    #######################
    #   Initialization    #
    #######################
    def init_metrics(
        self,
        metrics: Optional[  # SEPARATE FROM LOSS, only for evaluation
            List[
                Dict[  # NOTE: Any should be ... (kwargs) but not supported yet
                    str,
                    Union[str, Union[Metric, Callable[[Tensor, Tensor, Any], Tensor]]],
                ]
            ]
        ],
    ):
        """
        Load the default, but they need to be initialized inside the module (during lightnignmodule.__init__).
        Ref: https://github.com/Lightning-AI/lightning/issues/4909#issuecomment-736645699
        cw/ew are separate bc cw are only implement in functions, not torchmetric modules.
        torchmetrics need to be initialized inside moduledict/list if i want the internal states to be placed on the correct device
        """
        if metrics is None:
            ewmetrics = nn.ModuleDict({"RMSE": RMSEMetric(), "MAAPE": MAAPEMetric()})
            self.metrics = [
                {"name": "RMSE", "fn": CWRMSE, "reduction": "CW"},
                {"name": "MAAPE", "fn": CWMAAPE, "reduction": "CW"},
            ] + [{"name": k, "fn": v, "reduction": "EW"} for k, v in ewmetrics.items()]
        else:
            self.metrics = metrics

    def setup(self, stage: str):
        """Might update the columns after initialization of AEDitto but before calling fit on wrapper class, so check right before calling fit here.
        This needs to happen on setup, so pl calls it before configure optimizers.
        Setup is useful for dynamically building models.
        Logic is here since datamodule.setup() will be called in trainer.fit() before self.setup().
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == "fit" or stage == "train":
            self.hidden_and_cell_state = {}
            self.set_args_from_data()
            self._setup_logic()
            self.validate_args()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["data_attributes"] = {
            "nfeatures": self.nfeatures,
            "groupby": self.groupby,
            "columns": self.columns,
            "discretizations": self.discretizations,
            "inverse_target_encode_map": self.inverse_target_encode_map,
            "col_idxs_by_type": self.col_idxs_by_type,
            # cannot pickle lambda fns, save the feature_map name instead
            "feature_map": self.datamodule.feature_map,
        }
        checkpoint["hidden_and_cell_state"] = self.hidden_and_cell_state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # super().on_load_checkpoint(checkpoint)
        for name, value in checkpoint["data_attributes"].items():
            setattr(self, name, value)
        self.hidden_and_cell_state = checkpoint["hidden_and_cell_state"]
        # grab inversion function again since we can only pickle the name
        self.set_feature_map_inversion(checkpoint["data_attributes"]["feature_map"])
        # don't need to validate args since it should have been validated before serialization.
        self._setup_logic()

    def _setup_logic(self) -> None:
        # Requires lossn is set and if col indices set for CEMSE
        # if it's in one of them it should be in all of them
        # we dont need this in shared step bc we can tell from the batch, but in the setup steps we need to know
        self.data_version = "mapped" if "mapped" in self.groupby else "original"
        self.loss = self.configure_loss()
        self.set_layer_dims()
        # number of layers will always be even because it's symmetric
        self.code_index = len(self.layer_dims) // 2
        self.build_encoder()
        self.build_decoder()

    @staticmethod
    def _idxs_to_tensor(
        idxs: Union[List[int], List[List[int]]],
        device: Union[str, device] = device("cpu"),
    ) -> Tensor:
        """
        Converts col_idxs_by_type indices into a tensor.
        If List[List[int]] there may uneven length of group indices,
        which produces ValueError, so this tensor needs to be padded.
        """
        try:
            if len(idxs) > 1:  # Creating tensor from list of arrays is slow
                idxs = stack(idxs, axis=0)
            return tensor(idxs, dtype=torch_long, device=device)
        except ValueError:  # pad when onehot list of group indices
            return nn.utils.rnn.pad_sequence(
                [
                    tensor(group_idxs, dtype=torch_long, device=device)
                    for group_idxs in idxs
                ],
                batch_first=True,  # so they function as rows
                padding_value=PAD_VALUE,
            )

    def set_args_from_data(self):
        """
        Set model info that we can only dynamically get from the data after it's been setup() in trainer.fit().
        Any attribute set here should be serialized in `on_save_checkpoint`.
        NOTE: If there's onehot in col_idxs_by_type, it will be PADDED .
        Anything that uses it will need to account for that.
        """
        self.nfeatures = self.datamodule.nfeatures
        self.groupby = self.datamodule.groupby
        self.columns = self.datamodule.columns
        self.discretizations = self.datamodule.discretizations
        self.inverse_target_encode_map = self.datamodule.inverse_target_encode_map

        # used for BCE+MSE los, -> cat cols the VAE Loss,etc which require tensor
        # We still need this if we're loading the ae from a file and not calling fit
        # enforce Dtype = Long
        self.col_idxs_by_type = {
            data_version: {
                feature_type: self._idxs_to_tensor(idxs, self.device)
                for feature_type, idxs in col_idxs.items()
            }
            # original/mapped -> {cat/ctn -> indices}
            for data_version, col_idxs in self.datamodule.col_idxs_by_type.items()
        }
        self.set_feature_map_inversion(self.datamodule.feature_map)

    def set_feature_map_inversion(self, feature_map: str):
        """Lambdas are not pickle-able for checkpointing, so dynamically set."""
        # Add accuracy for number of bins correctly imputed if everything is discretized
        # Safe to assume "mapped" key exists for these feature maps
        if feature_map == "discretize_continuous":
            self.metrics.append(
                {"name": "Accuracy", "fn": categorical_accuracy, "reduction": "CW"}
            )
            self.feature_map_inversion = lambda data_tensor: invert_discretize_tensor(
                data_tensor,
                self.groupby["mapped"]["discretized_ctn_cols"]["data"],
                self.discretizations["data"],
                self.columns["original"],
            )
        elif feature_map == "target_encode_categorical":
            groupby = (
                self.groupby["mapped"]["combined_onehots"]["data"]
                if "combined_onehots" in self.groupby["mapped"]
                else None
            )
            self.feature_map_inversion = (
                lambda data_tensor: invert_target_encoding_tensor(
                    data_tensor,
                    self.inverse_target_encode_map,
                    self.columns["mapped"],
                    self.columns["original"],
                    groupby,
                )
            )
        else:
            self.feature_map_inversion = None

    def configure_optimizers(self):
        """Pick optimizer. PL Function called after model.setup()"""
        optim_choices = {
            "Adam": optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.l2_penalty
            ),
            "SGD": optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.l2_penalty,
            ),
        }
        return optim_choices[self.optimn]

    def configure_loss(self):
        """
        Pick loss from options.
        All losses expect logits, do not sigmoid/softmax in the architecture.
        """
        loss_choices = {
            "BCE": nn.BCEWithLogitsLoss(),
            "MSE": nn.MSELoss(),
            "CEMSE": CtnCatLoss(
                self.col_idxs_by_type[self.data_version]["continuous"],
                self.col_idxs_by_type[self.data_version]["binary"],
                self.col_idxs_by_type[self.data_version]["onehot"],
            ),
            "CEMAAPE": CtnCatLoss(
                self.col_idxs_by_type[self.data_version]["continuous"],
                self.col_idxs_by_type[self.data_version]["binary"],
                self.col_idxs_by_type[self.data_version]["onehot"],
                loss_ctn=MAAPEMetric(),
            ),
        }
        assert (
            self.lossn in loss_choices
        ), f"Passed invalid loss name. Please choose among: {list(loss_choices.keys())}"

        loss = loss_choices[self.lossn]
        # Add KL Divergence for VAE Loss
        if self.variational:
            # Not supporting this for now...I don't think it makes sense?
            # cat_cols_idx = self.cat_cols_idx if self.lossn == "CEMSE" else None
            loss = ReconstructionKLDivergenceLoss(loss, None)
        return loss

    def set_layer_dims(self):
        # Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        assert len(self.hidden_layers) > -1, "Passed no hidden layers."
        # if isinstance(hidden_layers[-1], int) or hidden_layers[0].is_integer():
        if isinstance(self.hidden_layers[-1], int):
            self.layer_dims = (
                [self.nfeatures[self.data_version]]
                + [int(dim) for dim in self.hidden_layers]
                + [self.nfeatures[self.data_version]]
            )
        else:  # assuming float, compute relative size of input
            self.layer_dims = (
                [self.nfeatures[self.data_version]]
                # ceil: e.g.: input_dim = 4,  rel_size: 0.3 and 0.2 when rounded down give 0, so we always round up to the nearest integer (to at least 1).
                + [
                    ceil(rel_size * self.nfeatures[self.data_version])
                    for rel_size in self.hidden_layers
                ]
                + [self.nfeatures[self.data_version]]
            )

    def validate_args(self):
        #### Assertions ####
        if self.lossn == "CEMSE" or self.lossn == "CEMAAPE":
            assert self.col_idxs_by_type[
                self.data_version
            ], "Failed to get indices of continuous and categorical columns. Likely failed to pass list of columns and list of continuous columns. This is required for CEMSE and CEMAAPE loss."
            assert (
                self.datamodule.feature_map != "discretize_continuous"
            ), "Passed a loss of CEMSE or CEMAAPE but indicated you discretized the data. These cannot both happen (since if all the data is discretized you want to use BCE loss instead)."
        # for now do not support vae with categorical features
        # TODO: create VAE prior/likelihood for fully categorical features, or beta-div instead.
        if self.variational:
            assert (
                self.datamodule.feature_map != "discretize_continuous"
            ), "Indicated you discretized the data and also wanted to use a variational autoencoder. These cannot be used together."
            # cat_cols are only fine if you're target encoding
            assert len(
                self.col_idxs_by_type["original"].get("categorical", [])
            ) == 0 or (
                self.datamodule.feature_map == "target_encode_categorical"
            ), "Indicated you wanted to use a variational autoencoder and also have categorical features, but these cannot be used togther."
            assert (
                self.lossn != "CEMSE" and self.lossn != "CEMAAPE"
            ), "Indicated CEMSE / CEMAAPE loss which refers to mixed data loss, but also indicated you want to use a variational autoencoder which only support continuous features."

        if self.datamodule.feature_map == "discretize_continuous":
            assert (
                self.columns["original"] is not None
            ), "The data is discretized. To invert the discretization and maintain order, we need the original column order."

        assert hasattr(
            nn, self.activation
        ), "Failed to choose a valid activation function. Please choose one of the activation functions from torch.nn module."

        if type(self.replace_nan_with) is str:
            assert (
                self.replace_nan_with == "simple"
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
        if self.batchswap_corruption:
            encoder_layers.append(BatchSwapNoise(self.batchswap_corruption))
        if self.dropout_corruption:
            # Dropout as binomial corruption
            encoder_layers += [
                ResetSeed(self.seed),
                nn.Dropout(self.dropout_corruption),
            ]

        stop_at = self.code_index
        if self.variational:  # for fc_mu and fc_var, stop before code
            stop_at -= 1

        for i in range(stop_at):
            encoder_layers.append(
                self.select_layer_type(self.layer_dims[i], self.layer_dims[i + 1])
            )
            encoder_layers.append(self.select_activation())
            if self.dropout:
                encoder_layers += [
                    ResetSeed(self.seed),  # ensures reprodudicibility
                    nn.Dropout(self.dropout),
                ]
        # self.encoder = nn.Sequential(*encoder_layers)
        self.encoder = nn.ModuleList(encoder_layers)
        if self.variational:
            self.fc_mu = nn.Linear(
                self.layer_dims[self.code_index - 1], self.layer_dims[self.code_index]
            )
            self.fc_var = nn.Linear(
                self.layer_dims[self.code_index - 1], self.layer_dims[self.code_index]
            )

    def build_decoder(self):
        """Builds just the decoder layers in the autoencoder (second half).
        Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        """
        decoder_layers = []
        # -2: exclude the last layer (-1), and also account i,i+1 (-1)
        for i in range(self.code_index, len(self.layer_dims) - 2):
            decoder_layers.append(
                self.select_layer_type(self.layer_dims[i], self.layer_dims[i + 1])
            )
            decoder_layers.append(self.select_activation())
            if self.dropout:
                decoder_layers += [
                    ResetSeed(self.seed),  # ensures reprodudicibility
                    nn.Dropout(self.dropout),
                ]
        decoder_layers.append(nn.Linear(self.layer_dims[-2], self.layer_dims[-1]))

        # will will NOT sigmoid/softmax here since our loss expects logits
        self.decoder = nn.ModuleList(decoder_layers)

    def select_layer_type(self, dim1: int, dim2: int) -> nn.Module:
        """LSTM/RNN if longitudinal, else Linear."""
        if self.longitudinal:
            return nn.LSTM(dim1, dim2, batch_first=True)
        return nn.Linear(dim1, dim2)

    def select_activation(self) -> nn.Module:
        kwargs = {"inplace": True} if self.activation == "ReLU" else {}
        return getattr(nn, self.activation)(**kwargs)
        if self.activation == "ReLU":
            return nn.ReLU(inplace=True)
        elif self.activation == "sigmoid":
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
        non_missing_mask: Tensor,
        original_data: Optional[Tensor],
        original_ground_truth: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        1. Invert any feature mapping
        2. sigmoid/softmax categorical columns only + threshold
        3. keep original values where it's not missing.
        """
        # get unmapped versions of everything
        if original_data is not None and original_ground_truth is not None:
            if self.feature_map_inversion is not None:
                reconstruct_batch = self.feature_map_inversion(reconstruct_batch)
            data = original_data
            ground_truth = original_ground_truth
            # re-compute non_missing_mask
            non_missing_mask = (~isnan(data)).bool()

        # Sigmoid/softmax and threshold but in original space
        reconstruct_batch = binary_column_threshold(
            reconstruct_batch, self.col_idxs_by_type["original"].get("binary", []), 0.5
        )  # do nothing if no "binary" cols (empty list [])
        reconstruct_batch = onehot_column_threshold(
            reconstruct_batch, self.col_idxs_by_type["original"].get("onehot", [])
        )  # do nothing if no "binary" cols (empty list [])

        # Keep original where it's not missing
        imputed = data.where(non_missing_mask, reconstruct_batch)
        # If the original dataset contains nans (no fully observed), we need to fill in ground_truth too for the metric computation
        # potentially nan in different places than data if amputing (should do nothing if originally fully observed/amputing)
        ground_truth_non_missing_mask = (~isnan(ground_truth)).bool()
        ground_truth = ground_truth.where(
            ground_truth_non_missing_mask, reconstruct_batch
        )

        if imputed.isnan().sum():
            rank_zero_print("WARNING: NaNs still found in imputed data.")
        return imputed, ground_truth, non_missing_mask

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
