from argparse import ArgumentParser
from math import ceil
import sys
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from torchmetrics import Metric
from numpy import ndarray
from warnings import warn

#### Pytorch ####
from torch import long as torch_long
from torch import exp, isnan, nan_to_num, randn_like, tensor, Tensor, zeros
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

## Lightning ##
import pytorch_lightning as pl

from autopopulus.models.utils import (
    BCEMSELoss,
    BatchSwapNoise,
    ReconstructionKLDivergenceLoss,
    ResetSeed,
)
from autopopulus.data.transforms import (
    sigmoid_cat_cols,
    undiscretize_tensor,
    simple_impute_tensor,
)
from autopopulus.utils.impute_metrics import AccuracyPerBin, RMSE, MAAPE
from autopopulus.utils.utils import flatten_groupby
from autopopulus.utils.cli_arg_utils import YAMLStringListToList, str2bool
from autopopulus.data import CommonDataModule
from autopopulus.data.types import DataTypeTimeDim

HiddenAndCellState = Tuple[Tensor, Tensor]
MuLogVar = Tuple[Tensor, Tensor]

DEFAULT_METRICS = {
    "RMSE": RMSE,
    "MAAPE": MAAPE,
}


LOSS_CHOICES = ["BCE", "MSE", "BCEMSE"]
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
            - variational via vae flag.
        - {static, longitudinal}
        - {dropout, no dropout}
        - warm start {0, simple (mean/mode)}
        - losses {mvec, all} x {BCE, MSE, BCE (CAT) + MSE (CTN), reconstruction+kldivergence, accuracyPerBin (discrete version only)}
            - BCE is always with logits.
            - reconstruction+kldivergence is always when vae flag is chosen.
            - mvec ensures loss is only computed on originally non missing data
        TODO[LOW]: new accuracy metrics for ctn and cat separately: https://walkwithfastai.com/tab.ae
    """

    def __init__(
        self,
        hidden_layers: List[Union[int, float]],
        learning_rate: float,
        seed: int,
        l2_penalty: float = 0,
        lossn: str = "BCEMSE",
        optimn: str = "Adam",
        activation: str = "ReLU",
        metrics: Dict[  # SEPARATE FROM LOSS, only for evaluation
            # NOTE: Any should be ... (kwargs) but not supported yet
            str,
            Union[Metric, Callable[[Tensor, Tensor, Any], Tensor]],
        ] = DEFAULT_METRICS,
        mvec: bool = False,
        vae: bool = False,
        longitudinal: bool = False,  # Convenience on top of data type time dim
        data_type_time_dim: DataTypeTimeDim = DataTypeTimeDim.STATIC,
        undiscretize_data: bool = False,
        replace_nan_with: Optional[
            Union[int, str]
        ] = None,  # Only used if not undiscretizing
        dropout: Optional[float] = None,
        dropout_corruption: Optional[float] = None,
        batchswap_corruption: Optional[float] = None,
        groupby: Optional[Dict[int, int]] = None,
        discretizations: Optional[
            Dict[str, Union[List[Tuple[float, float]], List[int]]]
        ] = None,  # orig col name to: list bin ranges, list indices
        original_columns: Optional[List[str]] = None,
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
        self.metrics = metrics
        # Other options
        self.mvec = mvec
        self.vae = vae
        self.longitudinal = longitudinal
        self.data_type_time_dim = data_type_time_dim
        self.undiscretize_data = undiscretize_data
        self.replace_nan_with = replace_nan_with
        self.dropout_corruption = dropout_corruption
        self.batchswap_corruption = batchswap_corruption

        # used for undiscretize, accuracyperbin
        self.groupby = groupby
        self.discretizations = discretizations
        self.original_columns = original_columns
        self.hidden_layers = hidden_layers

        # Add accuracy for number of bins correctly imputed if everything is discretized
        if self.undiscretize_data:
            self.metrics["AccuracyPerBin"] = AccuracyPerBin

        # Required for serialization
        self.save_hyperparameters()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return None
        return (
            Variable(zeros(1, 1, self.hidden_dim)),
            Variable(zeros(1, 1, self.hidden_dim)),
        )

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
        if self.vae:
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
        for layer in self.encoder:
            X = self.apply_layer(split, layer, X, orig_seq_lens)

        if self.vae:
            mu, logvar = self.fc_mu(X), self.fc_var(X)
            return self.reparameterize(mu, logvar), (mu, logvar)

        return X

    def decode(
        self,
        split: str,
        X: Tensor,
        orig_seq_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, HiddenAndCellState]:
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
        rnn_output_packed, self.prev_hidden_and_cell_state[split] = rnn_layer(
            X_packed, self.prev_hidden_and_cell_state[split]
        )
        # undo the packing operation
        # lstm_output, lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # grab last hidden state
        # return hidden[-1]
        return rnn_output_packed

    #######################
    # Training/Eval Logic #
    #######################
    # Ref: https://pytorch-lightning.readthedocs.io/en/stable/advanced/sequences.html
    # For (h,c) passthrough (currently not using, since validation doesn't accept hiddens)
    def training_step(self, batch, batch_idx):
        self.prev_hidden_and_cell_state["train"] = self.init_hidden()
        loss, outputs = self.shared_step(batch, "train")
        self.shared_logging_step_end(outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.prev_hidden_and_cell_state["val"] = self.init_hidden()
        loss, outputs = self.shared_step(batch, "val")
        self.shared_logging_step_end(outputs, "val")
        return loss

    def test_step(self, batch, batch_idx):
        self.prev_hidden_and_cell_state["test"] = self.init_hidden()
        loss, outputs = self.shared_step(batch, "test")
        self.shared_logging_step_end(outputs, "test")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this will reinitialize for every predict
        self.prev_hidden_and_cell_state["predict"] = self.init_hidden()
        return self.shared_step(batch, "predict")

    def shared_step(self, batch, split: str) -> Tuple[float, Dict[str, float]]:
        if self.undiscretize_data:
            # unpack seq length here
            if self.longitudinal:
                (
                    (data, ground_truth, seq_len),
                    (
                        undiscretized_data,
                        undiscretized_ground_truth,
                        undiscretized_seq_len,
                    ),
                ) = batch
            else:
                (
                    data,
                    ground_truth,
                    undiscretized_data,
                    undiscretized_ground_truth,
                ) = batch
                seq_len = None
        else:
            if self.longitudinal:
                data, ground_truth, seq_len = batch
            else:
                data, ground_truth = batch
                seq_len = None
            undiscretized_data = None
            undiscretized_ground_truth = None

        # set this before filling in data with replacement (if doing so)
        non_missing_mask = ~(isnan(data)).bool()

        if self.replace_nan_with is not None:
            # replace nan in ground truth too if its missing any
            if self.replace_nan_with == "simple":  # simple impute warm start
                # TODO[LOW]: this fails if the whole column is accidentally nan as part of the amputation process
                data = simple_impute_tensor(data, self.ctn_cols_idx, self.cat_cols_idx)
                if split != "predict":
                    ground_truth = simple_impute_tensor(
                        ground_truth, self.ctn_cols_idx, self.cat_cols_idx
                    )
            else:  # Replace nans with a single value provided
                data = nan_to_num(data, nan=self.replace_nan_with)
                if split != "predict":
                    ground_truth = nan_to_num(ground_truth, nan=self.replace_nan_with)

        # pass through the sequence length (if they're none, nothing happens)
        reconstruct_batch = self(split, data, seq_len)
        if self.vae:  # unpack when vae
            reconstruct_batch, (mu, logvar) = reconstruct_batch
        else:
            reconstruct_batch = reconstruct_batch

        # Detach hidden state for next step
        # ref: https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384/7
        if self.prev_hidden_and_cell_state[split] is not None:
            h, c = self.prev_hidden_and_cell_state[split]
            self.prev_hidden_and_cell_state[split] = (h.detach(), c.detach())

        """
        Note: mvec will treat missing values as 0 (ignore in loss during training)
        data * mask: normalize by # all features (missing and observed).
        data[mask]: normalize by only # observed features. (Want)
        """
        eval_pred = (
            reconstruct_batch[non_missing_mask]
            if self.mvec  # and self.training
            else reconstruct_batch
        )
        eval_true = (
            ground_truth[non_missing_mask]
            if self.mvec  # and self.training
            else ground_truth
        )

        if split != "predict":
            if self.vae:
                loss = self.loss(eval_pred, eval_true, mu, logvar)
            else:
                # NOTE: if no mvec and no vae for some reason it says the loss is modifying the output in place, the clone is a quick hack
                loss = self.loss(eval_pred.clone(), eval_true)
                # loss = self.loss(eval_pred, eval_true)

            # save discretized outputs for evaluation
            if (
                undiscretized_data is not None
                and undiscretized_ground_truth is not None
            ):
                self.pre_undiscretize_logging_step(
                    reconstruct_batch.detach(),
                    ground_truth.detach(),
                    (~isnan(undiscretized_data)).bool(),
                    split,
                )

        (
            pred,
            ground_truth,
            non_missing_mask,
        ) = self.get_imputed_tensor_from_model_output(
            data,
            reconstruct_batch,
            ground_truth,
            non_missing_mask,
            undiscretized_data,
            undiscretized_ground_truth,
        )  # TODO: undiscretized DNE if not undiscretizing???

        return (
            (
                loss,
                {
                    "loss": loss.item(),
                    "pred": pred.detach(),
                    "ground_truth": ground_truth.detach(),
                    "non_missing_mask": non_missing_mask.detach(),
                },
            )
            if split != "predict"
            else pred.detach()  # Just get the predictions there's no loss/etc
        )

    def shared_logging_step_end(self, outputs: Dict[str, float], split: str):
        """Log metrics + loss at end of step.
        Compatible with dp mode: https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#classification-metrics."""
        # Log loss
        self.log(
            f"AE/{self.data_type_time_dim.name}/{split}-loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # NOTE: if you add too many metrics it will mess up the progress bar
        # Log all metrics
        for name, metricfn in self.metrics.items():
            if name == "AccuracyPerBin":  # skip, this is dealt with separately
                continue

            self.log(
                f"impute/{self.data_type_time_dim.name}/{split}-{name}",
                metricfn(outputs["pred"], outputs["ground_truth"]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            # Compute metrics for missing only data
            missing_only_mask = ~(outputs["non_missing_mask"].bool())
            if missing_only_mask.any():
                self.log(
                    f"impute/{self.data_type_time_dim.name}/{split}-{name}-missingonly",
                    metricfn(
                        outputs["pred"],
                        outputs["ground_truth"],
                        missing_only_mask,
                    ),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

    def pre_undiscretize_logging_step(
        self,
        pred_disc: Tensor,
        ground_truth_disc: Tensor,
        non_missing_mask: Tensor,
        split: str,
    ):
        """Log AccuracyPerBin before undiscretizing."""
        name = "AccuracyPerBin"
        if name in self.metrics:
            flattened_groupby = flatten_groupby(self.groupby)
            self.log(
                f"impute/{self.data_type_time_dim.name}/{split}-{name}",
                self.metrics[name](pred_disc, ground_truth_disc, flattened_groupby),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            # Compute metrics for missing only data
            missing_only_mask = ~(non_missing_mask.bool())
            if missing_only_mask.any():
                self.log(
                    f"impute/{self.data_type_time_dim.name}/{split}-{name}-missingonly",
                    self.metrics[name](
                        pred_disc,
                        ground_truth_disc,
                        flattened_groupby,
                        missing_only_mask,
                    ),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

    #######################
    #   Initialization    #
    #######################
    def setup(self, stage: str):
        """Might update the columns after initialization of AEDitto but before calling fit on wrapper class, so check right before calling fit here.
        This needs to happen on setup, so pl calls it before configure optimizers.
        Setup is useful for dynamically building models.
        Logic is here since datamodule.setup() will be called in trainer.fit() before self.setup().
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup"""
        if stage == "fit" or stage == "train":
            self.prev_hidden_and_cell_state = {}
            self.set_args_from_data()
            self._setup_logic()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["data_attributes"] = {
            "input_dim": self.input_dim,
            "groupby": self.groupby,
            "original_columns": self.original_columns,
            "discretizations": self.discretizations,
            "ctn_cols_idx": self.ctn_cols_idx,
            "cat_cols_idx": self.cat_cols_idx,
        }
        checkpoint["prev_hidden_and_cell_state"] = self.prev_hidden_and_cell_state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # super().on_load_checkpoint(checkpoint)
        for name, value in checkpoint["data_attributes"].items():
            setattr(self, name, value)
        self.prev_hidden_and_cell_state = checkpoint["prev_hidden_and_cell_state"]
        self._setup_logic()

    def _setup_logic(self) -> None:
        # Requires lossn is set and if col indices set for BCEMSE
        self.loss = self.configure_loss()
        self.set_layer_dims()
        # number of layers will always be even because it's symmetric
        self.code_index = len(self.layer_dims) // 2
        self.build_encoder()
        self.build_decoder()
        # For use with RNN
        # self.prev_hidden_and_cell_state = {
        #     stage: self.init_hidden()
        #     for stage in ["train", "val", "test", "predict"]
        # }
        self.validate_args()

    def set_args_from_data(self):
        """Set model info that we can only dynamically get from the data after it's been setup() in trainer.fit()."""
        self.input_dim = self.datamodule.n_features
        self.groupby = self.datamodule.groupby
        self.original_columns = self.datamodule.columns
        self.discretizations = self.datamodule.discretizations

        # used for BCE+MSE los, -> cat cols the VAE Loss,etc which require tensor
        # We still need this if we're loading the ae from a file and not calling fit
        ctn_cols_idx = self.datamodule.col_indices_by_type["continuous"]
        cat_cols_idx = self.datamodule.col_indices_by_type["categorical"]
        self.ctn_cols_idx = (
            tensor(
                ctn_cols_idx,
                dtype=torch_long,
            )
            if ctn_cols_idx is not None
            else ctn_cols_idx
        )
        self.cat_cols_idx = (
            tensor(cat_cols_idx, dtype=torch_long)
            if cat_cols_idx is not None
            else cat_cols_idx
        )

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
        """Pick loss from options."""
        loss_choices = {
            "BCE": nn.BCEWithLogitsLoss(),
            "MSE": nn.MSELoss(),
            "BCEMSE": BCEMSELoss(self.ctn_cols_idx, self.cat_cols_idx),
        }
        assert (
            self.lossn in loss_choices
        ), f"Passed invalid loss name. Please choose among: {list(loss_choices.keys())}"

        loss = loss_choices[self.lossn]
        # Add KL Divergence for VAE Loss
        if self.vae:
            cat_cols_idx = self.cat_cols_idx if self.lossn == "BCEMSE" else None
            loss = ReconstructionKLDivergenceLoss(loss, cat_cols_idx)
        return loss

    def set_layer_dims(self):
        # Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        assert len(self.hidden_layers) > -1, "Passed no hidden layers."
        # if isinstance(hidden_layers[-1], int) or hidden_layers[0].is_integer():
        if isinstance(self.hidden_layers[-1], int):
            self.layer_dims = (
                [self.input_dim]
                + [int(dim) for dim in self.hidden_layers]
                + [self.input_dim]
            )
        else:  # assuming float, compute relative size of input
            self.layer_dims = (
                [self.input_dim]
                # ceil: e.g.: input_dim = 4,  rel_size: 0.3 and 0.2 when rounded down give 0, so we always round up to the nearest integer (to at least 1).
                + [ceil(rel_size * self.input_dim) for rel_size in self.hidden_layers]
                + [self.input_dim]
            )

    def validate_args(self):
        if (
            self.replace_nan_with is None
            and not self.undiscretize_data
            and self.datamodule.splits["data"]["train"].isna().any()
        ):
            warn(
                "WARNING: You did not indicate what value to replace nans with and are not undiscretizing the data, but NaNs were detected in the input. Please indicate what value you'd like to replace nans with."
            )

        #### Assertions ####
        if self.lossn == "BCEMSE":
            assert (
                self.ctn_cols_idx is not None and self.cat_cols_idx is not None
            ), "Failed to get indices of continuous and categorical columns. Likely failed to pass list of columns and list of continuous columns. This is required for BCEMSE loss."
            assert (
                not self.undiscretize_data
            ), "Passed a loss of BCEMSE but indicated you will undiscretize the data. These cannot both happen (since if all the data is discretized you want to use BCE loss instead)."
        # for now do not support vae with undiscretizing
        assert not (
            self.vae and self.undiscretize_data
        ), "Indicated you wanted to undiscretize the data and also wanted to use a variational autoencoder. These cannot both be true (as the data is fully discretized."

        assert (
            self.undiscretize_data is not None or self.replace_nan_with is not None
        ), "To deal with nans either discretize/undiscretize data, or please indicate what values to replace nans with."

        if self.undiscretize_data:
            assert (
                self.original_columns is not None
            ), "To undiscretize and maintain order, we need the original column order."

        if type(self.replace_nan_with) is str:
            assert (
                self.replace_nan_with == "simple"
            ), "Gave invalid choice to replace nan."

        assert hasattr(
            nn, self.activation
        ), "Failed to choose a valid activation function. Please choose one of the activation functions from torch.nn module."

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
        if self.vae:  # for fc_mu and fc_var, stop before code
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
        if self.vae:
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
        # decoder_layers.append(nn.Sigmoid())
        # self.decoder = nn.Sequential(*decoder_layers)
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
        epsilon = randn_like(std_dev)

        # reparameterize z = mu + std_dev*epsilon
        return mu + (std_dev * epsilon)

    def get_imputed_tensor_from_model_output(
        self,
        data: Tensor,
        reconstruct_batch: Tensor,
        ground_truth: Tensor,
        non_missing_mask: Tensor,
        undiscretized_data: Optional[Tensor],
        undiscretized_ground_truth: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Undiscretize if requested, sigmoid categorical columns only, keep original (potentially undiscretized) values where it's not missing."""
        if undiscretized_data is not None and undiscretized_ground_truth is not None:
            # get undiscretized versions of everything
            reconstruct_batch = undiscretize_tensor(
                reconstruct_batch,
                self.groupby["discretized_ctn_cols"]["data"],
                self.discretizations["data"],
                self.original_columns,
            )
            data = undiscretized_data
            ground_truth = undiscretized_ground_truth
            # re-compute non_missing_mask
            non_missing_mask = (~isnan(data)).bool()
        # sigmoid only on categorical columns
        # TODO: sigmoid per var?
        reconstruct_batch = sigmoid_cat_cols(reconstruct_batch, self.cat_cols_idx)
        # Keep original where it's not missing
        imputed = data.where(non_missing_mask, reconstruct_batch)
        # If the original dataset contains nans (no fully observed), we need to fill in ground_truth too for the metric computation
        # potentially nan in different places than data if amputing (should do nothing if originally fully observed/amputing)
        ground_truth_non_missing_mask = (~isnan(ground_truth)).bool()
        ground_truth = ground_truth.where(
            ground_truth_non_missing_mask, reconstruct_batch
        )

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
        return p
