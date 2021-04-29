from math import ceil
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from pytorch_lightning.metrics.metric import Metric

#### Pytorch ####
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.optim as optim

## Lightning ##
import pytorch_lightning as pl

from models.utils import (
    BCEMSELoss,
    BatchSwapNoise,
    ReconstructionKLDivergenceLoss,
    ResetSeed,
    sigmoid_cat_cols,
)
from data.transforms import undiscretize_tensor, simple_impute_tensor
from utils.impute_metrics import AccuracyPerBin, RMSE, MAAPE

DEFAULT_METRICS = {
    "RMSE": RMSE,
    "MAAPE": MAAPE,
}


LOSS_CHOICES = ["BCE", "MSE", "BCEMSE"]
OPTIM_CHOICES = ["Adam", "SGD"]
ACTIVATION_CHOICES = ["ReLU", "TanH", "sigmoid"]


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
    Flavors:
        - {overcomplete, undercomplete}
            - depends on hidden_layers
        - {Denoising, Vanilla, Variational}
            - Denoising via {dropout (noise is value 0), batchswap}_corruption
            - variational via vae flag.
        - with/without dropout
        - losses {BCE, MSE, BCE (CAT) + MSE (CTN), reconstruction+kldivergence}
            - BCE is always with logits.
            - reconstruction+kldivergence is always when vae flag is chosen.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[Union[int, float]],
        lr: float,
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
        undiscretize_data: bool = False,
        replace_nan_with: Optional[
            Union[int, str]
        ] = None,  # Only used if not undiscretizing
        dropout: Optional[float] = None,
        dropout_corruption: Optional[float] = None,
        batchswap_corruption: Optional[float] = None,
        columns: Optional[List[str]] = None,
        ctn_columns: Optional[List[str]] = None,
        discrete_columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.lossn = lossn
        self.optimn = optimn
        self.seed = seed
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.activation = activation
        self.metrics = metrics
        # Other options
        self.mvec = mvec
        self.vae = vae
        self.undiscretize_data = undiscretize_data
        self.replace_nan_with = replace_nan_with
        self.dropout_corruption = dropout_corruption
        self.batchswap_corruption = batchswap_corruption
        # used for undiscretization
        self.columns = columns
        # used for BCE+MSE los, -> cat cols the VAE Loss, and undiscretize
        self.ctn_columns = ctn_columns
        # used for undiscretize
        self.discrete_columns = discrete_columns
        # self.set_coltype_indices()
        self.hidden_layers = hidden_layers

        # Add accuracy for number of bins correctly imputed if everything is discretized
        if self.undiscretize_data:
            self.metrics["AccuracyPerBin"] = AccuracyPerBin

        # NOTE: things that depends on the columns like the loss (BCEMSE) are set and checked on_fit_start() as well
        # We still need this if we're loading the ae from a file and not calling fit
        self.set_coltype_indices()

        # Assumes layer_dims describes full autoencoder (is symmetric list of numbers).
        assert len(hidden_layers) > 0, "Passed no hidden layers."
        # if isinstance(hidden_layers[0], int) or hidden_layers[0].is_integer():
        if isinstance(hidden_layers[0], int):
            self.layer_dims = (
                [input_dim] + [int(dim) for dim in hidden_layers] + [input_dim]
            )
        else:  # assuming float, compute relative size of input
            self.layer_dims = (
                [input_dim]
                # ceil: e.g.: input_dim = 5,  rel_size: 0.3 and 0.2 when rounded down give 0, so we always round up to the nearest integer (to at least 1).
                + [ceil(rel_size * input_dim) for rel_size in self.hidden_layers]
                + [input_dim]
            )
        # number of layers will always be even because it's symmetric
        self.code_index = len(self.layer_dims) // 2
        self.build_encoder()
        self.build_decoder()

    def encode(self, x):
        """Returns code for AE/DAE, mu/var for VAE."""
        hidden = self.encoder(x)
        if self.vae:
            mu, logvar = self.fc_mu(hidden), self.fc_var(hidden)
            return self.reparameterize(mu, logvar), mu, logvar
        return hidden

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """What happens when you pass data through an object of this class."""
        if self.vae:
            code, mu, logvar = self.encode(x)
        else:
            code = self.encode(x)
        if self.vae:
            return self.decode(code), mu, logvar
        return self.decode(code)

    #######################
    # Training/Eval Logic #
    #######################
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

    def shared_step(self, batch, split: str) -> Dict[str, float]:
        if self.undiscretize_data:
            (
                (data, ground_truth),
                (undiscretized_data, undiscretized_ground_truth),
            ) = batch
        else:
            data, ground_truth = batch
            undiscretized_data = None
            undiscretized_ground_truth = None

        # set this before filling in data with replacement (if doing so)
        non_missing_mask = ~(torch.isnan(data)).bool()

        if self.replace_nan_with is not None:
            # replace nan in ground truth too if its missing any
            if self.replace_nan_with == "simple":  # simple impute warm start
                data = simple_impute_tensor(data, self.ctn_cols_idx, self.cat_cols_idx)
                ground_truth = simple_impute_tensor(
                    ground_truth, self.ctn_cols_idx, self.cat_cols_idx
                )
            else:  # Replace nans with a single value provided
                data = torch.nan_to_num(data, nan=self.replace_nan_with)
                ground_truth = torch.nan_to_num(ground_truth, nan=self.replace_nan_with)

        reconstruct_batch = self(data)
        if self.vae:  # unpack when vae
            reconstruct_batch, mu, logvar = reconstruct_batch

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
        if self.vae:
            loss = self.loss(eval_pred, eval_true, mu, logvar)
        else:
            # NOTE: if no mvec and no vae for some reason it says the loss is modifying the output in place, the clone is a quick hack
            loss = self.loss(eval_pred.clone(), eval_true)
            # loss = self.loss(eval_pred, eval_true)

        # save discretized outputs for evaluation
        if undiscretized_data is not None and undiscretized_ground_truth is not None:
            self.pre_undiscretize_logging_step(
                reconstruct_batch.detach(),
                ground_truth.detach(),
                (~torch.isnan(undiscretized_data)).bool(),
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
        )

        return (
            loss,
            {
                "loss": loss,
                "pred": pred.detach(),
                "ground_truth": ground_truth.detach(),
                "non_missing_mask": non_missing_mask.detach(),
            },
        )

    def shared_logging_step_end(self, outputs: Dict[str, float], split: str):
        """Log metrics + loss at end of step.
        Compatible with dp mode: https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#classification-metrics."""
        # Log loss
        self.log(
            f"AE/{split}-loss",
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
                f"impute/{split}-{name}",
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
                    f"impute/{split}-{name}-missingonly",
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
            self.log(
                f"impute/{split}-{name}",
                self.metrics[name](
                    pred_disc, ground_truth_disc, self.discrete_columns, self.columns
                ),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            # Compute metrics for missing only data
            missing_only_mask = ~(non_missing_mask.bool())
            if missing_only_mask.any():
                self.log(
                    f"impute/{split}-{name}-missingonly",
                    self.metrics[name](
                        pred_disc,
                        ground_truth_disc,
                        self.discrete_columns,
                        self.columns,
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
    def configure_optimizers(self):
        """Pick optimizer."""
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

    def on_fit_start(self):
        """Might update the columns after initialization of AEDitto but before calling fit on wrapper class, so check right before calling fit here."""
        self.set_coltype_indices()
        # Requires lossn is set and if col indices set for BCEMSE
        self.loss = self.configure_loss()

        #### Assertions ####
        if self.lossn == "BCEMSE":
            assert (
                self.ctn_columns is not None and self.columns is not None
            ), "Failed to pass list of continuous columns and list of all columns, required for BCEMSE loss."
            assert (
                self.ctn_cols_idx is not None and self.cat_cols_idx is not None
            ), "Failed to get indices of continuous and categorical columns. Likely failed to pass list of columns and list of continuous columns. This is required for BCEMSE loss."
            assert (
                not self.undiscretize_data
            ), "Passed a loss of BCEMSE but indicated you will undiscretize the data. These cannot both happen (since if all the data is discretized you want to use BCE loss instead)."
        if self.undiscretize_data:
            assert (
                self.columns is not None
            ), "Failed to pass list of columns, required for undiscretizing the data."
            assert (
                self.ctn_columns is not None
            ), "Failed to pass list of continuous columns, required for undiscretizing the data."
            assert (
                self.discrete_columns is not None
            ), "Failed to pass list of discrete columns, required for undiscretizing the data."
        # for now do not support vae with undiscretizing
        assert not (
            self.vae and self.undiscretize_data
        ), "Indicated you wanted to undiscretize the data and also wanted to use a variational autoencoder. These cannot both be true (as the data is fully discretized."

        assert (
            self.undiscretize_data is not None or self.replace_nan_with is not None
        ), "To deal with nans either discretize/undiscretize data, or please indicate what values to replace nans with."

        if type(self.replace_nan_with) is str:
            assert (
                self.replace_nan_with == "simple"
            ), "Gave invalid choice to replace nan."

        assert (
            self.activation in ACTIVATION_CHOICES
        ), f"Failed to choose a valid activation function. Please choose one of {ACTIVATION_CHOICES}"

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
            encoder_layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if self.activation == "ReLU":
                encoder_layers.append(nn.ReLU(inplace=True))
            elif self.activation == "sigmoid":
                encoder_layers.append(nn.Sigmoid())
            else:  # TanH
                encoder_layers.append(nn.Tanh())

            if self.dropout:
                encoder_layers += [
                    ResetSeed(self.seed),  # ensures reprodudicibility
                    nn.Dropout(self.dropout),
                ]
        self.encoder = nn.Sequential(*encoder_layers)
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
            decoder_layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if self.activation == "ReLU":
                decoder_layers.append(nn.ReLU(inplace=True))
            elif self.activation == "sigmoid":
                decoder_layers.append(nn.Sigmoid())
            else:  # TanH
                decoder_layers.append(nn.Tanh())
            if self.dropout:
                decoder_layers += [
                    ResetSeed(self.seed),  # ensures reprodudicibility
                    nn.Dropout(self.dropout),
                ]
        decoder_layers.append(nn.Linear(self.layer_dims[-2], self.layer_dims[-1]))
        # decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    #######################
    #       Helpers       #
    #######################
    def set_coltype_indices(self):
        """Returns list of indices as LongTensor (which can be used as indexer) of continuous cols and categorical cols in the tensor."""
        if self.ctn_columns is None:
            self.ctn_cols_idx = None
            self.cat_cols_idx = None
            return

        n_cols = len(self.columns)
        ctn_cols = []
        cat_cols = []
        for i in range(n_cols):
            if self.columns[i] in self.ctn_columns:
                ctn_cols.append(i)
            else:
                cat_cols.append(i)
        self.ctn_cols_idx = torch.tensor(ctn_cols, dtype=torch.long)
        self.cat_cols_idx = torch.tensor(cat_cols, dtype=torch.long)

    def col_indices(self, cols: List[str]) -> LongTensor:
        """Return (longtensor) list of indices of passed in columns in tensor."""
        idx = []
        for i in range(len(self.columns)):
            if self.columns[i] in cols:
                idx.append(i)
        return LongTensor(idx)

    def reparameterize(self, mu, logvar):  # mean of N(mu, var)  # ln variance
        """Reparameterization trick for VAE.

        If we assume that the latent var z ~ p(z | x) = N(mu, std_dev^2)
        A valid reparameterization is z = mu + std_dev*epsilon
            where epsilon is noise: eps ~ N(0,1)
        """
        # since we're using log of variance e^.5ln(var) = var^.5
        std_dev = torch.exp(0.5 * logvar)
        # noise following N(0,1)
        epsilon = torch.randn_like(std_dev)

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
                reconstruct_batch, self.columns, self.discrete_columns, self.ctn_columns
            )
            data = undiscretized_data
            ground_truth = undiscretized_ground_truth
            # re-compute non_missing_mask
            non_missing_mask = (~torch.isnan(data)).bool()
        # sigmoid only on categorical columns
        reconstruct_batch = sigmoid_cat_cols(reconstruct_batch, self.cat_cols_idx)
        # Keep original where it's not missing
        imputed = data.where(non_missing_mask, reconstruct_batch)
        # If the original dataset contains nans (no fully observed), we need to fill in ground_truth too for the metric computation
        # potentially nan in different places than data if amputing (should do nothing if originally fully observed/amputing)
        ground_truth_non_missing_mask = (~torch.isnan(ground_truth)).bool()
        ground_truth = ground_truth.where(
            ground_truth_non_missing_mask, reconstruct_batch
        )

        return imputed, ground_truth, non_missing_mask
