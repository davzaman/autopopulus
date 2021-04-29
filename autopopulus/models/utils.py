from typing import Any, Callable, Dict, List, Optional, Sequence
from pickle import dump

import torch
from torch import LongTensor, Tensor
import torch.nn as nn

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import Callback


class BatchSwapNoise(nn.Module):
    """Swap Noise Module
    Ref: https://walkwithfastai.com/tab.ae"""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        """Swaps each element with probability p with an alement that correponds to the same column."""
        if self.training:
            # each element assigned random number from uniform dist [0,1)
            # ref: https://github.com/KeremTurgutlu/deeplearning/blob/master/porto_seguro/DAE.py
            mask = torch.rand(x.size()) > (1 - self.p)
            l1 = torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor)
            l2 = mask.type(torch.LongTensor) * x.size(1)
            res = (l1 * l2).view(-1)
            idx = torch.arange(x.nelement()) + res
            idx[idx >= x.nelement()] = idx[idx >= x.nelement()] - x.nelement()
            return x.flatten()[idx].view(x.size())
        else:
            return x


class BCEMSELoss(nn.Module):
    """BCE for categorical columns, MSE for continuous columns."""

    def __init__(self, ctn_columns: LongTensor, cat_columns: LongTensor):
        super().__init__()
        self.ctn_columns = ctn_columns
        self.cat_columns = cat_columns
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor):
        # slicing is differentiable: https://stackoverflow.com/questions/51361407/is-column-selection-in-pytorch-differentiable/51366171
        bce = self.bce(pred[:, self.cat_columns], target[:, self.cat_columns])
        mse = self.mse(pred[:, self.ctn_columns], target[:, self.ctn_columns])
        return bce + mse


class ReconstructionKLDivergenceLoss(nn.Module):
    """Combines reconstruction loss and KL Divergence for VAE Loss.
    Ref: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py#L133
    and https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565"""

    def __init__(
        self,
        reconstruction_loss: nn.Module,
        cat_col_idx: Optional[LongTensor] = None,
        kl_coeff: float = 0.1,
    ) -> None:
        super().__init__()
        self.recon_loss = reconstruction_loss
        self.cat_col_idx = cat_col_idx
        self.kldiv = torch.distributions.kl_divergence
        self.kl_coeff = kl_coeff

    def forward(self, pred: Tensor, target: Tensor, mu: Tensor, logvar: Tensor):
        sigma = torch.exp(0.5 * logvar)
        # p ~ N(0,1)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        # q ~ N(mu, sigma)
        q = torch.distributions.Normal(mu, sigma)
        kldiv = self.kldiv(p, q).mean()

        #### Reconstruction Loss ####
        pred = sigmoid_cat_cols(pred, self.cat_col_idx)
        recon_loss = self.recon_loss(pred, target)

        return recon_loss + self.kl_coeff + kldiv


def sigmoid_cat_cols(data: Tensor, cat_col_idx: Optional[LongTensor]) -> Tensor:
    """Puts categorical columns of tensor through sigmoid.
    Uses list of continuous columns/ctn/cat_col_idx to put only categorical columns through sigmoid.
    """
    if cat_col_idx is not None:
        data[:, cat_col_idx] = torch.sigmoid(data[:, cat_col_idx])
    else:
        data = torch.sigmoid(data)
    return data


class ErrorAnalysisCallback(Callback):
    def __init__(self, limit_n: int = 10) -> None:
        super().__init__()
        self.limit_n = limit_n
        self.best_n: Dict[str, Dict[str, Tensor]] = {}
        self.worst_n: Dict[str, Dict[str, Tensor]] = {}

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # pickle the results
        with open(f"error_analysis/best_{self.limit_n}", "wb") as file:
            dump(self.best_n, file)

        with open(f"error_analysis/worst_{self.limit_n}", "wb") as file:
            dump(self.worst_n, file)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ):
        for name, metricfn in pl_module.metrics.items():
            metric_fn_args = [outputs["pred"], outputs["ground_truth"]]
            if name == "AccuracyPerBin":
                continue
                (
                    (data, ground_truth),
                    (undiscretized_data, undiscretized_ground_truth),
                ) = batch
                args = [
                    batch,
                    ground_truth,
                    pl_module.discrete_columns,
                    pl_module.columns,
                ]
            self.compute_metrics_and_merge_batch(
                name, metricfn, outputs, metric_fn_args
            )

            # Compute metrics for missing only data
            missing_only_mask = ~(outputs["non_missing_mask"].bool())
            if missing_only_mask.any():
                metric_fn_args.append(missing_only_mask)
                self.compute_metrics_and_merge_batch(
                    name, metricfn, outputs, metric_fn_args
                )

    def compute_metrics_and_merge_batch(
        self,
        name: str,
        metricfn: Callable,
        outputs: Dict[str, Tensor],
        metric_fn_args: List[Any] = None,
    ):
        result = metricfn(*metric_fn_args, reduction="none")

        # sort the metric values/results
        sorted_result, indices = torch.sort(result)
        # get n best and worst values, and get their corresponding observations
        best_n = {
            "pred": outputs["pred"].iloc[indices[:10]],
            "true": outputs["ground_truth"].iloc[indices[:10]],
            "values": sorted_result[:10],
        }
        worst_n = {
            "pred": outputs["pred"].iloc[indices[-10:]],
            "true": outputs["ground_truth"].iloc[indices[-10:]],
            "values": sorted_result[-10:],
        }

        if name in self.best_n:  # update
            self.merge_batch(best_n, worst_n, name)
        else:  # add in directly
            self.best_n[name] = best_n
            self.worst_n[name] = worst_n

    def merge_batch(
        self, best_n: Dict[str, Tensor], worst_n: Dict[str, Tensor], name: str
    ):
        # sort 20 values (10 from best/worst 10, and then 10 best/worst from newest batch.)
        sorted_values, indices = torch.sort(
            torch.cat(self.best_n[name]["values"], best_n["values"])
        )

        self.best_n[name] = {
            "pred": torch.cat(self.best_n[name]["pred"], best_n["pred"]).iloc[
                indices[:10]
            ],
            "true": torch.cat(self.best_n[name]["true"], best_n["true"]).iloc[
                indices[:10]
            ],
            "values": sorted_values[:10],
        }

        self.worst_n[name] = {
            "pred": torch.cat(self.worst_n[name]["pred"], worst_n["pred"]).iloc[
                indices[-10:]
            ],
            "true": torch.cat(self.worst_n[name]["true"], worst_n["true"]).iloc[
                indices[-10:]
            ],
            "values": sorted_values[-10:],
        }


class Print(nn.Module):
    """For debugging purposes."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x)
        return x


class ResetSeed(nn.Module):
    """Dropout seems to be nondeterminstic, reset seed to ensure reproducibility."""

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

    def forward(self, x):
        torch.manual_seed(self.seed)
        return x
