from typing import Optional
import torch
from torch import LongTensor, Tensor
import torch.nn as nn


from autopopulus.data.transforms import sigmoid_cat_cols


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
        if len(pred.shape) == 2:  # static
            bce = self.bce(pred[:, self.cat_columns], target[:, self.cat_columns])
            mse = self.mse(pred[:, self.ctn_columns], target[:, self.ctn_columns])
        elif len(pred.shape) == 3:  # longitudinal
            bce = self.bce(pred[:, :, self.cat_columns], target[:, :, self.cat_columns])
            mse = self.mse(pred[:, :, self.ctn_columns], target[:, :, self.ctn_columns])
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
