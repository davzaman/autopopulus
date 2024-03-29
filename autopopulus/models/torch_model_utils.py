from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric

from autopopulus.data.constants import PAD_VALUE
from autopopulus.utils.utils import rank_zero_print


def detach_tensor(
    t: torch.Tensor, dtype: torch.dtype = torch.float, to_cpu: bool = False
) -> torch.Tensor:
    t = t.detach()
    if to_cpu:
        t = t.cpu()
    return t


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


class CtnCatLoss(nn.Module):
    def __init__(
        self,
        ctn_cols_idx: torch.LongTensor,
        bin_cols_idx: torch.LongTensor,
        onehot_cols_idx: torch.LongTensor,
        loss_bin: Union[_Loss, Metric] = nn.BCEWithLogitsLoss(),
        loss_onehot: Union[_Loss, Metric] = nn.CrossEntropyLoss(),
        loss_ctn: Union[_Loss, Metric] = nn.MSELoss(),
    ):
        super().__init__()
        # we need to register these as buffers in the loss module as well for it to move to the correct device.
        # these should't be updated at all in this loss so we don't have to worry about resetting
        self.register_buffer("ctn_cols_idx", ctn_cols_idx)
        self.register_buffer("bin_cols_idx", bin_cols_idx)
        self.register_buffer("onehot_cols_idx", onehot_cols_idx)
        self.loss_bin = loss_bin
        self.loss_onehot = loss_onehot
        self.loss_ctn = loss_ctn
        self.assert_losses_differentiable_if_metric()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # If metric apply data directly (interally should filter), but if loss we need to filter the data first
        # Loss for a component where there are no features of that type will be 0
        # slicing is differentiable: https://stackoverflow.com/questions/51361407/is-column-selection-in-pytorch-differentiable/51366171
        # Select in dim 1 (static)  == [:, select], or dim 2 (longitudinal) == [:, :, select]
        slice_dim = len(pred.shape) - 1
        loss_onehot = 0
        if isinstance(self.loss_onehot, Metric):
            loss_onehot += self.loss_onehot(pred, target)
        else:
            for onehot_group in self.onehot_cols_idx:
                # ignore pads of -1
                onehot_group = onehot_group[onehot_group != PAD_VALUE]
                loss_onehot += self.loss_onehot(
                    pred.index_select(slice_dim, onehot_group),
                    # Last dim: whether static/longitudinal the last dim is features
                    torch.argmax(
                        target.index_select(slice_dim, onehot_group), dim=slice_dim
                    ),
                )
        if isinstance(self.loss_bin, Metric):
            loss_bin = self.loss_bin(pred, target)
        else:
            if self.bin_cols_idx.numel() == 0:
                loss_bin = 0
            else:
                loss_bin = self.loss_bin(
                    pred.index_select(slice_dim, self.bin_cols_idx),
                    target.index_select(slice_dim, self.bin_cols_idx),
                )
        if isinstance(self.loss_ctn, Metric):
            loss_ctn = self.loss_ctn(pred, target)
        else:
            if self.ctn_cols_idx.numel() == 0:
                loss_ctn = 0
            else:
                loss_ctn = self.loss_ctn(
                    pred.index_select(slice_dim, self.ctn_cols_idx),
                    target.index_select(slice_dim, self.ctn_cols_idx),
                )
        self.reset_if_metric()
        return loss_onehot + loss_bin + loss_ctn

    def assert_losses_differentiable_if_metric(self):
        """If the loss is a metric it must be differentiable."""
        assert not isinstance(self.loss_bin, Metric) or self.loss_bin.is_differentiable
        assert (
            not isinstance(self.loss_onehot, Metric)
            or self.loss_onehot.is_differentiable
        )
        assert not isinstance(self.loss_ctn, Metric) or self.loss_ctn.is_differentiable

    def reset_if_metric(self):
        """
        If i'm using a torchmetric for loss it needs to be reset each batch.
        Technically the loss should not have state.
        """
        if isinstance(self.loss_bin, Metric):
            self.loss_bin.reset()
        if isinstance(self.loss_onehot, Metric):
            self.loss_onehot.reset()
        if isinstance(self.loss_ctn, Metric):
            self.loss_ctn.reset()


class ReconstructionKLDivergenceLoss(nn.Module):
    """Combines reconstruction loss and KL Divergence for VAE Loss.
    Ref: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py#L133
    and https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565"""

    def __init__(
        self,
        reconstruction_loss: nn.Module,
        cat_cols_idx: Optional[torch.LongTensor] = None,
        kl_coeff: float = 0.1,
    ) -> None:
        super().__init__()
        self.recon_loss = reconstruction_loss
        self.cat_cols_idx = cat_cols_idx
        self.kldiv = torch.distributions.kl_divergence
        self.kl_coeff = kl_coeff

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        ## KL-Divergence
        sigma = torch.exp(0.5 * logvar)
        # p ~ N(0,1)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        # q ~ N(mu, sigma)
        q = torch.distributions.Normal(mu, sigma)
        kldiv = self.kldiv(p, q).mean()

        #### Reconstruction Loss ####
        recon_loss = self.recon_loss(pred, target)

        return recon_loss + (self.kl_coeff * kldiv)


class Print(nn.Module):
    """For debugging purposes."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        rank_zero_print(x)
        return x


class ResetSeed(nn.Module):
    """Dropout seems to be nondeterminstic, reset seed to ensure reproducibility."""

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

    def forward(self, x):
        torch.manual_seed(self.seed)
        return x


def binary_column_threshold(
    X: torch.Tensor, bin_col_idxs: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Apply sigmoid and threshold to the given columns.
    NOTE: modifies tensor in place as well.
    These are tested through their nn.module wrappers.
    """
    longitudinal = len(X.shape) == 3
    if bin_col_idxs.numel() > 0:
        if longitudinal:
            X[:, :, bin_col_idxs] = (
                torch.sigmoid(X[:, :, bin_col_idxs]) >= threshold
            ).to(X.dtype)
        else:
            X[:, bin_col_idxs] = (torch.sigmoid(X[:, bin_col_idxs]) >= threshold).to(
                X.dtype
            )
    return X


def onehot_column_threshold(
    X: torch.Tensor, onehot_cols_idx: torch.Tensor
) -> torch.Tensor:
    """
    Apply log-softmax, get numerical encoded bin, and explode back to onehot.
    NOTE: modifies tensor in place as well.
    The wrapper function is tested.
    These are tested through their nn.module wrappers.
    """
    longitudinal = len(X.shape) == 3
    for onehot_group in onehot_cols_idx:
        # ignore pads of -1
        onehot_group = onehot_group[onehot_group != PAD_VALUE]
        # Last dim: whether static/longitudinal the last dim is features
        if longitudinal:
            X[:, :, onehot_group] = torch.nn.functional.one_hot(
                torch.argmax(torch.log_softmax(X[:, :, onehot_group], dim=2), dim=2),
                num_classes=len(onehot_group),
            ).to(X.dtype)
        else:
            X[:, onehot_group] = torch.nn.functional.one_hot(
                torch.argmax(torch.log_softmax(X[:, onehot_group], dim=1), dim=1),
                num_classes=len(onehot_group),
            ).to(X.dtype)

    return X


class BinColumnThreshold(nn.Module):
    """Pytorch wrapper for binary_column_threshold."""

    # Ref: https://discuss.pytorch.org/t/slice-layer-solution/12474
    def __init__(self, col_idxs: torch.Tensor, threshold: float = 0.5):
        super().__init__()
        self.col_idxs = col_idxs
        self.threshold = threshold

    def forward(self, x):
        return binary_column_threshold(x, self.col_idxs, self.threshold)


class OnehotColumnThreshold(nn.Module):
    """Pytorch wrapper for onehot_column_threshold."""

    def __init__(self, col_idxs: torch.Tensor):
        super().__init__()
        self.col_idxs = col_idxs

    def forward(self, x):
        return onehot_column_threshold(x, self.col_idxs)


class ColumnEmbedder(nn.Module):
    """
    Apply a linear layer to the given columns.
    Embeddings must be invertible: they cannot bottleneck and the activation must be invertible.
    Ref: https://stats.stackexchange.com/questions/489429/why-are-the-tied-weights-in-autoencoders-transposed-and-not-inverted
    """

    # Ref: https://discuss.pytorch.org/t/slice-layer-solution/12474
    def __init__(self, col_idxs: List[int]):
        super().__init__()
        self.col_idxs = col_idxs
        dim = len(self.col_idxs)
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear.forward(x[:, self.col_idxs])


class InvertEmbedding(nn.Module):
    """Inverse apply a layer to the given columns."""

    def __init__(self, embedding_layer: ColumnEmbedder) -> None:
        self.layer = embedding_layer.linear
        self.col_idxs = embedding_layer.col_idxs

    def forward(self, x):
        # the inverse == transpose because embedding layer is square.
        # Ref: https://stackoverflow.com/a/59886624/1888794
        # Ref: https://discuss.pytorch.org/t/transpose-of-linear-layer/12411
        return torch.matmul(
            self.layer.weight.T, (x[:, self.col_idxs] - self.layer.bias)
        )
