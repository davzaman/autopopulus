from os import makedirs
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from os.path import join
from pytorch_lightning import Callback, LightningModule, Trainer
import torch
import torch.nn as nn

# This is modified to be a callback from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html


class VisualizeModelCallback(Callback):
    def __init__(
        self,
        gradient: bool = True,
        activation: bool = True,
        weight: bool = True,
        variances: bool = True,
    ) -> None:
        super().__init__()
        self.gradient = gradient
        self.activation = activation
        self.weight = weight
        self.variances = variances
        self.epoch = 0

    def plot_dists(self, val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
        columns = len(val_dict)
        fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))
        fig_index = 0
        for key in sorted(val_dict.keys()):
            key_ax = ax[fig_index % columns]
            sns.histplot(
                val_dict[key],
                ax=key_ax,
                color=color,
                bins=50,
                stat=stat,
                kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
            )  # Only plot kde if there is variance
            hidden_dim_str = (
                r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0])
                if len(val_dict[key].shape) > 1
                else ""
            )
            key_ax.set_title(f"{key} {hidden_dim_str}")
            if xlabel is not None:
                key_ax.set_xlabel(xlabel)
            fig_index += 1
        fig.subplots_adjust(wspace=0.4)
        return fig

    def plot_stats(
        self, means: List[float], stds: List[float], names: List[str], save_dir: str
    ):
        fig, axes = plt.subplots(ncols=2, sharey=False)
        for i, (stat_name, values) in enumerate(
            [("Means", means), ("Standard Deviations", stds)]
        ):
            sns.lineplot(y=values, x=names, ax=axes[i]).set(
                title=f"Gradient {stat_name}"
            )
        fig.savefig(join(save_dir, f"grad_stats"))
        plt.close()

    def _is_activation_layer(self, layer: nn.Module) -> bool:
        return (
            isinstance(layer, nn.Tanh)
            or isinstance(layer, nn.ReLU)
            or isinstance(layer, nn.Sigmoid)
        )

    def _is_epoch_end(self, trainer: Trainer):
        train_dataloader = trainer.datamodule.train_dataloader()
        return (
            trainer.global_step != 0
            and trainer.global_step % len(train_dataloader) == 0
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # on_before_zero_grad doesn't work
        train_dataloader = trainer.datamodule.train_dataloader()
        if self._is_epoch_end(trainer):
            save_dir = join("model_viz", f"epoch_{self.epoch}")
            makedirs(save_dir, exist_ok=True)
            if self.gradient:
                # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
                grads = {
                    name: params.grad.view(-1).cpu().clone().numpy()
                    if params.grad is not None
                    else None
                    for name, params in pl_module.named_parameters()
                    if "weight" in name
                }
                fig = self.plot_dists(grads, color="C0", xlabel="Grad magnitude")
                fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
                fig.tight_layout()
                fig.savefig(join(save_dir, "grad_dist"))
                plt.close()
                if self.variances:
                    # TODO: add names to x axis
                    grad_variances = [np.var(v) for v in grads.values()]
                    fig = sns.lineplot(y=grad_variances, x=list(grads.keys())).figure
                    plt.xlabel("Gradient Variance")
                    fig.tight_layout()
                    fig.savefig(join(save_dir, "grad_variances"))
                    plt.close()

            if self.activation:
                activations = {}

                one_batch = next(iter(train_dataloader))[
                    pl_module.hparams.data_feature_space
                ]["data"]
                with torch.no_grad():
                    for layer_index, layer in enumerate(
                        pl_module.encoder + pl_module.decoder
                    ):
                        one_batch = layer(one_batch)
                        if isinstance(layer, nn.Linear):
                            activations[f"Layer {layer_index}"] = (
                                one_batch.view(-1).detach().cpu().numpy()
                            )
                fig = self.plot_dists(
                    activations, color="C0", stat="density", xlabel="Activation vals"
                )
                fig.suptitle("Activation distribution", fontsize=14, y=1.05)
                fig.tight_layout()
                fig.savefig(join(save_dir, "activation_dist"))
                plt.close()
                if self.variances:
                    activation_variances = [np.var(v) for v in activations.values()]
                    fig = sns.lineplot(activation_variances).figure
                    plt.xlabel("Activation Variance")
                    fig.tight_layout()
                    fig.savefig(join(save_dir, "activation_variances"))
                    plt.close()
            if self.weight:
                weights = {}
                for name, param in pl_module.named_parameters():
                    if name.endswith(".bias"):
                        continue
                    key_name = f"Layer {name.split('.')[1]}"
                    weights[key_name] = param.detach().view(-1).cpu().numpy()
                fig = self.plot_dists(weights, color="C0", xlabel="Weight vals")
                fig.suptitle("Weight distribution", fontsize=14, y=1.05)
                fig.tight_layout()
                fig.savefig(join(save_dir, "weight_dist"))
                plt.close()
            self.epoch += 1
