from argparse import ArgumentParser, Namespace
import inspect
import os
from inspect import getmembers, isfunction
from typing import List, Union, Any

import torch
from pytorch_lightning import seed_everything as pl_seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich import (
    print,
)  # https://rich.readthedocs.io/en/stable/markup.html#console-markup


def should_ampute(args: Namespace) -> bool:
    return (
        ("percent_missing" in args and args.percent_missing)
        and ("amputation_patterns" in args and args.amputation_patterns)
        and args.method != "none"
    )


# Printing with rich but only in rank zero
@rank_zero_only
def rank_zero_print(*args: Any, **kwargs: Any):
    return print(*args, **kwargs)


def seed_everything(seed: int):
    """Sets seeds and also makes cuda deterministic for pytorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl_seed_everything(seed)
    # RNN/LSTM determininsm error with cuda 10.1/10.2
    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    # https://github.com/pytorch/pytorch/issues/1637#issuecomment-730423426
    # https://github.com/Lightning-AI/lightning/issues/4420#issuecomment-926495956
    # Problem with running on 4 gpus
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_DEBUG"] = "WARN"
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def get_module_function_names(module) -> List[str]:
    return [
        name
        for name, fn in getmembers(module, isfunction)
        # ignore imported methods
        if fn.__module__ == module.__name__
    ]


class CLIInitialized:
    @classmethod
    def from_argparse_args(
        cls,
        args: Union[Namespace, ArgumentParser],
        inner_classes: List = None,
        **kwargs
    ):
        """
        Create an instance from CLI arguments.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
        If there are inner classes to instantiate and you need their signature, you can overload this method and pass a list of classes:
            e.g. return super().from_argparse_args(cls, args, [AEDitto], **kwargs)
        # Ref: https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/trainer/trainer.py#L750
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid args, the rest may be user specific
        # returns a immutable dict MappingProxyType, want to combine so copy
        valid_kwargs = inspect.signature(cls.__init__).parameters.copy()
        if inner_classes is not None:
            for inner_class in inner_classes:  # Update with inner classes
                valid_kwargs.update(
                    inspect.signature(inner_class.__init__).parameters.copy()
                )
        data_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        data_kwargs.update(**kwargs)

        return cls(**data_kwargs)
