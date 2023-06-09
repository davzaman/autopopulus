from argparse import Namespace
from typing import Dict
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.profilers.advanced import AdvancedProfiler

## Local Modules
from autopopulus.data import CommonDataModule
from autopopulus.models.ap import AEImputer
from autopopulus.utils.log_utils import AutoencoderLogger
from autopopulus.utils.utils import rank_zero_print
from autopopulus.task_logic.utils import AE_METHOD_SETTINGS


@rank_zero_only
def ae_transform(
    data_module: CommonDataModule, ae_imputer: AEImputer, split_name: str
) -> ndarray:
    split_dataloader = getattr(data_module, f"{split_name}_dataloader")
    return ae_imputer.transform(split_dataloader())


def ae_imputation_logic(
    args: Namespace, data: CommonDataModule
) -> Dict[str, Dict[str, DataFrame]]:
    """Output: top-level lookup: static/long. second-level: train/val/test."""
    # combine two dicts python 3.5+
    settings = AE_METHOD_SETTINGS[args.method]["train"]

    if args.ae_from_checkpoint:
        if args.tune_n_samples:
            rank_zero_warn(
                "Specified a checkpoint and to run tuning, we default to loading the checkpoint."
                " If this was not the intention, please do not specify a checkpoint."
            )
        rank_zero_print(f"Loading AEImputer from {args.ae_from_checkpoint}")
        ae_imputer = AEImputer.from_checkpoint(args)
        data.setup("fit")
    else:
        ae_imputer = AEImputer.from_argparse_args(
            args,
            logger=AutoencoderLogger(args),
            tune_callback=None,
            **settings,
        )
        if args.tune_n_samples:
            ae_imputer.tune(
                args.experiment_name,
                args.tune_n_samples,
                args.total_cpus_on_machine,
                args.total_gpus_on_machine,
                args.n_gpus_per_trial,
                data=data,
            )
        else:
            ae_imputer.fit(data)

    transformed = {
        split_name: ae_transform(data, ae_imputer, split_name)
        for split_name in ["train", "val", "test"]
    }

    return transformed
