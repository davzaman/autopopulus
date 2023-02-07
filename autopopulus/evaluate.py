import torch
import argparse  # needed to guild knows to import flags

from autopopulus.utils.get_set_cli_args import init_cli_args, load_cli_args
from autopopulus.utils.utils import seed_everything
from autopopulus.models.ap import AEImputer
from autopopulus.utils.log_utils import get_serialized_model_path
from autopopulus.task_logic.ae_imputation import AE_METHOD_SETTINGS


def main():
    """
    Only for autoencoder imputation, as for pytorch while training can be
    distributed across multiple GPUs, testing must be done on one GPU.
    It is also most recommended to put testing in a separate script.
    """
    load_cli_args()
    args = init_cli_args()

    if not args.runtest or args.method not in AE_METHOD_SETTINGS:  # Do nothing
        return

    seed_everything(args.seed)
    test_dataloader = torch.load(
        get_serialized_model_path(
            f"{args.data_type_time_dim.name}_test_dataloader", "pt"
        )
    )
    imputer = AEImputer.from_checkpoint(
        args,
        get_serialized_model_path(f"AEDitto_{args.data_type_time_dim.name}", "pt"),
    )
    imputer.inference_trainer.test(imputer.ae, test_dataloader)


if __name__ == "__main__":
    main()
