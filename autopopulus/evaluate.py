import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from numpy.random import default_rng
from tqdm import tqdm
import argparse  # needed to guild knows to import flags

from autopopulus.utils.get_set_cli_args import init_cli_args, load_cli_args
from autopopulus.utils.utils import resample_indices_only, seed_everything
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

    if args.bootstrap_eval_imputer:
        test_dataset = test_dataloader.dataset
        gen = default_rng(args.seed)
        for b in tqdm(range(args.num_bootstraps)):
            bootstrap_indices = resample_indices_only(len(test_dataset), gen)
            bootstrap_subset = Subset(test_dataset, bootstrap_indices)
            bootstrap_loader = DataLoader(
                bootstrap_subset,
                batch_size=test_dataloader.batch_size,
                shuffle=False,
                prefetch_factor=test_dataloader.prefetch_factor,
                num_workers=test_dataloader.num_workers,
                collate_fn=test_dataloader.collate_fn,
                pin_memory=test_dataloader.pin_memory,
            )
            # to manually set the step for pytorch_lightning.LoggerConnector:log_metrics
            imputer.inference_trainer.fit_loop.epoch_loop._batches_that_stepped = b
            imputer.inference_trainer.test(imputer.ae, bootstrap_loader)
    else:
        imputer.inference_trainer.test(imputer.ae, test_dataloader)


if __name__ == "__main__":
    main()
