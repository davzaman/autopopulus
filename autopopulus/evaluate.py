import argparse  # needed to guild knows to import flags
from argparse import Namespace
from typing import Optional

import torch
from numpy.random import default_rng
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from autopopulus.models.ap import AEImputer
from autopopulus.task_logic.baseline_imputation import evaluate_baseline_imputation
from autopopulus.task_logic.utils import AE_METHOD_SETTINGS, ImputerT
from autopopulus.utils.get_set_cli_args import init_cli_args, load_cli_args
from autopopulus.utils.log_utils import (
    SERIALIZED_AE_IMPUTER_MODEL_FORMAT,
    get_serialized_model_path,
    load_artifact,
    mlflow_end,
    mlflow_init,
)
from autopopulus.utils.utils import (
    resample_indices_only,
    seed_everything,
)


def main():
    """
    Only for autoencoder imputation, as for pytorch while training can be
    distributed across multiple GPUs, testing must be done on one GPU.
    It is also most recommended to put testing in a separate script.
    """
    load_cli_args()
    args = init_cli_args()

    if not args.runtest:  # Do nothing
        return

    seed_everything(args.seed)
    mlflow_init(args)

    parent_hash = getattr(args, "parent_hash", None)
    imputer_type = ImputerT.type(args.method)
    if imputer_type == ImputerT.AE:
        evaluate_autoencoder_imputer(args, parent_hash)
    elif imputer_type == ImputerT.BASELINE:
        evaluate_baseline_imputer(args, parent_hash)
    mlflow_end()


def evaluate_baseline_imputer(args: Namespace, parent_hash: Optional[str] = None):
    """Similar to AE except we expect the model has already been run on test and we don't need to load the  model and run it."""
    # get model output on test
    imputed_data, _ = load_artifact("imputed_data", "pkl", run_id=parent_hash)
    # get test data
    test_data = load_artifact(
        f"{args.data_type_time_dim.name}_test_dataloader", "pt", run_id=parent_hash
    )

    evaluate_baseline_imputation(
        args,
        split="test",
        pred=imputed_data["test"],
        input_data=test_data["data"],
        true=test_data["ground_truth"],
        col_idxs_by_type=test_data["col_idxs_by_type"],
        semi_observed_training=test_data["semi_observed_training"],
        bootstrap=args.bootstrap_evaluate_imputer,
    )


def evaluate_autoencoder_imputer(args: Namespace, parent_hash: Optional[str] = None):
    test_dataloader = torch.load(
        get_serialized_model_path(
            f"{args.data_type_time_dim.name}_test_dataloader",
            "pt",
            run_id=parent_hash,
        )
    )
    imputer = AEImputer.from_checkpoint(
        args,
        get_serialized_model_path(
            SERIALIZED_AE_IMPUTER_MODEL_FORMAT.format(
                data_type_time_dim=args.data_type_time_dim.name
            ),
            "pt",
            run_id=parent_hash,
        ),
    )

    if args.bootstrap_evaluate_imputer:
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
    import sys

    sys.argv += ["--parent-hash", "b4b49a9117cb41afadef3f0573c8c38e"]
    main()
