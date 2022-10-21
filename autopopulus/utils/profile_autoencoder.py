from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
import pandas as pd
import os
from numpy.random import default_rng
from torch import vstack

from autopopulus.data import CommonDataModule
from autopopulus.data.dataset_classes import SimpleDatasetLoader
from autopopulus.models.ae import AEDitto
from autopopulus.data.utils import onehot_multicategorical_column
from autopopulus.utils.cli_arg_utils import YAMLStringListToList

PROFILERS = {
    "simple": SimpleProfiler(dirpath="profiling-results", filename="simple"),
    "advanced": AdvancedProfiler(dirpath="profiling-results", filename="advanced"),
    "pytorch": PyTorchProfiler(dirpath="profiling-results", filename="pytorch"),
}

if __name__ == "__main__":
    seed = 0
    nsamples = 1028
    nfeatures = 10
    fast_dev_run = True
    rng = default_rng(seed)

    p = ArgumentParser()
    p.add_argument("--profilers", type=str, default=None, action=YAMLStringListToList())
    args = p.parse_known_args()[0]
    profilers = args.profilers
    if profilers is not None:
        profilers = [PROFILERS[profiler_name] for profiler_name in profilers]
    else:
        profilers = [None]

    # Make data
    onehot_prefixes = ["multicatA", "multicatB"]
    bin_cols = [f"bin_{i}" for i in range(round(nfeatures / 2) - 2)]
    cat_cols = bin_cols + [
        "multicatA_0.0",
        "multicatA_1.0",
        "multicatA_2.0",
        "multicatB_0.0",
        "multicatB_1.0",
        "multicatB_2.0",
        "multicatB_3.0",
    ]
    ctn_cols = [f"ctn_{i}" for i in range(nfeatures // 2)]
    # create mixed dataset
    X = pd.concat(
        [
            # half categorical, sandwhiched between 2 multicat
            pd.Series(rng.integers(0, 3, nsamples), name="multicatA"),
            pd.DataFrame(
                rng.integers(0, 2, (nsamples, round(nfeatures / 2) - 2)),
                columns=bin_cols,
            ),
            pd.Series(rng.integers(0, 4, nsamples), name="multicatB"),
            # half continuous
            pd.DataFrame(
                rng.random((nsamples, nfeatures // 2)),
                columns=ctn_cols,
            ),
        ],
        axis=1,
    )
    X = onehot_multicategorical_column(onehot_prefixes)(X)

    # load into objects
    dataset_loader = SimpleDatasetLoader(
        X,
        pd.Series(rng.integers(0, 1, nsamples)),
        continuous_cols=ctn_cols,
        categorical_cols=cat_cols,
        onehot_prefixes=onehot_prefixes,
    )
    data = CommonDataModule(
        seed=seed,
        val_test_size=0.5,
        test_size=0.5,
        batch_size=4,
        num_gpus=1,
        scale=True,
        # feature_map="target_encode_categorical",
        feature_map="discretize_continuous",
        uniform_prob=True,
        dataset_loader=dataset_loader,
    )

    model = AEDitto(
        seed=seed,
        hidden_layers=[0.5],
        learning_rate=0.1,
        mvec=False,
        variational=False,
        activation="ReLU",
        optimn="Adam",
        lossn="BCE",
        datamodule=data,
    )

    # appends to file so remove old ones
    for filename in ["simple", "advanced", "pytorch"]:
        try:
            os.remove(os.path.join("profiling-results", filename))
        except OSError:
            pass
    for profiler in profilers:
        trainer = Trainer(
            max_epochs=3,
            fast_dev_run=fast_dev_run,
            deterministic=True,
            gpus=[0],
            accelerator="gpu",
            strategy=DDPPlugin(find_unused_parameters=False),
            enable_checkpointing=False,
            profiler=profiler,
        )
        trainer.fit(model, datamodule=data)

    # sanity check output
    loader = data.test_dataloader()
    preds_list = trainer.predict(model, loader)
    # stack the list of preds from dataloader
    preds = vstack(preds_list).cpu().numpy()
    columns = data.splits["data"]["train"].columns

    # Recover IDs, we use only indices used by the batcher (so if we limit to debug, this still works, even if it's shuffled)
    ids = (
        loader.dataset.split_ids["data"][: fast_dev_run * data.batch_size]
        if fast_dev_run
        else loader.dataset.split_ids["data"]
    )
    recovered_X = pd.DataFrame(preds, columns=columns, index=ids)
    recovered_X.to_csv("profiling-results/output_sanity_check.csv")
