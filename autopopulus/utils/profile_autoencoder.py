from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
)
import pandas as pd
import os
from numpy.random import default_rng

from autopopulus.data import CommonDataModule
from autopopulus.data.dataset_classes import SimpleDatasetLoader
from autopopulus.data.utils import onehot_multicategorical_column
from autopopulus.utils.cli_arg_utils import YAMLStringListToList
from autopopulus.models.ap import AEImputer

PROFILERS = {
    "simple": SimpleProfiler(dirpath="profiling-results", filename="simple"),
    "advanced": AdvancedProfiler(dirpath="profiling-results", filename="advanced"),
    # Ref: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.profilers.PyTorchProfiler.html#pytorch_lightning.profilers.PyTorchProfiler
    # https://pytorch.org/docs/master/profiler.html
    "pytorch": PyTorchProfiler(
        dirpath="profiling-results",
        filename="pytorch",
        # emit_nvtx=True,
        profile_memory=True,
        with_stack=True,
        record_module_names=True,
    ),
}

if __name__ == "__main__":
    seed = 0
    batch_size = 256
    # nsamples = 1000028
    nsamples = 100028
    nfeatures = 20
    fast_dev_run = 0
    rng = default_rng(seed)
    num_gpus = 1
    num_workers = 4
    optimn = "Adam"
    max_epoch = 1

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
        batch_size=batch_size,
        num_workers=num_workers,
        scale=True,
        # feature_map="target_encode_categorical",
        feature_map="discretize_continuous",
        uniform_prob=True,
        dataset_loader=dataset_loader,
    )

    # appends to file so remove old ones
    for filename in ["simple", "advanced", "pytorch"]:
        for name in ["fit", "predict"]:
            try:
                os.remove(os.path.join("profiling-results", f"{name}-{filename}.txt"))
            except OSError:
                pass

    for profiler in profilers:
        model = AEImputer(
            seed=seed,
            num_gpus=num_gpus,
            hidden_layers=[0.5],
            learning_rate=0.1,
            mvec=False,
            variational=False,
            activation="ReLU",
            optimn=optimn,
            lossn="BCE",
            datamodule=data,
            fast_dev_run=fast_dev_run,
            profiler=profiler,
            max_epochs=max_epoch,
        )

        model.fit(data)

    # sanity check output
    loader = data.test_dataloader()
    preds = model.transform(loader)
    preds.to_csv("profiling-results/output_sanity_check.csv")
