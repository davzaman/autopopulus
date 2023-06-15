"""
Deprecated until Optuna devs update their pytorch lightning integration with recent versions.
"""
from argparse import Namespace
from typing import Any, Dict
from os import getcwd
from os.path import join
from shutil import copytree, rmtree

from torch.cuda import empty_cache

from optuna import Study, create_study
from optuna.trial import FrozenTrial, Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from autopopulus.data.dataset_classes import CommonDataModule
from autopopulus.models.ap import AEImputer
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    SERIALIZED_AE_IMPUTER_MODEL_FORMAT,
    AutoencoderLogger,
    BasicLogger,
    copy_log_from_tune,
    get_serialized_model_path,
    TUNE_LOG_DIR,
)


def create_autoencoder(
    args: Namespace, data: CommonDataModule, settings: Dict
) -> AEImputer:
    # TODO: this is probably broken now that i've changed everything about logging.
    logdir = BasicLogger.get_logdir(BasicLogger.get_base_context_from_args(args))
    if args.tune_n_samples:
        best_trial_num, best_model_config = run_tune(args, data, settings)
        args.trial_num = best_trial_num
        best_tune_logdir = BasicLogger.get_logdir(
            BasicLogger.get_base_context_from_args(args)
        )
        # Copy serialized model and logged values to local path, ignoring tune artifacts
        copy_log_from_tune(best_tune_logdir, logdir)
        copytree(
            join(best_tune_logdir, "serialized_models"),
            join(getcwd(), "serialized_models"),
            # ignore=ignore_patterns(
            # "checkpoint_*", "params.*", "progress.csv", "result.json", "*tfevents*"
            # ),
            dirs_exist_ok=True,
        )

        # Load up (now we can do from local because we copied over)
        best_checkpoint = get_serialized_model_path(
            SERIALIZED_AE_IMPUTER_MODEL_FORMAT.format(
                data_type_time_dim=data.data_type_time_dim.name
            ),
            "pt",
            best_trial_num,
        )
        ae_imputer = AEImputer.from_checkpoint(
            args, ae_from_checkpoint=best_checkpoint, **best_model_config
        )

        # Cleanup
        empty_cache()
        rmtree(TUNE_LOG_DIR)  # delete tune files

        return ae_imputer

    # If not tuning assume we've been given a specific setting for hyperparams
    logger = AutoencoderLogger(logdir)
    ae_imputer = AEImputer.from_argparse_args(
        args,
        logger=logger,
        tune_callback=None,
        **settings,
    )
    ae_imputer.fit(data)
    return ae_imputer


def get_tune_grid(args: Namespace, trial: Trial) -> Dict[str, Any]:
    if args.fast_dev_run or args.limit_data:
        hidden_layers_grid = [[0.5]]
        max_epochs = [3]
    else:
        hidden_layers_grid = [
            [0.5, 0.25, 0.5],
            [0.5],
            [1.0, 0.5, 1.0],
            [1.5],
            [1.0, 1.5, 1.0],
        ]
        max_epochs = [100, 200]

    # Ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "l2_penalty": trial.suggest_loguniform("l2_penalty", 1e-5, 1),
        # cannot suggest a list so i will pick an index to my list of lists
        "hidden_layers": hidden_layers_grid[
            trial.suggest_int("hidden_layers", 0, len(hidden_layers_grid) - 1)
        ],
        # "max_epochs": trial.suggest_categorical("max_epochs", max_epochs),
        # "patience": trial.suggest_categorical("patience", [3,5,10]),
    }

    return config


def tune_model_optuna(
    args: Namespace, data: CommonDataModule, settings: Dict[str, Any], trial: Trial
) -> float:
    args.trial_num = trial.number

    config = get_tune_grid(args, trial)
    logger = AutoencoderLogger(args)
    metric = IMPUTE_METRIC_TAG_FORMAT.format(
        name="MAAPE",
        feature_space="original",
        filter_subgroup="missingonly",
        reduction="CW",
        split="val",
        feature_type="mixed",
    )

    ae_imputer = AEImputer.from_argparse_args(
        args,
        logger=logger,
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.PyTorchLightningPruningCallback.html
        tune_callback=PyTorchLightningPruningCallback(trial, monitor=metric),
        **settings,
        **config,
    )
    ae_imputer.fit(data)
    return ae_imputer.trainer.callback_metrics[metric].item()


def run_tune(
    args: Namespace,
    data: CommonDataModule,
    settings: Dict[str, Any],
):
    # Ref: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
    study = create_study(
        study_name=args.experiment_name,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(),
    )

    # Ref: https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
    study.optimize(
        lambda trial: tune_model_optuna(args, data, settings, trial),
        n_trials=args.tune_n_samples,
    )

    best_trial: FrozenTrial = study.best_trial
    return (best_trial.number, best_trial.params)
