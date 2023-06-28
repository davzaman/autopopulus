from argparse import Namespace
from logging import FileHandler, StreamHandler, basicConfig, INFO
from shutil import copy, rmtree, copytree
from typing import Any, Dict, List, Optional, Union

from cloudpickle import dump, load
from regex import search
from os.path import join, exists, dirname
from os import makedirs, walk
import sys

from torch.utils.tensorboard import SummaryWriter

from ray import air
from ray.air.integrations.mlflow import MLflowLoggerCallback
from aim import Text, Run as AimRun
import mlflow
from mlflow.entities import Run as MlFlowRun
from pytorch_lightning.loggers import Logger

from autopopulus.data.types import DataTypeTimeDim
from autopopulus.utils.utils import rank_zero_print

TUNE_LOG_DIR = "tune_results"
# LOGGER_TYPE = "TensorBoard"
LOGGER_TYPE = "mlflow"
# LOGGER_TYPE = "Aim"

"""
For tensorboard, to make sure baseline and AE are logging using tensorboard the same way
We split context between guild flags and stuff in the metric (which will go into the TB tag).

IMPUTE:
split: train/val/test
feature_space: original/mapped
filter_subgroup: all/missingonly
reduction: CW/EW/NA (columnwise/errorwise/not applicable)

PREDICT:
aggregate_type: mean/cilower/ciupper/none
"""
IMPUTE_METRIC_TAG_FORMAT = (
    "{split}/{feature_space}/{filter_subgroup}/{reduction}/{feature_type}/{name}"
)
PREDICT_METRIC_TAG_FORMAT = "{predictor}/{aggregate_type}/{name}"
TIME_TAG_FORMAT = "{split}/epoch_duration_sec"
MIXED_FEATURE_METRIC_FORMAT = "{ctn_name}_{cat_name}"
SERIALIZED_AE_IMPUTER_MODEL_FORMAT = "AEDitto_{data_type_time_dim}"
SERIALIZED_PREDICTOR_FORMAT = "{model}_predictor"


# Ref: https://stackoverflow.com/a/6794451/1888794
# Import Logger everywhere you want to use a logger.
if LOGGER_TYPE == "TensorBoard":
    from pytorch_lightning.loggers import TensorBoardLogger
elif LOGGER_TYPE == "Aim":
    from aim.pytorch_lightning import AimLogger
elif LOGGER_TYPE == "mlflow":
    from pytorch_lightning.loggers import MLFlowLogger


# Treat like a class for easy semantics to compare to BasicLogger
def AutoencoderLogger(args: Optional[Namespace] = None):
    if LOGGER_TYPE == "TensorBoard":
        base_context = BasicLogger.get_base_context_from_args(args)
        return TensorBoardLogger(save_dir=BasicLogger.get_logdir(**base_context))
    elif LOGGER_TYPE == "Aim":
        # return AimLogger(experiment="lightning_logs")
        return AimLogger(experiment=args.experiment_name)
    elif LOGGER_TYPE == "mlflow":
        # optionally continue the same run
        parent_hash = getattr(args, "parent_hash", None)
        return MLFlowLogger(
            experiment_name=args.experiment_name,
            run_name=args.method,
            run_id=parent_hash,
            tags=vars(args),
            log_model=True,
        )


def init_sys_logger(fname: Optional[str] = None):
    handlers = [StreamHandler(sys.stdout)]
    if fname is not None:
        handlers.append(FileHandler(fname))

    basicConfig(
        level=INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # print to stdout and log to file
        handlers=handlers,
    )


def get_serialized_model_path(
    modeln: str,
    ftype: str = "pkl",
    trial_num: Optional[int] = None,
    parent_hash: Optional[str] = None,
) -> str:
    """Path to dump serialized model, whether it's autoencoder, or predictive model."""
    dir_name = "serialized_models"
    if trial_num is not None:
        dir_name = join(TUNE_LOG_DIR, f"trial_{trial_num}", dir_name)
    if not exists(dir_name):
        makedirs(dir_name)
    serialized_model_path = join(dir_name, f"{modeln}.{ftype}")

    if LOGGER_TYPE == "mlflow":
        # Sandbox the resources (model, dat, etc) for a run if using mlflow
        base_path = mlflow.get_run(run_id=parent_hash).info.artifact_uri.replace(
            "file://", ""
        )
    else:
        base_path = ""
    return join(base_path, serialized_model_path)


def load_artifact(
    objn: str,
    ftype: str = "pkl",
    trial_num: Optional[int] = None,
    parent_hash: Optional[str] = None,
) -> Any:
    path = get_serialized_model_path(
        objn, ftype, trial_num=trial_num, parent_hash=parent_hash
    )
    rank_zero_print(f"Loading pickled {objn}...")
    with open(path, "rb") as file:
        obj = load(file)
    return obj


def dump_artifact(
    obj,
    objn: str,
    ftype: str = "pkl",
    trial_num: Optional[int] = None,
    parent_hash: Optional[str] = None,
):
    if LOGGER_TYPE == "mlflow":
        active_run = mlflow.active_run()
        if active_run:
            parent_hash = active_run.info.run_id

    path = get_serialized_model_path(
        objn, ftype, trial_num=trial_num, parent_hash=parent_hash
    )
    makedirs(dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        dump(obj, file)

    # TODO: get active run or attach to parent run
    if LOGGER_TYPE == "mlflow":
        mlflow.log_artifact(path, path)


class BasicLogger:
    def __init__(
        self,
        run_hash: Optional[str] = None,
        experiment_name: Optional[str] = None,
        verbose: bool = False,
        base_context: Optional[Dict[str, Any]] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        This is used for baseline_imputation, and prediction performance.
        If base context is passed, it uses it. Otherwise it looks for args.
        This is especially required if using tensorboard, otherwise you won't get a logger.
        """
        self.verbose = verbose
        if base_context is None and args is not None:
            base_context = self.get_base_context_from_args(args)

        if LOGGER_TYPE == "TensorBoard":
            if base_context is None and args is None:
                self.logger = None
            else:  # Get the universal logger for tensorboard.
                self.logger = SummaryWriter(log_dir=self.get_logdir(**base_context))
        elif LOGGER_TYPE == "Aim":
            # return AimRun(repo=target_path)
            self.logger = AimRun(run_hash=run_hash, experiment=experiment_name)
            self.logger["hparams"] = base_context
        elif LOGGER_TYPE == "mlflow":
            mlflow.set_experiment(experiment_name=experiment_name)
            mlflow.start_run(
                run_hash, run_name=args.method if args is not None else None
            )
            self.logger: MlFlowRun = mlflow.get_run(
                run_id=mlflow.active_run().info.run_id
            )

        self.base_context = base_context

    @classmethod
    @staticmethod
    def get_base_context_from_args(args: Namespace) -> Dict[str, Any]:
        return {
            k: getattr(args, k) if hasattr(args, k) else None
            for k in [
                "method",
                "amputation_patterns",
                "percent_missing",
                "trial_num",
                "data_type_time_dim",
            ]
        }

    @classmethod
    @staticmethod
    def get_logdir(
        method: str,  # impute method
        amputation_patterns: Optional[List[Dict]] = None,
        percent_missing: Optional[float] = None,
        trial_num: Optional[int] = None,
        data_type_time_dim: Optional[DataTypeTimeDim] = None,
        put_impute_flags_in_path: bool = False,
    ) -> str:
        """
        Get logging directory based on experiment settings.
        This is the scalars "path" in tensorboard and guild.
        """
        # if tune_prefix is empty string, os.path.join will ignore it
        tune_prefix = ""
        if data_type_time_dim is not None:
            tune_prefix = join(tune_prefix, data_type_time_dim.name)
        if trial_num is not None:
            tune_prefix = join(tune_prefix, TUNE_LOG_DIR, f"trial_{trial_num}")
        if put_impute_flags_in_path:
            # Missingness scenario could be 1 mech or mixed
            if amputation_patterns:
                pattern_mechanisms = ",".join(
                    [pattern["mechanism"] for pattern in amputation_patterns]
                )
                dir_name = join(
                    tune_prefix,
                    "F.O.",
                    str(percent_missing),
                    pattern_mechanisms,
                    method,
                )
            else:
                dir_name = join(tune_prefix, "full", method)
        else:  # Flags wont go into path, use guild to get that info
            dir_name = join(tune_prefix, method)

        if not exists(dir_name):
            makedirs(dir_name)
        return dir_name

    def add_scalar(
        self,
        metric: float,
        name: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        # formatting name for tensorboard using context, relies on "{name}" being used in the format string
        tb_name_format: Optional[str] = None,
    ):
        if not self.logger:
            return
        if tb_name_format is not None:
            logged_name = tb_name_format.format(name=name, **context)
        else:
            logged_name = name

        if self.verbose:
            if global_step is not None:
                rank_zero_print(f"{logged_name}[{global_step}]: {metric}")
            else:
                rank_zero_print(f"{logged_name}: {metric}")

        if isinstance(self.logger, SummaryWriter):
            self.logger.add_scalar(logged_name, metric, global_step, walltime)
        elif isinstance(self.logger, AimRun):
            # this doesn't use logged_name
            self.logger.track(metric, name, global_step, context={**context})
        elif isinstance(self.logger, MlFlowRun):
            mlflow.log_metric(key=logged_name, value=metric, step=global_step)

    def add_scalars(
        self,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        tb_name_format: Optional[str] = None,
    ) -> None:
        """Adds scalars from dict but not to same plot."""
        if not self.logger:
            return
        for name, metric in tag_scalar_dict.items():
            self.add_scalar(
                metric,
                name,
                global_step,
                walltime,
                context=context,
                tb_name_format=tb_name_format,
            )

    def add_all_text(
        self,
        tag_scalar_dict: Dict[str, str],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds text from dict."""
        if not self.logger:
            return
        for tag, text_string in tag_scalar_dict.items():
            if isinstance(self.logger, SummaryWriter):
                self.logger.add_text(tag, text_string, global_step, walltime)
            elif isinstance(self.logger, AimRun):
                self.logger.add_text(
                    Text(text_string),
                    tag,
                    global_step,
                    context={**context},
                )
            elif isinstance(self.loger, MlFlowRun):
                mlflow.log_text(text_string, f"{tag}.txt")

    def close(self):
        if self.logger is not None:
            if isinstance(self.logger, AimRun):
                rank_zero_print(f"Logger Hash: {self.logger.hash}")
            elif isinstance(self.logger, MlFlowRun):
                rank_zero_print(f"Logger Hash: {self.logger.info.run_id}")
                mlflow.end_run()
            else:
                self.logger.close()


def copy_log_from_tune(
    best_tune_logdir: str, logdir: str, logger: SummaryWriter = None
):
    """
    NOTE: This is for older version of Ray.
    We want to copy these over locally so we can remove the tune files and readily compare the output later.
    Walk through the best tune run logdirectory,
        ignoring top-level (which is tune metadata we don't care about),
        and copy over all tfevents.
    """
    for root, dirs, files in walk(best_tune_logdir):
        if root != best_tune_logdir:  # ignore top-level
            for file in files:
                if search("tfevents", file):  # only tfevent files
                    copy(join(root, file), logdir)


def metric_path_from_run_id(run_id: str) -> str:
    return join(
        dirname(mlflow.get_run(run_id).info.artifact_uri).replace("file://", ""),
        "metrics",
    )


def copy_artifacts_from_tune(
    best_result: air.Result,
    model_path: str,
    logger: Logger,
    mlflow_callback: Optional[MLflowLoggerCallback] = None,
):
    """
    Copy metrics and model.
    Ray Tune output in 2.4.0 looks like:
    long_tune_name/
        checkpoint_000<num>/
            model --> KEEP
        rank_0/
            checkpoints/
                *.ckpt
            *tfevents*
            hparams.yaml
        rank_.../
        ...
        *tfevents* --> KEEP
        params.json --> KEEP
        params.pkl (just a pkl of the json)
        progress.csv
        result.json --> KEEP

    We don't want the per-rank tfevents, just the high level one.
    """
    # Model path should already exist
    if isinstance(logger, MLFlowLogger):
        # don't want a to_path, mlflow will figure it out
        save_fn = lambda from_path, to_path: logger.experiment.log_artifact(
            logger._run_id, from_path, to_path
        )
        keep_regex = ".*.json"
        metric_path = None
    else:
        save_fn = copy
        keep_regex = "tfevents|.*.json"
        metric_path = logger.save_dir
        if not exists(metric_path):
            makedirs(metric_path)

    save_fn(join(best_result.checkpoint.path, "model"), model_path)

    # copy metrics
    for root, dirs, files in walk(best_result.log_dir):
        if root == str(best_result.log_dir):  # only look at high level (not per rank)
            for file in files:  # only tfevent and json files
                if search(keep_regex, file):
                    save_fn(join(root, file), metric_path)

    # above won't copy over metrics saved separately/synced to mlflow
    # ray-tune will save it disjoint for each epoch
    if isinstance(logger, MLFlowLogger):
        # copy over metrics if mlflow
        best_mlflow_run_id = {
            str(trial): run_id for trial, run_id in mlflow_callback._trial_runs.items()
        }[best_result.metrics["trial_id"]]
        copytree(
            metric_path_from_run_id(best_mlflow_run_id),
            metric_path_from_run_id(logger._run_id),
            dirs_exist_ok=True,
        )

    rmtree(TUNE_LOG_DIR)  # delete tune files
