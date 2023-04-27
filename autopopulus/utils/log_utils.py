from argparse import Namespace
from logging import FileHandler, StreamHandler, basicConfig, INFO
from shutil import copy
from typing import Any, Dict, List, Optional, Union

from regex import search
from os.path import join, exists
from os import makedirs, walk
import sys

from torch.utils.tensorboard import SummaryWriter

from aim import Run, Text

TUNE_LOG_DIR = "tune_results"
LOGGER_TYPE = "TensorBoard"
# LOGGER_TYPE = "Aim"

# Ref: https://stackoverflow.com/a/6794451/1888794
# Import Logger everywhere you want to use a logger.
if LOGGER_TYPE == "TensorBoard":
    from pytorch_lightning.loggers import TensorBoardLogger
elif LOGGER_TYPE == "Aim":
    from aim.pytorch_lightning import AimLogger


# Treat like a class for easy semantics to compare to BasicLogger
def AutoencoderLogger(args: Optional[Namespace] = None):
    if LOGGER_TYPE == "TensorBoard":
        base_context = BasicLogger.get_base_context_from_args(args)
        return TensorBoardLogger(save_dir=BasicLogger.get_logdir(**base_context))
    elif LOGGER_TYPE == "Aim":
        # return AimLogger(experiment="lightning_logs")
        return AimLogger(experiment=args.experiment_name)


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
    modeln: str, ftype: str = "pkl", trial_num: Optional[int] = None
) -> str:
    """Path to dump serialized model, whether it's autoencoder, or predictive model."""
    dir_name = "serialized_models"
    if trial_num is not None:
        dir_name = join(TUNE_LOG_DIR, f"trial_{trial_num}", dir_name)
    if not exists(dir_name):
        makedirs(dir_name)
    serialized_model_path = join(dir_name, f"{modeln}.{ftype}")
    return serialized_model_path


class BasicLogger:
    def __init__(
        self,
        run_hash: Optional[str] = None,
        experiment_name: Optional[str] = None,
        predictive_model: Optional[str] = None,
        base_context: Optional[Dict[str, Any]] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        This is used for baseline_imputation, and prediction performance.
        If base context is passed, it uses it. Otherwise it looks for args.
        This is especially required if using tensorboard, otherwise you won't get a logger.
        """
        if base_context is None and args is not None:
            base_context = self.get_base_context_from_args(args)

        if LOGGER_TYPE == "TensorBoard":
            if base_context is None and args is None:
                self.logger = None
            else:
                # Get the universal logger for tensorboard.
                self.logger = SummaryWriter(
                    log_dir=self.get_logdir(
                        **base_context, predictive_model=predictive_model
                    )
                )
        elif LOGGER_TYPE == "Aim":
            # return Run(repo=target_path)
            self.logger = Run(run_hash=run_hash, experiment=experiment_name)
            self.logger["hparams"] = base_context

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
            ]
        }

    @classmethod
    @staticmethod
    def get_logdir(
        method: str,  # impute method
        amputation_patterns: Optional[List[Dict]] = None,
        percent_missing: Optional[float] = None,
        trial_num: Optional[int] = None,
        predictive_model: Optional[str] = None,
    ) -> str:
        """Get logging directory based on experiment settings."""
        # if tune_prefix is empty string, os.path.join will ignore it
        tune_prefix = (
            join(TUNE_LOG_DIR, f"trial_{trial_num}") if trial_num is not None else ""
        )
        # Missingness scenario could be 1 mech or mixed
        if amputation_patterns:
            pattern_mechanisms = ",".join(
                [pattern["mechanism"] for pattern in amputation_patterns]
            )
            dir_name = join(
                tune_prefix, "F.O.", str(percent_missing), pattern_mechanisms, method
            )
        else:
            dir_name = join(tune_prefix, "full", method)

        if predictive_model:
            dir_name = join(dir_name, predictive_model)

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
        if isinstance(self.logger, SummaryWriter):
            if tb_name_format is not None:
                logged_name = tb_name_format.format(name=name, **context)
            else:
                logged_name = name
            self.logger.add_scalar(logged_name, metric, global_step, walltime)
        elif isinstance(self.logger, Run):
            self.logger.track(metric, name, global_step, context={**context})

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
            elif isinstance(self.logger, Run):
                self.logger.add_text(
                    Text(text_string),
                    tag,
                    global_step,
                    context={**context},
                )

    def close(self):
        if self.logger is not None:
            if isinstance(self.logger, Run):
                print(f"Aim Logger Hash: {self.logger.hash}")
            self.logger.close()


def copy_log_from_tune(
    best_tune_logdir: str, logdir: str, logger: SummaryWriter = None
):
    """
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
