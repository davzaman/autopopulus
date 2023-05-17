from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from pickle import dump
from numpy import percentile
from scipy.stats import sem, t, shapiro

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor, cat, sort


def bootstrap_confidence_interval(
    metric: List[float], confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Returns confidence interval at given confidence level for statistical
    distributions established with bootstrap sampling.
    """
    alpha = 1 - confidence_level
    lower = alpha / 2 * 100
    upper = (alpha / 2 + confidence_level) * 100
    return percentile(metric, lower), percentile(metric, upper)


def confidence_interval(
    metric: List[float], confidence_level: float = 0.95
) -> float:  # ) -> Tuple[float, float]:
    """Returns confidence interval at given confidence level for data on metric.
    Assumes normality and will produce symmetric bounds.
    Note if sem is 0 st.t.interval will throw a runtime error.
    Ref: https://stackoverflow.com/questions/15033511
    """
    """
    # for # bootstrap samples < 100 it is better to look up in t dist table
    # if we want to use z use st.norm.interval instead.
    return st.t.interval(confidence_level, len(metric) - 1,
                         loc=np.mean(metric), scale=st.sem(metric))
    """
    # mean = np.mean(metric)
    half_interval = sem(metric) * t.ppf((1 + confidence_level) / 2.0, len(metric) - 1)
    return half_interval
    # return (mean - half_interval, mean + half_interval)


def shapiro_wilk_test(metric: List[float], confidence_level: float = 0.95) -> bool:
    """Outputs W,p. We want p > (alpha = 1 - CI = .05 usually)
    Ref: https://machinelearningmastery.com/
        a-gentle-introduction-to-normality-tests-in-python/
    p > alpha: Sample looks Gaussian (fail to reject H0)
         else: Sample does not look Gaussian (reject H0)
    """
    return shapiro(metric)[1] > (1 - confidence_level)


class ErrorAnalysisCallback(Callback):
    def __init__(self, limit_n: int = 10) -> None:
        super().__init__()
        self.limit_n = limit_n
        self.best_n: Dict[str, Dict[str, Tensor]] = {}
        self.worst_n: Dict[str, Dict[str, Tensor]] = {}

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # pickle the results
        with open(f"error_analysis/best_{self.limit_n}", "wb") as file:
            dump(self.best_n, file)

        with open(f"error_analysis/worst_{self.limit_n}", "wb") as file:
            dump(self.worst_n, file)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ):
        for name, metricfn in pl_module.metrics.items():
            metric_fn_args = [outputs["pred"], outputs["ground_truth"]]
            self.compute_metrics_and_merge_batch(
                name, metricfn, outputs, metric_fn_args
            )

            # Compute metrics for missing only data
            missing_only_mask = ~(outputs["non_missing_mask"].bool())
            if missing_only_mask.any():
                metric_fn_args.append(missing_only_mask)
                self.compute_metrics_and_merge_batch(
                    name, metricfn, outputs, metric_fn_args
                )

    def compute_metrics_and_merge_batch(
        self,
        name: str,
        metricfn: Callable,
        outputs: Dict[str, Tensor],
        metric_fn_args: List[Any] = None,
    ):
        result = metricfn(*metric_fn_args, reduction="none")

        # sort the metric values/results
        sorted_result, indices = sort(result)
        # get n best and worst values, and get their corresponding observations
        best_n = {
            "pred": outputs["pred"].iloc[indices[:10]],
            "true": outputs["ground_truth"].iloc[indices[:10]],
            "values": sorted_result[:10],
        }
        worst_n = {
            "pred": outputs["pred"].iloc[indices[-10:]],
            "true": outputs["ground_truth"].iloc[indices[-10:]],
            "values": sorted_result[-10:],
        }

        if name in self.best_n:  # update
            self.merge_batch(best_n, worst_n, name)
        else:  # add in directly
            self.best_n[name] = best_n
            self.worst_n[name] = worst_n

    def merge_batch(
        self, best_n: Dict[str, Tensor], worst_n: Dict[str, Tensor], name: str
    ):
        # sort 20 values (10 from best/worst 10, and then 10 best/worst from newest batch.)
        sorted_values, indices = sort(
            cat(self.best_n[name]["values"], best_n["values"])
        )

        self.best_n[name] = {
            "pred": cat(self.best_n[name]["pred"], best_n["pred"]).iloc[indices[:10]],
            "true": cat(self.best_n[name]["true"], best_n["true"]).iloc[indices[:10]],
            "values": sorted_values[:10],
        }

        self.worst_n[name] = {
            "pred": cat(self.worst_n[name]["pred"], worst_n["pred"]).iloc[
                indices[-10:]
            ],
            "true": cat(self.worst_n[name]["true"], worst_n["true"]).iloc[
                indices[-10:]
            ],
            "values": sorted_values[-10:],
        }
