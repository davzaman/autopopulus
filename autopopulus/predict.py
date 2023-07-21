import sys

from autopopulus.models.prediction_models import Predictor
from autopopulus.utils.get_set_cli_args import init_cli_args, load_cli_args
from autopopulus.utils.utils import rank_zero_print, seed_everything
from autopopulus.utils.log_utils import (
    BasicLogger,
    load_artifact,
    mlflow_end,
    mlflow_init,
)


def main():
    """
    Only for making predictions on imputed data.
    It is also most recommended to put testing in a separate script from training.
    """
    load_cli_args()
    args = init_cli_args()

    seed_everything(args.seed)
    mlflow_init(args)

    parent_hash = getattr(args, "parent_hash", None)
    imputed_data, labels = load_artifact("imputed_data", "pkl", run_id=parent_hash)
    rank_zero_print(f"Beginning downstream prediction on {args.data_type_time_dim}")
    predictor = Predictor.from_argparse_args(
        args,
        base_logger_context=BasicLogger.get_base_context_from_args(args),
        experiment_name=args.experiment_name,
        parent_run_hash=parent_hash,
        data_type_time_dim=args.data_type_time_dim,
    )
    predictor.fit(imputed_data, labels)
    mlflow_end()


if __name__ == "__main__":
    # sys.argv += ["--parent-hash", "b4b49a9117cb41afadef3f0573c8c38e"]
    main()
