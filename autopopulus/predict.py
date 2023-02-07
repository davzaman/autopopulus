from os.path import join
import pickle as pk


from autopopulus.models.prediction_models import Predictor
from utils.get_set_cli_args import init_cli_args, load_cli_args
from autopopulus.utils.utils import rank_zero_print, seed_everything
from utils.log_utils import BasicLogger


def main():
    """
    Only for making predictions on imputed data.
    It is also most recommended to put testing in a separate script from training.
    """
    load_cli_args()
    args = init_cli_args()

    seed_everything(args.seed)

    pickled_imputed_data_path = join("serialized_models", "imputed_data.pkl")
    rank_zero_print("Loading pickled imputed data...")
    with open(pickled_imputed_data_path, "rb") as file:
        imputed_data, labels = pk.load(file)

    rank_zero_print(f"Beginning downstream prediction on {args.data_type_time_dim}")

    predictor = Predictor.from_argparse_args(
        args,
        base_logger_context=BasicLogger.get_base_context_from_args(args),
        experiment_name=args.experiment_name,
        parent_run_hash=args.aim_hash if hasattr(args, "aim_hash") else None,
        data_type_time_dim=args.data_type_time_dim,
    )
    predictor.fit(imputed_data, labels)


if __name__ == "__main__":
    main()
