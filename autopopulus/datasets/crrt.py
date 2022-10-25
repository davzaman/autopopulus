from argparse import ArgumentParser
from typing import List, Optional, Union
from os.path import join
import pandas as pd

from autopopulus.data.dataset_classes import AbstractDatasetLoader
from autopopulus.data.types import (
    DataTypeTimeDim,
    LongitudinalFeatureAndLabel,
    StaticFeatureAndLabel,
)

CRRT_STATIC_FILE_NAME = "df_[startdate-7d,startdate].parquet"
CRRT_LONGITUDINAL_FILE_NAME = "df_1Dagg_[startdate-7d,startdate].parquet"
CRRT_CAT_COLS = []
CRRT_STATIC_CTN_COLS = []
CRRT_FLATTENED_LONGITUDINAL_COLS = []
CRRT_ONEHOT_PREFIXES = []


class CrrtDataLoader(AbstractDatasetLoader):
    def __init__(
        self,
        crrt_data_path: str,
        crrt_static_file_name: str = CRRT_STATIC_FILE_NAME,
        cure_ckd_longitudinal_file_name: str = CRRT_LONGITUDINAL_FILE_NAME,
        crrt_categorical_cols: List[str] = CRRT_CAT_COLS,
        crrt_static_continuous_cols: List[str] = CRRT_STATIC_CTN_COLS,
        crrt_longitudinal_cols: List[str] = CRRT_FLATTENED_LONGITUDINAL_COLS,
        crrt_onehot_prefixes: List[str] = CRRT_ONEHOT_PREFIXES,
        crrt_target: str = "recommend_crrt",
    ) -> None:
        self.data_path = crrt_data_path
        # keys: static, longitudinal
        self.preproc_file_name = crrt_static_file_name
        self.longitudinal_file_name = cure_ckd_longitudinal_file_name
        self.categorical_cols = crrt_categorical_cols
        self.static_continuous_cols = crrt_static_continuous_cols
        self.flattened_longitudinal_cols = crrt_longitudinal_cols
        self.onehot_prefixes = crrt_onehot_prefixes
        self.target = crrt_target

        self.continuous_cols = crrt_static_continuous_cols + crrt_longitudinal_cols

    def load_features_and_labels(
        self, data_type_time_dim: Optional[DataTypeTimeDim] = DataTypeTimeDim.STATIC
    ) -> Union[LongitudinalFeatureAndLabel, StaticFeatureAndLabel]:
        """
        Returns static or longitudinal forms of the dataset.
        When longitudinal not none will return the longitudinal portion vs static portion.
        """
        fname = (
            self.preproc_file_name
            if data_type_time_dim == DataTypeTimeDim.STATIC
            else self.longitudinal_file_name
        )
        preprocessed_df = pd.read_parquet(join(self.data_path, fname))
        df, labels = (
            preprocessed_df.drop(self.target, axis=1),
            preprocessed_df[self.target],
        )
        # return (df.drop("patient_id", axis=1), labels)
        return (self.preprocess_data(df), labels)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes the data for use by ML model."""

        drop_columns = [
            "Month",
            "Hospital name",
            "CRRT Total Days",
            "End Date",
            "Machine",
            "ICU",
            "Recov. renal funct.",
            "Transitioned to HD",
            "Comfort Care",
            "Expired ",
        ]
        df = df.drop(drop_columns, axis=1)
        # Get rid of "Unnamed" Column
        df = df.drop(df.columns[df.columns.str.contains("^Unnamed")], axis=1)
        # drop columns with all nan values
        df = df[df.columns[~df.isna().all()]]

        # Exclude pediatric data, adults considered 21+
        is_adult_mask = df["Age"] >= 21
        df = df[is_adult_mask]

        return df

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)

        p.add_argument(
            "--crrt-data-path",
            type=str,
            required=True,
            help="Path name to all the data required for the CRRT preprocesesd files.",
        )
        p.add_argument(
            "--crrt-static-file-name",
            type=str,
            default=CRRT_STATIC_FILE_NAME,
            help="Name of the file produced by the crrt repo in a static setting.",
        )
        p.add_argument(
            "--crrt-longitudinal-file-name",
            type=str,
            default=CRRT_LONGITUDINAL_FILE_NAME,
            help="Name of file that processes the flat df into longitudinal columns and longitudinal form (batch size, sequence length, features).",
        )

        return p


# Testing
if __name__ == "__main__":
    from autopopulus.main import init_cli_args
    from autopopulus.utils.cli_arg_utils import load_cli_args
    import sys

    load_cli_args()
    sys.argv[sys.argv.index("--dataset") + 1] = "crrt"
    args = init_cli_args()
    data_loader = CrrtDataLoader.from_argparse_args(args)
    X, y = data_loader.load_features_and_labels(args.data_type_time_dim)
