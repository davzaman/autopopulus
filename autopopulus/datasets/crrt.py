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
from autopopulus.utils.utils import rank_zero_print
from autopopulus.data.constants import PATIENT_ID, TIME_LEVEL
from autopopulus.data.utils import regex_safe_colname

RACE_COLS = [
    f"RACE_{race}"
    for race in [
        "American Indian or Alaska Native",
        "Asian",
        "Black or African American",
        "Multiple Races",
        "Native Hawaiian or Other Pacific Islander",
        "Unknown",
        "White or Caucasian",
    ]
]

CRRT_STATIC_FILE_NAME = "df_[startdate-7d,startdate].parquet"
CRRT_LONGITUDINAL_FILE_NAME = "df_1Dagg_[startdate-7d,startdate].parquet"
CRRT_BIN_COLS = ["SEX", "ETHNICITY", "surgery_indicator"]
CRRT_CAT_COLS = CRRT_BIN_COLS + RACE_COLS
CRRT_ONEHOT_PREFIXES = ["ALLERGEN_ID", "RACE", "TOBACCO_USER", "SMOKING_TOB_STATUS"]
"""
Prefix a OR b = (a|b) followed by _ and 1+ characters of any char.
{ diagnoses: dx, meds: PHARM_SUBCLASS, problems: pr, procedures: CPT }
"""
CATEGORICAL_COL_REGEX = r"(dx|PHARM_SUBCLASS|pr|CPT|)_.*"


class CrrtDataLoader(AbstractDatasetLoader):
    def __init__(
        self,
        crrt_data_path: str,
        crrt_static_file_name: str = CRRT_STATIC_FILE_NAME,
        crrt_longitudinal_file_name: str = CRRT_LONGITUDINAL_FILE_NAME,
        crrt_categorical_cols: List[str] = CRRT_CAT_COLS,
        crrt_onehot_prefixes: List[str] = CRRT_ONEHOT_PREFIXES,
        crrt_target: str = "recommend_crrt",
        crrt_missingness_threshold: float = 0.8,
    ) -> None:
        self.data_path = crrt_data_path
        # keys: static, longitudinal
        self.preproc_file_name = crrt_static_file_name
        self.longitudinal_file_name = crrt_longitudinal_file_name
        # Set in `load_features_and_labels` using the df cols and a regex
        self.categorical_cols = crrt_categorical_cols
        self.onehot_prefixes = crrt_onehot_prefixes
        self.target = crrt_target
        self.missingness_threshold = crrt_missingness_threshold

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

        if data_type_time_dim == DataTypeTimeDim.STATIC:
            static_features = pd.read_parquet(
                join(self.data_path, "static_data.parquet")
            )
            # Merge would mess it up since static doesn't have UNIVERSAL_TIME_COL_NAME, join will broadcast.
            preprocessed_df = preprocessed_df.join(static_features, how="inner")
        mapping = {"IP_PATIENT_ID": PATIENT_ID, "DATE": TIME_LEVEL}
        preprocessed_df.index.names = [
            mapping[name] if name in mapping else name
            for name in preprocessed_df.index.names
        ]
        preprocessed_df = self.preprocess_data(preprocessed_df)
        df, labels = (
            preprocessed_df.drop(self.target, axis=1),
            preprocessed_df[self.target],
        )
        df.columns = self.sanitize_col_names(df.columns)
        self.categorical_cols = self.sanitize_col_names(pd.Index(self.categorical_cols))

        # set after preproc so we don't include dropped cols.
        self.categorical_cols = df.columns.intersection(self.categorical_cols)
        self.continuous_cols = df.columns.difference(self.categorical_cols)
        # keep only onehot prefixes that remain
        self.onehot_prefixes = [
            col
            for col in self.onehot_prefixes
            if df.columns.str.contains(f"^{regex_safe_colname(col)}").any()
        ]

        # return (df.drop("patient_id", axis=1), labels)
        return (df, labels)

    def sanitize_col_names(self, cols: pd.Index) -> pd.Index:
        return (
            cols.str.replace("&", "and")
            .str.replace("+", "plus")
            .str.replace(",", "")
            .str.replace("'", "")
            .str.replace("(", "__")
            .str.replace(")", "__")
            .str.replace("%", "_PRCNT")
        )

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes the data for use by ML model."""

        drop_columns = [
            "Month",
            "Hospital name",
            "CRRT Total Days",
            "CRRT Year",
            "End Date",
            "Machine",
            "ICU",
            "Recov. renal funct.",
            "Transitioned to HD",
            "Comfort Care",
            "Expired ",
            "KNOWN_DECEASED",
        ]
        df = df.drop(drop_columns, axis=1)
        # Get rid of "Unnamed" Column
        df = df.drop(df.columns[df.columns.str.contains("^Unnamed")], axis=1)

        # Exclude pediatric data, adults considered 21+
        is_adult_mask = df["AGE"] >= 21
        df = df[is_adult_mask]

        df = df.select_dtypes(["number"])

        acceptable_missing_cols = df.isna().mean() < self.missingness_threshold
        rank_zero_print(
            f"Dropping the following columns for missing more than {self.missingness_threshold*100}% data:\n{df.columns[~acceptable_missing_cols]}"
        )
        df = df[df.columns[acceptable_missing_cols]]
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
    from autopopulus.utils.get_set_cli_args import init_cli_args, load_cli_args
    import sys

    load_cli_args()
    sys.argv[sys.argv.index("--dataset") + 1] = "crrt"
    args = init_cli_args()
    data_loader = CrrtDataLoader.from_argparse_args(args)
    X, y = data_loader.load_features_and_labels(args.data_type_time_dim)
    print("Done!")
