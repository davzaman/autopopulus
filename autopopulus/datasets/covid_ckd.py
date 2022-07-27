from argparse import ArgumentParser
import sys
from os.path import join
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from autopopulus.data import (
    AbstractDatasetLoader,
    LongitudinalFeatureAndLabel,
    StaticFeatureAndLabel,
)
from autopopulus.utils.cli_arg_utils import YAMLStringDictToDict, YAMLStringListToList

COVID_PREPROC_FILE_NAME = "covid_ckd_covid19_stat_anal_v2.csv"
ENCODINGS = {"site-source": {"ucla": 0, "providence": 1}}

# Note: I cannot include features that are missing too much
# Otherwise, I will be left with 0 samples after filtering to fully observed
COVID_CTN_COLS = (
    [
        "registry_entry_age",
        # CONSTRUCTED_FEATURE: discharge - encounter day
        "days_hospitalized",
    ]
    + [f"AKIs_prior_entry_month_{month}" for month in range(1, 6)]
    # EGFR columns missing a ton of values
    # + [f"egfr_stdev_value_prior_entry_month_{month}" for month in range(1, 9)]
)

RACE_MAP = {
    "1A": "White Non-Latino",
    "1B": "White Latino",
    "2": "Black",
    "3": "Asian",
    "4": "American Indian",
    "5": "Hawaiian",
    "8": "Other",
    "-9": "Not Categorized",
}
# when one-hot-encoding it will sort the keys (strings), this will match the ordering
RACE_COLS = [f"ethnicity_{RACE_MAP[col]}" for col in sorted(RACE_MAP.keys())]

COVID_ONEHOT_PREFIXES = ["ethnicity"]

COVID_CAT_COLS = [
    "site_source",
    "sex_code",
    # "race_ethnicity_cat", # will be using RACE_COLS instead
    # "vital_status_code", # All are "not known deceased"/0
    "dm",
    "pdm",
    "htn",
    "ckd",
    # "pcornet_visit_type",  # needs one hot encoding
] + RACE_COLS

COVID_TARGETS = [
    # covid_ckd_19_status
    "positive_flag",
    "covid_ckd_admit_flag",
    "severe_flag",
    "icu_flag",
    "death_flag",
]
# DEFAULT_COVID_TARGET = COVID_TARGETS[2]  # or 0
COVID_DEFAULT_TARGET = COVID_TARGETS[0]  # or 2


class CovidCKDDataLoader(AbstractDatasetLoader):
    def __init__(
        self,
        covid_ckd_data_path: str,
        covid_ckd_missing_cols: List[str],
        covid_ckd_observed_cols: List[str],
        covid_ckd_subgroup_filter: Optional[Dict[str, str]] = None,
        covid_ckd_preproc_file_name: str = COVID_PREPROC_FILE_NAME,
        # time_points: List[str] = CKD_TIME_POINTS,
        covid_ckd_categorical_cols: List[str] = COVID_CAT_COLS,
        covid_ckd_continuous_cols: List[str] = COVID_CTN_COLS,
        covid_ckd_onehot_prefixes: List[str] = COVID_ONEHOT_PREFIXES,
        covid_ckd_target: str = COVID_DEFAULT_TARGET,
    ) -> None:
        self.data_path = covid_ckd_data_path
        self.missing_cols = covid_ckd_missing_cols
        self.observed_cols = covid_ckd_observed_cols
        self.subgroup_filter = covid_ckd_subgroup_filter
        self.preproc_file_name = covid_ckd_preproc_file_name
        self.categorical_cols = covid_ckd_categorical_cols
        self.continuous_cols = covid_ckd_continuous_cols
        self.onehot_prefixes = covid_ckd_onehot_prefixes
        self.target = covid_ckd_target

    def load_features_and_labels(
        self, longitudinal: bool = False
    ) -> Union[StaticFeatureAndLabel, LongitudinalFeatureAndLabel]:
        """Returns X and y for use with models."""
        df = pd.read_csv(join(self.data_path, self.preproc_file_name))
        df = self.fix_missing_value_representation(df)
        df = self.map_cat_vars(df)
        df = self.construct_additional_features(df)
        # do this last in case other processing steps needs site source
        df = self.filter_subgroup(df, ENCODINGS)

        features = COVID_CAT_COLS + COVID_CTN_COLS
        return (df[features], df[self.target])

    def map_cat_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps the categorical variables of the dataset, shift strangely coded binary vars."""
        mapping = {
            "site_source": {2.0: 1, 1.0: 0},  # 2/1 = providence, 1/0 = ucla
            "sex_code": {2.0: 1, 1.0: 0},  # 2/1 = female, 1/0 = male
            "ruca_code": {
                1.0: 0,
                1.1: 1,
            },  # https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes/documentation/
        }
        df = df.apply(
            lambda col: col.map(mapping[col.name]) if col.name in mapping else col
        )
        df[RACE_COLS] = self.one_hot_enc_race(df)
        return df

    def one_hot_enc_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns the 1hot encoded columns for race."""
        enc = OneHotEncoder(handle_unknown="ignore")
        one_hot_races = enc.fit_transform(
            df["race_ethnicity_cat"].values.reshape(-1, 1)
        ).toarray()
        return pd.DataFrame(data=one_hot_races, columns=RACE_COLS)

    def fix_missing_value_representation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Some missing values are represented as numerical values."""
        # egfr cols: missing is -999.99
        for egfr_col in [col for col in df.columns if col.startswith("egfr")]:
            df[egfr_col].replace(to_replace=-999.99, value=np.nan, inplace=True)

        # at ucla only, 0 is assumed to mean missing.
        # TODO[LOW]: but if they actually have 0 aki's how do we differentiate?
        # ucla_only = df["site_source"] == 1
        # for aki_col in [col for col in df.columns if col.startswith("AKI")]:
        #     df[aki_col][ucla_only].replace(to_replace=0.0, value=np.nan, inplace=True)

        # other cols where 0 means missing
        for zero_missing_col in ["severe_flag", "icu_flag", "death_flag"]:
            df[zero_missing_col].replace(to_replace=0.0, value=np.nan, inplace=True)

        return df

    def construct_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct any additional features from the dataset."""
        days_hospitalized = df["discharge_day"] - df["encounter_day"]
        df["days_hospitalized"] = days_hospitalized
        return df

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)

        p.add_argument(
            "--covid-ckd-data-path",
            type=str,
            required=True,
            help="Path name to all the data required for the COVID CKD registry + serialized files.",
        )
        p.add_argument(
            "--covid-ckd-subgroup-filter",
            type=Dict[str, str],
            action=YAMLStringDictToDict(),
            help="Dictionary of variable names to which group within that variable you want to filter down to.",
        )
        p.add_argument(
            "--covid-ckd-preproc-file-name",
            type=str,
            default=COVID_PREPROC_FILE_NAME,
            help="Name of file containing data for COVID CKD.",
        )
        p.add_argument(
            "--covid-ckd-categorical-cols",
            type=List[str],
            default=COVID_CAT_COLS,
            help="List of categorical columns in the dataset.",
        )
        p.add_argument(
            "--covid-ckd-continuous-cols",
            type=List[str],
            default=COVID_CTN_COLS,
            help="List of continuous columns in the dataset.",
        )
        p.add_argument(
            "--covid-ckd-onehot-prefixes",
            type=List[str],
            default=COVID_ONEHOT_PREFIXES,
            help="Prefixes for multi-categorical columns that were one-hot encoded.",
        )
        p.add_argument(
            "--covid-ckd-target",
            type=str,
            default=COVID_DEFAULT_TARGET,
            help="Name of target variable for training.",
        )
        p.add_argument(
            "--covid-ckd-missing-cols",
            required="--fully-observed" in sys.argv and "none" not in sys.argv,
            action=YAMLStringListToList(),
            help="List of columns in the dataset that will be masked when amputing.",
        )
        p.add_argument(
            "--covid-ckd-observed-cols",
            required="MAR" in sys.argv,
            action=YAMLStringListToList(),
            help="List of columns in the dataset to use for masking when amputing under MAR.",
        )

        return p


# Testing
if __name__ == "__main__":
    from main import init_cli_args
    from utils.cli_arg_utils import load_cli_args

    load_cli_args()
    args = init_cli_args()
    data_loader = CovidCKDDataLoader.from_argparse_args(args)
    X, y = data_loader.load_features_and_labels()
