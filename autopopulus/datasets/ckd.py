"""
Uses repo `cure_ckd_preprocess` to create preprocessed dataset.
https://github.com/davzaman/cure_ckd_preprocess
"""

from argparse import ArgumentParser
import pickle
import sys
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from os.path import join
import pandas as pd
from functools import reduce
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from logging import error

#### Local Module ####
from autopopulus.data import (
    AbstractDatasetLoader,
    DataTypeTimeDim,
    LongitudinalFeatureAndLabel,
    StaticFeatureAndLabel,
)
from autopopulus.utils.cli_arg_utils import YAMLStringDictToDict, YAMLStringListToList


ENCODINGS = {
    "site-source": {"ucla": 0, "providence": 1},
    "egfr_entry_ckd_flag": {"atrisk": 0, "ckd": 1},
}

## Defaults ##
CKD_PREPROC_FILE_NAME = "cure_ckd_preprocessed.feather"
# all my preprocessing on top of what you get from using the preprocess repo
CKD_FLATTENED_DF_FILE_NAME = "cure_ckd_preprocessed_lvl2.feather"
# the preproc used feather, but we can't use featuer for multiindex
CKD_LONGITUDINAL_FILE_NAME = "cure_ckd_longitudinal.parquet"


def map_year_to_feature_name(year: int) -> str:
    return "entry_period" if year == 0 else f"year{year}"


RANGE_OF_YEARS = range(14)
# CKD_TIME_POINTS = ["entry_period"] + [f"year{i}" for i in RANGE_OF_YEARS]
CKD_TIME_POINTS = [map_year_to_feature_name(year) for year in RANGE_OF_YEARS]
# Categorical columns
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

# Disease flags and also broken down by algorithms used to decide
# DISEASE_FLAGS = {
#     "ckd": [
#         "norace",
#         "egfr",
#         "egfr_norace",
#         "dx",
#         "albpro",
#         "anylab",
#         "anylab_norace",
#     ],
#     "dm": ["a1c", "fbg", "rbg", "dxip", "dxop", "medication"],
#     "pdm": ["a1c", "fbg", "rbg", "dx"],
#     "htn": ["vit", "dx"],
# }
CKD_DISEASE_FLAGS = [
    f"egfr_entry_period_{ckd_flag}_flag"
    for ckd_flag in ["egfrckd", "egfrckd_norace", "dxckd", "albprockd"]
]
DISEASE_FLAGS = [
    f"egfr_entry_{disease_flag}_flag" for disease_flag in ["dm", "pdm", "htn"]
] + CKD_DISEASE_FLAGS
BINARY_COLS = [
    "site_source",
    "sex_code",
    "vital_status_code",
] + DISEASE_FLAGS

MULTICAT_COLS = [
    # "ruca_code",
    "ruca_4_class",
    # "ruca_7_class",
    # "patient_race",
    # "race_ethnicity_cat",
    "patient_state",
    "patient_country",
] + RACE_COLS

CKD_CAT_COLS = BINARY_COLS + MULTICAT_COLS

# Continuous columns
LAB_TESTS = {
    "count": ["scr", "hba1c", "uacr", "upcr", "bp", "av", "ipv"],
    "mean": ["", "norace", "hba1c", "uacr", "upcr", "sbp", "dbp", "pp", "map"],
    "coverage": ["aceiarb", "sglt2", "glp1", "nsaid", "ppi"],
}

CKD_FLATTENED_LONGITUDINAL_COLS = [
    f"egfr_{time}_{lab}_{stat}" if lab != "" else f"egfr_{time}_{stat}"
    for time in CKD_TIME_POINTS
    for stat, labs in LAB_TESTS.items()
    for lab in labs
]

# capture groups: time (e.g. year1), lab (e.g., sbp), stat (e.g., count)
LONGITUDINAL_COL_REGEX = (
    r"egfr_?(?P<Time>year\d+|entry_period+)?_?(?P<Lab>[\w\d]+)?_(?P<Stat>\w+)"
)

CKD_STATIC_CTN_COLS = [
    "egfr_entry_age",
    "egfr_years_followed",
    # "egfr_entry_year",
    # "egfr_last_folow_year",
]

CKD_ONEHOT_PREFIXES = [
    "ethnicity",  # "ruca_code",
    "ruca_4_class",
    # "ruca_7_class",
    # "patient_race",
    # "race_ethnicity_cat",
    "patient_state",
    "patient_country",
]
CKD_DEFAULT_TARGET = "rapid_decline"


# TODO[LOW]: what to do with reduction cols?
class CureCKDDataLoader(AbstractDatasetLoader):
    def __init__(
        self,
        cure_ckd_data_path: str,
        cure_ckd_subgroup_filter: Optional[Dict[str, str]] = None,
        cure_ckd_preproc_file_name: str = CKD_PREPROC_FILE_NAME,
        cure_ckd_flattened_df_file_name: str = CKD_FLATTENED_DF_FILE_NAME,
        cure_ckd_longitudinal_file_name: str = CKD_LONGITUDINAL_FILE_NAME,
        # time_points: List[str] = CKD_TIME_POINTS,
        cure_ckd_categorical_cols: List[str] = CKD_CAT_COLS,
        cure_ckd_static_continuous_cols: List[str] = CKD_STATIC_CTN_COLS,
        cure_ckd_longitudinal_cols: List[str] = CKD_FLATTENED_LONGITUDINAL_COLS,
        cure_ckd_onehot_prefixes: List[str] = CKD_ONEHOT_PREFIXES,
        cure_ckd_target: str = CKD_DEFAULT_TARGET,
        cure_ckd_time_window: Optional[Tuple[int, int]] = (0, 1),  # [a,b] (inclusive)
        cure_ckd_missingness_threshold: float = 0.8,
    ) -> None:
        self.data_path = cure_ckd_data_path
        # keys: static, longitudinal
        self.subgroup_filter = cure_ckd_subgroup_filter
        self.preproc_file_name = cure_ckd_preproc_file_name
        # preprocessing on top of what you get from using the preprocess repo
        self.flattened_df_file_name = cure_ckd_flattened_df_file_name
        self.longitudinal_file_name = cure_ckd_longitudinal_file_name
        self.categorical_cols = cure_ckd_categorical_cols
        self.static_continuous_cols = cure_ckd_static_continuous_cols
        self.flattened_longitudinal_cols = cure_ckd_longitudinal_cols
        self.onehot_prefixes = cure_ckd_onehot_prefixes
        self.target = cure_ckd_target
        self.time_window = cure_ckd_time_window
        self.missingness_threshold = cure_ckd_missingness_threshold

        self.continuous_cols = (
            cure_ckd_static_continuous_cols + cure_ckd_longitudinal_cols
        )

    @staticmethod
    def rapid_decline(df: pd.DataFrame) -> pd.Series:
        """Generate a bool series indicating if a row shows rapid decline."""
        # 40% decline from time_zero to year 2
        return df["egfr_year2_mean"] <= (0.6 * df["egfr_entry_period_mean"])

    def load_features_and_labels(
        self, data_type_time_dim: Optional[DataTypeTimeDim] = DataTypeTimeDim.STATIC
    ) -> Union[LongitudinalFeatureAndLabel, StaticFeatureAndLabel]:
        """
        Returns static or longitudinal forms of the dataset.
        When longitudinal not none will return the longitudinal portion vs static portion.
        """
        df, labels = self.load_flattened_df()
        df = self.filter_adhoc(df)
        labels = labels[df.index]  # filter down labels too
        # Fix column name for egfr since it's egfr_time_mean, instead of egfr_time_egfr_mean to match the pattern of other labs
        def egfr_mean_regex(col: str) -> str:
            return re.sub(
                r"egfr_?(?P<Time>year\d+|entry_period+)?_mean",
                r"egfr_\g<Time>_egfr_mean",
                col,
            )

        df.rename(
            columns=egfr_mean_regex,
            inplace=True,
        )
        # fix the same thing in continuous cols and flattened longitudinal cols
        self.continuous_cols = [egfr_mean_regex(col) for col in self.continuous_cols]
        self.flattened_longitudinal_cols = [
            egfr_mean_regex(col) for col in self.flattened_longitudinal_cols
        ]
        if data_type_time_dim.is_longitudinal():
            longitudinal_file_path = join(self.data_path, self.longitudinal_file_name)
            try:
                longitudinal_df = pd.read_parquet(longitudinal_file_path)
            except Exception:
                rank_zero_info("Longitudinal file not found, creating...")
                longitudinal_df = self.load_longitudinal(df)
                # Need parquet to serialize multindex dfs
                longitudinal_df.to_parquet(longitudinal_file_path)

            labels.index = longitudinal_df.index.get_level_values("patient_id").unique()

            return (longitudinal_df, labels)
        elif data_type_time_dim == DataTypeTimeDim.STATIC_SUBSET:
            static_df = df[df.columns.difference(self.flattened_longitudinal_cols)]
            static_df.set_index("patient_id", inplace=True)
            return (static_df, labels)

        # otherwise return df and labels
        # make sure to keep labels for whatever was filtered down for the df
        return (df.drop("patient_id", axis=1), labels)

    def load_flattened_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns X and y for use with models with everything static from the registry."""
        flattened_df_path = join(self.data_path, self.flattened_df_file_name)
        cat_pickle_fname = join(self.data_path, "cat_cols.pkl")
        ctn_pickle_fname = join(self.data_path, "ctn_cols.pkl")
        try:
            df = pd.read_feather(flattened_df_path)
            with open(cat_pickle_fname, "rb") as cat_col_f:
                self.categorical_cols = pickle.load(cat_col_f)
            with open(ctn_pickle_fname, "rb") as ctn_col_f:
                self.continuous_cols = pickle.load(ctn_col_f)
        except IOError:
            rank_zero_info("Pickled features and labels do not exist. Creating...")
            df = self.load_preprocessed_data()
            # dump df
            df.to_feather(flattened_df_path)
            # dump colnames
            with open(cat_pickle_fname, "wb") as cat_col_f:
                pickle.dump(self.categorical_cols, cat_col_f)
            with open(ctn_pickle_fname, "wb") as ctn_col_f:
                pickle.dump(self.continuous_cols, ctn_col_f)

        features = self.categorical_cols + self.continuous_cols + ["patient_id"]
        return (df[features], df[self.target])

    def melt_lab(
        self, data_df: pd.DataFrame, lab_name: str, lab_values: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Takes the data df, and melts each lab using the info for that lab.
        Data is currently flat: [aceiarb_count_entry, aceiarb_count_year1]
        We want it long (melted) for LSTM format (batch size, seq length, num features).
        e.g. for aceiarb count we can get the value for each patient for each time point (entry, year1, etc.)
        """
        old_cols = lab_values["Column Names"].to_list()
        new_cols = lab_values["Time"].to_list()
        tmp = data_df[old_cols]
        tmp.columns = new_cols
        return tmp.melt(
            value_vars=lab_values["Time"],
            var_name="Time",
            value_name="_".join(lab_name),
            ignore_index=False,
        )

    def load_longitudinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loads longitudinal form of the dataset.
        Main challenges being that the registry is flat:
                        aceiarb_year1, aceiarb_year2, ...
            patient1        ...             ...
            patient2        ...             ...
        But we want it to be long/melted:
                                aceiarb
            patient1    year 1    ...
                        year 2    ...
                        ...
            patient2    year 1    ...
                        year 2    ...
                        ...
        This function takes each lab, what kind of stat (count, mean, etc), and time point and melts the flattened registry into longitudinal form.
        """

        extracted_df = (
            pd.Index(self.flattened_longitudinal_cols)  # enforce index dtype
            .str.extract(LONGITUDINAL_COL_REGEX)  # extract the regex groups
            .set_axis(self.flattened_longitudinal_cols)  # each row = col/var name
            .dropna(axis=0, how="all")  # drop any that fail to grab all 3 groups
            .dropna(subset=["Time"])  # drop any that don't have a time point
            # .fillna("egfr")  # egfr doesn't have an extra lab name, fillna with egfr
            .reset_index()  # want a column with the var names instead of index
            .rename(
                columns={"index": "Column Names"}
            )  # rename column that used to be index
        )
        # original cols in df that are longitudinal (used to keep static separate)
        # longitudinal_cols = extracted_df["Column Names"]

        # index by patient (pt)
        pt_df = df.set_index("patient_id")
        melted_values = [
            self.melt_lab(pt_df, lab_name, lab_values)
            for lab_name, lab_values in extracted_df.groupby(["Lab", "Stat"])
        ]

        merge_on = ["patient_id", "Time"]
        # merge all the longitudinal features
        longitudinal_df = (
            reduce(
                lambda df1, df2: pd.merge(df1, df2, on=merge_on, how="outer"),
                melted_values,
            )
            .set_index("Time", append=True)  # indexed by [pt, time]
            .sort_index(
                level="patient_id"
            )  # sort by pt to get desired [pt, seq, feature] dimensions
        )
        return longitudinal_df

    def load_preprocessed_data(self) -> pd.DataFrame:
        """Loads feather/preprocessed data file from the preprocess repo into pandas dataframe + extra preprocessing.
        The extra preprocessing is expensive and we don't want to do it on the fly.
        The result will be serialized ontop of the serialized file from the preprocess repo.
        """
        try:
            df = pd.read_feather(join(self.data_path, self.preproc_file_name))
        except IOError:
            error(
                "Preprocessed file does not exist! Please use the `cure_ckd_preprocess` repo to create.",
            )

        df = self.create_ckd_flag(df)
        df = self.onehot_encode(df, MULTICAT_COLS)
        df = self.filter_data(df)
        # Create default label
        df["rapid_decline"] = self.rapid_decline(df)
        return df

    def create_ckd_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        ## COMBINE CKD FLAGS INTO ONE
        df["egfr_entry_ckd_flag"] = (df[CKD_DISEASE_FLAGS] == 1).any(axis=1)
        # remove the more granular flags
        for ckd_col in CKD_DISEASE_FLAGS:
            self.categorical_cols.remove(ckd_col)
        self.categorical_cols.append("egfr_entry_ckd_flag")
        return df

    def onehot_encode(
        self, df: pd.DataFrame, cat_cols_to_encode: List[str]
    ) -> pd.DataFrame:
        """Creates dummies for passed in columns. Drops them from the global var keeping track of what categorical columns are available."""
        for multicat_col in cat_cols_to_encode:
            self.categorical_cols.remove(multicat_col)
        return pd.concat(
            [
                df.drop(cat_cols_to_encode, axis=1),
                pd.get_dummies(df[cat_cols_to_encode]),
            ],
            axis=1,
        )

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """This is going to be serialized vs filter_subgroup which could chagne often and will happen ad-hoc."""
        # filter rows first
        age_mask = df["egfr_entry_age"] < 100
        rank_zero_info("Dropping patients over 100 years old.")
        df = df[age_mask].reset_index(drop=True)

        return df

    #### AD-HOC FILTERING ####
    def filter_adhoc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modifies self.continuous_columns depending on what you want to filter.
        """
        df = self.filter_subgroup(df, ENCODINGS)
        df = self.filter_time_window(df)
        df = self.filter_highly_missing_columns(df)
        return df

    def filter_highly_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter columns
        acceptable_missing_cols = df.isna().mean() < self.missingness_threshold
        rank_zero_info(
            f"Dropping the following columns for missing more than {self.missingness_threshold*99}% data:\n{df.columns[~acceptable_missing_cols]}"
        )
        for col in df.columns[~acceptable_missing_cols]:
            if col in self.continuous_cols:
                self.continuous_cols.remove(col)
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)
            if col in self.flattened_longitudinal_cols:
                self.flattened_longitudinal_cols.remove(col)

        return df[df.columns[acceptable_missing_cols]]

    def filter_time_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep columns within year range provided, dropping those outside of that."""

        def time_window_filter(time_window_range: List[str]) -> str:
            """Returns regex for columns to exclude outside of time window provided."""
            # symmetric difference of the ranges
            time_points_to_exclude = set(time_window_range) ^ set(CKD_TIME_POINTS)
            """Generated a regex for the names of the time windows we want."""
            regex = r".*("
            # OR all of the time periods
            regex += r"|".join(time_points_to_exclude)
            regex += r").*"
            return regex

        time_window_range = [
            map_year_to_feature_name(range_point)
            for range_point in range(self.time_window[0], self.time_window[1] + 1)
        ]
        rank_zero_info(
            f"Filtering features between {time_window_range[0]} and {time_window_range[-1]}."
        )

        cols_to_exclude = df.filter(
            regex=time_window_filter(time_window_range), axis=1
        ).columns

        # drop cols to exclude that aren't in our window
        for col_to_exclude in cols_to_exclude:
            try:
                self.continuous_cols.remove(col_to_exclude)
                self.flattened_longitudinal_cols.remove(col_to_exclude)
            except ValueError:  # key doesn't exist, do nothing
                pass
        return df.drop(cols_to_exclude, axis=1)

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        p = ArgumentParser(parents=[parent_parser], add_help=False)

        p.add_argument(
            "--cure-ckd-data-path",
            type=str,
            required=True,
            help="Path name to all the data required for the CURE CKD registry + serialized files.",
        )
        p.add_argument(
            "--cure-ckd-subgroup-filter",
            # type=Dict[str, str],
            action=YAMLStringDictToDict(choices=ENCODINGS.keys()),
            help="Dictionary of variable names to which group within that variable you want to filter down to.",
        )
        p.add_argument(
            "--cure-ckd-preproc-file-name",
            type=str,
            default=CKD_PREPROC_FILE_NAME,
            help="Name of the file produced by the cure_ckd_preprocess repo.",
        )
        p.add_argument(
            "--cure-ckd-flattened-df-file-name",
            type=str,
            default=CKD_FLATTENED_DF_FILE_NAME,
            help="Name of file that may be additionally preprocessed ontop of cure_ckd_preprocess output. This df is a flattened representation of data in the registry.",
        )
        p.add_argument(
            "--cure-ckd-longitudinal-file-name",
            type=str,
            default=CKD_LONGITUDINAL_FILE_NAME,
            help="Name of file that processes the flat df into longitudinal columns and longitudinal form (batch size, sequence length, features).",
        )
        # time_points: List[str] = CKD_TIME_POINTS,
        p.add_argument(
            "--cure-ckd-categorical-cols",
            type=List[str],
            default=CKD_CAT_COLS,
            help="List of categorical columns in the dataset.",
        )
        p.add_argument(
            "--cure-ckd-static-continuous-cols",
            type=List[str],
            default=CKD_STATIC_CTN_COLS,
            help="List of continuous columns in the dataset.",
        )
        p.add_argument(
            "--cure-ckd-longitudinal-cols",
            type=List[str],
            default=CKD_FLATTENED_LONGITUDINAL_COLS,
            help="List of longitudinal columns in the dataset.",
        )
        p.add_argument(
            "--cure-ckd-onehot-prefixes",
            type=List[str],
            default=CKD_ONEHOT_PREFIXES,
            help="Prefixes for multi-categorical columns that were one-hot encoded.",
        )
        p.add_argument(
            "--cure-ckd-target",
            type=str,
            default=CKD_DEFAULT_TARGET,
            help="Name of target variable for training.",
        )
        p.add_argument(
            "--cure-ckd-time-window",
            action=YAMLStringListToList(convert=int),
            default=(0, 1),
            help="Time window of registry data, [a,b] (inclusive) If you want to include years 4-6 for example, then (4,6).",
        )
        p.add_argument(
            "--cure-ckd-missingness-threshold",
            type=float,
            default=0.8,
            help="Allowable percentage of missingness in columns (as a decimal), otherwise they are dropped.",
        )

        return p


# Testing
if __name__ == "__main__":
    from main import init_cli_args
    from utils.cli_arg_utils import load_cli_args

    load_cli_args()
    args = init_cli_args()
    data_loader = CureCKDDataLoader.from_argparse_args(args)
    X, y = data_loader.load_features_and_labels(args.longitudinal)
