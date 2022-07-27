import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Union, Dict, Any, Optional
import warnings

"""
Module includes the following main functionality:
- writing preprocessed df to file
- loading features + targets from file
- processing raw data
- generating labels
- one-hot encoding features
"""

#######################
#     Global Vars     #
#######################
DATA_PATH = "/home/davina/Private/ckd-data/"

# Categorical columns
CAT_COLS = [
    "site_source_cat",
    "cohort_cat",
    # "misclassified_egfr",
    # "misclassified_albpro",
    "study_entry_DM_flag",
    "study_entry_PDM_flag",
    "study_entry_HTN_flag",
    "demo_sex",
    # "demo_race_ethnicity_cat",
    "demo_rural_cat",
]
TIME_ZERO_COLS = [
    # Note: hba1c is the same as the _a1c for the following years
    "time_zero_egfr_mean",
    "time_zero_hba1c_mean",
    "time_zero_sbp_mean",
    "time_zero_av_count",
    "time_zero_ipv_count",
    "time_zero_aceiarb_coverage",
]
# Continuous columns at entry of study
CTN_ENTRY_COLS = [
    "study_entry_age",
    "study_entry_egfr",
    "study_entry_a1c",
    "study_entry_sbp",
]
# Continuous columns
CTN_COLS = (
    CTN_ENTRY_COLS
    + TIME_ZERO_COLS
    + ["year" + str(i) + "_egfr_mean" for i in range(1, 11)]
    + ["year" + str(i) + "_a1c_mean" for i in range(1, 11)]
    + ["year" + str(i) + "_sbp_mean" for i in range(1, 11)]
    + ["year" + str(i) + "_av_count" for i in range(1, 11)]
    + ["year" + str(i) + "_ipv_count" for i in range(1, 11)]
    + ["year" + str(i) + "_aceiarb_coverage" for i in range(1, 11)]
)
RACE_COLS = [
    "Not Otherwise Categorized",
    "White Non-Latino",
    "White Latino",
    "Black",
    "Asian",
    "American Indian/Alaskan Native",
    "Native Hawaiian or Other Pacific Islander",
]
RACE_COLS = list(map(lambda x: f"ethnicity_{x}", RACE_COLS))

ONEHOT_PREFIXES = ["ethnicity"]

# Used by load_features_and_labels
DEFAULT_TARGET = "decl_base2any_>=40_bin"

#######################
#     Main Driver     #
#######################
def preprocess_data(data_path: str) -> None:
    """The whole preprocessing pipeline.

    Writes df to file at the end.
    """
    df = process_raw_data(data_path)
    df = construct_targets(df)
    df = one_hot_enc(df)
    #### Write to file ####
    # Feather pickles pandas DFs so that they load much faster
    # Might take more storage space though
    # df.to_feather(data_path + "preproc_data.feather")
    # changed because of annoying pyarrow issues, using fastparquet
    df.to_parquet(data_path + "preproc_data.parquet", compression="gzip")


def load_features_and_labels(
    data_path: str,
    target: str = DEFAULT_TARGET,
    subgroup: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns X and y for use with models.
    - It can optionally take a target as a string.
    - It can optionally take a subgroup of the whole dataset to filter for.
        - For example, if you want to filter for CKD and UCLA.
        - Refer to documentation for filter_subgroup.
    """
    df = load_preprocessed_data(data_path, subgroup)

    # Could also be: ... + CTN_ENTRY_COLS + TIME_ZERO_COLS + RACE_COLS
    # or ... + CTN_ENTRY_COLS + TIME_ZERO_COLS
    features = (
        [col for col in CAT_COLS if "race" not in col]
        + CTN_ENTRY_COLS
        + TIME_ZERO_COLS
        + RACE_COLS
    )
    return (df[features], df[target])


def load_preprocessed_data(
    data_path: str, subgroup: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
) -> pd.DataFrame:
    """Loads feather/preprocessed data file into pandas dataframe.

    If it cannot find a file with the already preprocessed data,
    it will run the preprocess_data pipeline (create the file).
    """
    try:
        # raise IOError
        # df = pd.read_feather(data_path + "preproc_data.feather")
        df = pd.read_parquet(data_path + "preproc_data.parquet")
    except IOError:
        print("Preprocessed file does not exist! Creating...")
        preprocess_data(data_path)
        # df = pd.read_feather(data_path + "preproc_data.feather")
        df = pd.read_parquet(data_path + "preproc_data.parquet")

    ## FILTERING
    # Filter for subgroup if requested
    if subgroup:
        df = filter_subgroup(df, subgroup)
    # Remove misclassified patients
    misclassified_cols = [col for col in df.columns if "miscl" in col]
    not_misclassified_mask = (df[misclassified_cols] == 1).all(
        axis=1
    )  # Logical AND of all the columns(correcty classified = 1)
    # Keep patients under 100 years old
    age_mask = df["study_entry_age"] < 100
    # Filter age & misclasisfied
    combined_masks = not_misclassified_mask & age_mask
    df = df[combined_masks].reset_index(drop=True)

    return df


def get_subgroup(cohort: str, site: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generates subgroups to filter on the ckd dataset.
    Each stratification is represented as a dictionary:
    it will have a name, and an encoding (unless you want 'all')
    """
    encodings = {"ckd": 0, "ucla": 0, "atrisk": 1, "providence": 1}
    cohort = (
        {"name": cohort, "encoding": encodings[cohort]}
        if cohort in encodings
        else {"name": cohort}
    )
    site = (
        {"name": site, "encoding": encodings[site]}
        if site in encodings
        else {"name": site}
    )
    return (cohort, site)


def filter_subgroup(
    df: pd.DataFrame, subgroup: Tuple[Dict[str, Any], Dict[str, Any]]
) -> pd.DataFrame:
    """Filters the features/label for the given subgroup.

    Args:
        - subgroup: tuple of some sort of combo of cohort and site source
            - Each stratification is represented as a dictionary
                - it will have a name, and an encoding (unless you want 'all')
                - excluding encoding means don't filter on this column
    """
    cohort, site_source = subgroup
    if "encoding" in cohort:
        cohort_filter = df["cohort_cat"] == cohort["encoding"]
        # drop cohort var if filtering down to a cohort
        df.drop("cohort_cat", axis=1, errors="ignore", inplace=True)
        CAT_COLS.remove("cohort_cat")
    else:
        # "True" will not filter on that column (if it doesn't have encoding)
        cohort_filter = (
            df["cohort_cat"] | True
        )  # Ensures a series of same length of all true

    if "encoding" in site_source:
        site_source_filter = df["site_source_cat"] == site_source["encoding"]
        # drop site source var if filtering down to one
        df.drop("site_source_cat", axis=1, errors="ignore", inplace=True)
        CAT_COLS.remove("site_source_cat")
    else:
        site_source_filter = df["site_source_cat"] | True
    # Combine filters
    subgroup_filter = cohort_filter & site_source_filter
    # if filters are both entirely true, there's nothing to filter
    return df[subgroup_filter]


#######################
#   Data Processing   #
#######################
def process_raw_data(data_path: str) -> pd.DataFrame:
    """Processes raw csv data.

    - Calculate percentage change in egfr (2 year chunks over the 10 years)
    - Discretize/bin percentage change for labels
    - Sanitize numerical values that should be NaN
    - Trims down the columns returned to categorical, continuous columns,
        and percentage of change in egfr (discretized)
    """
    # Load raw data/CSV
    df = pd.read_csv(data_path + "Daratha eGFR Trajectory Data Version31.txt")

    # Grab columns with egfr data
    eGFR_cols = [i for i in df.columns if "gfr" in i and "year" in i]

    #### Calculate percentage change in egfr ###
    perc_delta_eGFR = []
    for ind in range(len(eGFR_cols) - 2):
        # create new column to calculate difference
        col_name = "diff_eGFR" + str(ind)
        perc_delta_eGFR.append(col_name)
        # formula: decrease = (egfr[i+2] - egfr[i]) / egfr[i]
        # Note: decrease is calculated over the range of 2  years
        df[col_name] = (df[eGFR_cols[ind + 2]] - df[eGFR_cols[ind]]) / df[
            eGFR_cols[ind]
        ]

    #### Discretize the change in egfr ####
    perc_delta_eGFR_discr = []

    # create bins/labels: [<-100%, -100, -90, ... 90%, 100%, > 100%]
    bins = np.arange(-1, 1.1, 0.1)
    labels = (
        ["<-100%"]
        + [str(int(np.ceil(i * 100))) + "%" for i in bins if int(np.ceil(i * 100)) != 0]
        + [">100%"]
    )
    # update bins for outer range (<-100%, >100%)
    bins = [-np.inf] + list(bins) + [np.inf]

    # Add discretized change into the df
    for col in perc_delta_eGFR:
        col_name = col + "_discr"
        perc_delta_eGFR_discr.append(col_name)
        df[col_name] = pd.cut(df[col], bins=bins, labels=labels)

    #### Fix Missing Value Representation ####
    # Some missing values are represented as numerical values
    df["study_entry_a1c"].replace(to_replace=-99.99, value=np.nan, inplace=True)
    df["study_entry_sbp"].replace(to_replace=-999.99, value=np.nan, inplace=True)
    df["time_zero_hba1c_mean"].replace(to_replace=-99.99, value=np.nan, inplace=True)
    df["demo_sex"].replace(to_replace=-9, value=np.nan, inplace=True)
    df["demo_rural_cat"].replace(to_replace=-9, value=np.nan, inplace=True)

    # Make the cat columns 0/1 instead of 1/2
    # Keep the current ordering, just shift to 0/1
    cat_shifter = {1: 0, 2: 1}
    cat_columns_to_shift = [
        "cohort_cat",
        "demo_sex",
        "site_source_cat",
        "misclassified_egfr",
        "misclassified_albpro",
    ]

    df[cat_columns_to_shift] = df[cat_columns_to_shift].replace(cat_shifter)

    # keep_cols = CAT_COLS + CTN_COLS + perc_delta_eGFR_discr
    return df


#########################
#  Target Construction  #
#########################
def construct_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Creates possible targets to use in a model.

    - %decline (regression)
    - 40% decline (binary)
    - multiclass (10 groups)
    - multiclass (20 groups)
    - custom rapid decline from baseline (binary)
    - custom rapid decline 2 year windows (binary)
    """
    df["year_of_>=40_decl_base2any"] = get_year_of_first_decline_from_baseline(
        df, perc=0.4
    )
    df["decl_base2any_>=40_bin"] = binarize_year_of_decl_base2any(
        df["year_of_>=40_decl_base2any"]
    )
    df["ten_groups"] = get_ten_groups(df)
    df["twenty_groups"] = get_twenty_groups(df)
    df = pd.concat([df, rapid_decline_from_baseline(df)], axis=1)
    df = pd.concat([df, window_2yr_rapid_decline(df)], axis=1)
    # if you call both from_baseline and 2 year window theres 1 that overlaps
    # remove duplicates
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def rapid_decline_from_baseline(
    df: pd.DataFrame,
    perc: float = 0.4,  # percent decline
    egfr_delta_threshold: int = 15,  # num egfr points is dangerous to drop
) -> pd.DataFrame:
    """Binary label of rapid decline from baseline to each year."""
    labels = pd.DataFrame()
    for i in range(1, 11):
        labels[f"rapid_decline_base_to_{i}"] = get_rapid_decline(df, "base", i, perc)
    #             get_rapid_decline(df, 'base', i, perc, egfr_delta_threshold))
    return labels


def window_2yr_rapid_decline(
    df: pd.DataFrame,
    perc: float = 0.4,  # percent decline
    egfr_delta_threshold: int = 15,  # num egfr points is dangerous to drop
) -> pd.DataFrame:
    """Binary labels of rapid decline (sliding 2 year window)."""
    labels = pd.DataFrame()
    start = "base"
    # end_egfr ranges from 0+2 (2) to 11-2 (9) because of two year window
    for end in range(2, 9):
        labels[f"rapid_decline_{start}_to_{end}"] = get_rapid_decline(
            df, start, end, perc
        )
        #                                                     perc, egfr_delta_threshold)
        start = 1 if start == "base" else start + 1
    return labels


def get_rapid_decline(
    df: pd.DataFrame,
    t_start: Union[str, int] = "base",  # Can either be 'base' or int of year
    t_end: Union[str, int] = "last",  # Can either be 'last' or int of year
    perc: float = 0.4,  # percent decline
    egfr_delta_threshold: Optional[
        int
    ] = None,  # how many egfr points is dangerous to drop
) -> np.ndarray:
    """Binary label of 'rapid decline' with our custom definition.

    For normal 40% over 2 years do not pass a threshold. Otherwise:
    We define rapid decline to account for the discrepancy that a
    40% drop at an egfr of 90 is a difference of 36, but 40% drop at
    an egfr of 20 is 8. It's "harder" for healthier patients to be caught
    by this so we adjust how we flag patients we suspect are experiencing
    rapid decline.
    """
    if t_start == "base":
        # Right now (with this dataset) no one is missing time_zero.
        # Technically baseline = time_zero OR the first existing egfr reading
        start = df["time_zero_egfr_mean"]
    else:  # t_start must be int within range (assume sane input)
        start = df[f"year{t_start}_egfr_mean"]

    if t_end == "last":
        eGFR_cols = [col for col in df.columns if "_egfr_" in col and "zero" not in col]
        end = df[eGFR_cols].ffill(axis=1).iloc[:, -1]
    else:  # t_end must be int within range (assume sane input)
        end = df[f"year{t_end}_egfr_mean"]

    # initialize all observations to be marked as no rapid decline
    rapid_decline = np.zeros_like(df.iloc[:, 0])

    # change in egfr between two time points
    egfr_delta = end - start
    # change as a percent of egfr at t_start
    percent_change = egfr_delta / start
    # binary label if percentage change is >= perc (usually 40%) decline
    # multiply percent change by -1 to indicate decline
    decline_occurred_by_x_percent = -percent_change >= perc
    # the ckd stage the patient lands in at t_end
    end_stage = get_egfr_stage(end)

    if egfr_delta_threshold:
        # filters to flag patients as having rapid decline
        # at higher egfr: drop of 15 points or dropping into stage 4/5
        flag_if_egfr_gt30 = (start > 30) & (
            (egfr_delta >= egfr_delta_threshold) | (end_stage == 4) | (end_stage == 5)
        )
        # at lower egfr: dropping into stage 5 or experiencing 40% decline
        flag_if_low_egfr = start.between(15, 30) & (
            (end_stage == 5) | decline_occurred_by_x_percent
        )
        # at very low egfr: experiencing 40% decline
        flag_if_failing = (start < 15) & decline_occurred_by_x_percent

        # mark all these cases as rapid decline
        rapid_decline[flag_if_egfr_gt30 | flag_if_low_egfr | flag_if_failing] = 1
        return rapid_decline
    else:
        return decline_occurred_by_x_percent


def get_percentage_change(
    df: pd.DataFrame,
    from_year: Union[str, int] = "base",  # Can either be 'base' or int of year
    to_year: Union[str, int] = "last",  # Can either be 'last' or int of year
) -> pd.Series:
    """Returns the percent decline for each patient.

    Helper method to generating different kinds of labels.
    """
    if from_year == "base":
        # Right now (with this dataset) no one is missing time_zero.
        # Technically baseline = time_zero OR the first existing egfr reading
        start = df["time_zero_egfr_mean"]
    else:  # from_year must be int within range (assume sane input)
        start = df[f"year{from_year}_egfr_mean"]

    if to_year == "last":
        eGFR_cols = [i for i in df.columns if "_egfr_" in i and "zero" not in i]
        end = df[eGFR_cols].ffill(axis=1).iloc[:, -1]
    else:  # to_year must be int within range (assume sane input)
        end = df[f"year{to_year}_egfr_mean"]

    return (end - start) / start


def binarize_year_of_decl_base2any(decline_base2any: pd.Series) -> pd.Series:
    """Binarizes the results of the decline from baseline to any year.

    Before 0 means no decline, otherwise it was the year we first
    see %perc decline. Now the years are just "on" flags.
    """
    bin_label = decline_base2any.copy()
    # Replace all non-zero values (year in which perc% decline first occured)
    bin_label[bin_label != 0] = 1
    return bin_label


def get_year_of_first_decline_from_baseline(
    df: pd.DataFrame, perc: float = 0.4
) -> pd.Series:
    """Returns categorical label according to Dennis' definition of perc% decline.

    Note that baseline is the time-zero or the first existing reading.

    Args:
        - perc: (-inf, 1] .4 corresponds to a >=40% drop in egfr.

    Returns:
        Label = in which year did we first see >=perc% decline? (categorical)
            - Implicitly asks:
                At any timepoint after baseline, is there a >=perc% decline?
            - 0 means that >=%perc decline never occured.
    """
    # get percent change from baseline to year_i for all 10 years
    # and then combine the labels as we go through each year
    per_year_labels = []
    for i in range(1, 11):
        # Invert to make declines positive values
        declines = -1 * get_percentage_change(df, "base", i)

        # answers: did >=perc% decline occur at year_i?
        # if >=perc% decline occurs again, we will keep the first
        label_base2year_i = declines.apply(lambda x: i if x >= perc else 0)
        per_year_labels.append(label_base2year_i)

    # Replace 0 values with nan as they indicate no decline occurred,
    # then backfill to get year of first decline in first column.
    # Access first column and swap nan back to 0.
    label = (
        pd.concat(per_year_labels, axis=1)
        .replace(0, np.nan)
        .bfill(axis=1)
        .iloc[:, 0]
        .fillna(0)
    )
    return label


#### DEPRECATED ####
# We don't want to look at these for official analysis anymore
# We can leave these in for personal experimentation.
def get_decline_from_baseline_to_last(
    df: pd.DataFrame,
    geq: bool = False,  # Greater than or equal to flag
    perc: float = 0.4,
) -> pd.Series:
    """Returns Label: if a decline of some percentage occured.

    Args:
    - perc: (percentage) has a .1/10% slack because it's actually a bin.
        For example: perc = .4 is actually [.3, .4]
    Return:
    - label: Series[int] (of either 0 or 1). 1 if the percentage change in egfr
        is really in that bin defined by perc), else 0.
    """
    warnings.warn(
        "This method is not used for official analysis anymore.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Invert to make declines positive values
    declines = -1 * get_percentage_change(df)

    # Remember the percentages are bins, so 40% is actually [30%, 40%]
    label = (
        declines.apply(lambda x: 1 if x >= perc - 0.1 and x <= perc else 0)
        if not geq
        else declines.apply(lambda x: 1 if x >= perc else 0)
    )
    return label


def get_ten_groups(df: pd.DataFrame) -> pd.Series:
    """Returns 10 groups of declines. (Really is 11)

    Groups: [[0, -10%], [-10%, -20%], ..., [-90%, -100%], [0, inf]]
    """
    # It's really 11 buckets, with one for no change or improvement
    declines = get_percentage_change(df)

    buckets = np.concatenate((np.linspace(-1, 0, 11), [np.inf]))
    return pd.cut(declines, buckets)


def get_twenty_groups(df: pd.DataFrame) -> pd.Series:
    """Returns 20 groupings of declines. (Really is 21)

    Similar to before, but we include a positive effect direction too now.
    Groups: [ [0, 10%], [10%, 20%], ..., [90%, 100%],
            [0, -10%], [-10%, -20%], ..., [-90%, -100%],
            [100%, inf]
            ]
    Note that we don't include [-100%, -inf] because it's not possible
    based on our definition of percent chage: (egfr2 - egfr1)/egfr1 =
    egfr2/egfr1 - 1. For this expression to be less than -1 (or -100%)
    egfr2/egfr has to be negative. As long as egfr is not negative,
    this is not possible.
    """
    # It's really 21 groups, with one for >100% change
    declines = get_percentage_change(df)

    buckets = np.concatenate((np.linspace(-1, 1, 21), [np.inf]))
    return pd.cut(declines, buckets)


def get_egfr_stage(egfr: pd.Series) -> pd.Series:
    """Helper method to bin egfr into stages.

    3.5 = 3b, 3 = 3a.
    """
    return pd.cut(
        egfr, [0, 15, 30, 45, 60, 90, 1000], right=False, labels=[5, 4, 3.5, 3, 2, 1]
    )


########################
#   One Hot Encoding   #
########################
def one_hot_enc(df: pd.DataFrame) -> pd.DataFrame:
    """One hot encodes some of the columns.

    Columns:
        - race
        - 10_groups
        - 20_groups
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(df["demo_race_ethnicity_cat"].values.reshape(-1, 1))

    #### RACE ####
    one_hot_races = enc.transform(
        df["demo_race_ethnicity_cat"].values.reshape(-1, 1)
    ).toarray()
    df[RACE_COLS] = pd.DataFrame(data=one_hot_races, columns=RACE_COLS)

    #### 10 Group ####
    # We add prefixes to differentiate between the 10s and 20s
    # There is overlap in column names otherwise (which creates problems)
    df = pd.concat([df, pd.get_dummies(df["ten_groups"], prefix="10")], axis="columns")
    df = df.drop("ten_groups", axis="columns")

    #### 20 Group ####
    df = pd.concat(
        [df, pd.get_dummies(df["twenty_groups"], prefix="20")], axis="columns"
    )
    df = df.drop("twenty_groups", axis="columns")

    # The dummies create column names that aren't strings
    # This creates problems for feather, so we will convert here
    df.columns = df.columns.astype(str)

    return df
