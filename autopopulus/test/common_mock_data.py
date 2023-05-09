"""
We are concerned with data that is:
Feature domain: {continuous, categorical binary, categorical one-hot multiclass}
x
Time domain: {static, longitudinal (variable length), longitudinal (fixed length)}
x
Transformed: {normal, discretized}
"""
from numpy import nan
from pandas import DataFrame, Series
from hypothesis import strategies as st
from hypothesis.extra.pandas import column

seed = 0

# Hypothesis setup
hypothesis = {
    "columns": [
        column("bin", elements=st.one_of(st.just(nan), st.integers(0, 1))),
        column(
            "mult1",
            elements=st.one_of(st.just(nan), st.sampled_from([0, 1, 2, 3])),
        ),
        column("ctn1", dtype=float),
        column("mult2", elements=st.one_of(st.just(nan), st.sampled_from([0, 1, 2]))),
        column("ctn2", dtype=float),
    ],
    "ctn_cols": ["ctn1", "ctn2"],
    "ctn_cols_idx": [2, 4],
    "cat_cols": ["bin", "mult1", "mult2"],
    "cat_cols_idx": [0, 1, 3],
    "onehot_prefixes": ["mult1", "mult2"],  # 4 and 3 categories respectively
    "onehot": {
        "ctn_cols": ["ctn1", "ctn2"],
        "cat_cols": [
            "bin",
            "mult1_0.0",
            "mult1_1.0",
            "mult1_2.0",
            "mult1_3.0",
            "mult2_0.0",
            "mult2_1.0",
            "mult2_2.0",
        ],
        "bin_cols": ["bin"],
        "onehot_cols": [
            "mult1_0.0",
            "mult1_1.0",
            "mult1_2.0",
            "mult1_3.0",
            "mult2_0.0",
            "mult2_1.0",
            "mult2_2.0",
        ],
        "onehot_expanded_prefixes": [
            "mult1",
            "mult1",
            "mult1",
            "mult1",
            "mult2",
            "mult2",
            "mult2",
        ],
        "ctn_cols_idx": [5, 9],
        "cat_cols_idx": [0, 1, 2, 3, 4, 6, 7, 8],
        "bin_cols_idx": [0],
        "onehot_cols_idx": [
            [1, 2, 3, 4],
            [6, 7, 8],
        ],
    },
}


#############
#  By Hand  #
#############
# TODO[LOW]: This can be cleaned up and replaced with hypothesis tests for mostly everything maybe minus metric calculation for which i definitely do not need all this.
columns = {  # Contains continuous, binary, and onehot
    "columns": ["age", "weight", "ismale", "fries_s", "fries_m", "fries_l"],
    "ctn_cols": ["age", "weight"],
    "onehot_prefix_names": ["fries"],
    "no_onehot": ["age", "weight", "ismale", "fries"],
}

col_idxs_by_type = {
    "original": {
        "continuous": [0, 1],
        "categorical": [2, 3, 4, 5],
        "binary": [2],
        "onehot": [[3, 4, 5]],
    }
}

groupby = {
    "original": {
        "categorical_onehots": {3: "fries", 4: "fries", 5: "fries"},
        "binary_vars": {2: "ismale"},
    }
}


discretization = {
    "cuts": [
        [(0, 20), (20, 40), (40, 80)],
        [(0, 15.5), (15.5, 31.0), (31.0, 50.5), (50.5, 80.4)],
    ],
}
discretization["discretizer_dict"] = {
    "age": {"bins": discretization["cuts"][0], "indices": [4, 5, 6]},
    "weight": {"bins": discretization["cuts"][1], "indices": [7, 8, 9, 10]},
}
columns["onehot_continuous"] = [
    f"{col}_{bin_range[0]} - {bin_range[1]}"
    for col, info in discretization["discretizer_dict"].items()
    for bin_range in info["bins"]
]
columns["discretized_columns"] = ["ismale", "fries_s", "fries_m", "fries_l"] + columns[
    "onehot_continuous"
]

X = {
    # Data missing in ctoninuous, and categorical
    "X": DataFrame(
        [
            [44, nan, 0, 0, 1, 0],
            [49, 57.2, 1, 0, 0, 1],
            [26, 26.3, 0, nan, nan, nan],
            [16, 73.4, 1, 1, 0, 0],
            [nan, 56.5, 1, 0, 1, 0],
            [57, 29.6, 0, 1, 0, 0],
        ],
        columns=columns["columns"],
    ),
    "nomissing": DataFrame(
        [
            [44, 15.1, 0, 0, 1, 0],
            [49, 57.2, 1, 0, 0, 1],
            [26, 26.3, 0, 0, 1, 0],
            [16, 73.4, 1, 1, 0, 0],
            [13, 56.5, 1, 0, 1, 0],
            [57, 29.6, 0, 1, 0, 0],
        ],
        columns=columns["columns"],
    ),
    "scale": DataFrame(
        [
            [1, nan, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [-2.6, -30.9, 0, nan, nan, nan],
            [-4.6, 16.2, 1, 1, 0, 0],
            [nan, -0.7, 1, 0, 1, 0],
            [3.6, -27.6, 0, 1, 0, 0],
        ],
        columns=columns["columns"],
    ),
    "disc": DataFrame(
        [
            [0, 0, 1, 0, 0, 0, 1, nan, nan, nan, nan],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, nan, nan, nan, 0, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, nan, nan, nan, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        ],
        columns=columns["discretized_columns"],
    ),
    "disc_true": DataFrame(
        [
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        ],
        columns=columns["discretized_columns"],
    ),
    "uniform": DataFrame(
        [
            [0, 0, 1, 0, 0, 0, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 1 / 3, 1 / 3, 1 / 3, 0, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        ],
        columns=columns["discretized_columns"],
    ),
    "no_onehot": DataFrame(
        [
            [44, 15.1, 0, "m"],
            [49, 57.2, 1, "l"],
            [26, 26.3, 0, "m"],
            [16, 73.4, 1, "s"],
            [13, 56.5, 1, "m"],
            [57, 29.6, 0, "s"],
        ],
        columns=columns["no_onehot"],
    ),
    "target_encoded": DataFrame(
        [
            [44, nan, 0.01, 0.6],
            [49, 57.2, 0.1, 0.7],
            [26, 26.3, 0.01, nan],
            [16, 73.4, 0.1, 0.5],
            [nan, 56.5, 0.1, 0.6],
            [57, 29.6, 0.01, 0.5],
        ],
        columns=columns["no_onehot"],
    ),
    "target_encoded_true": DataFrame(
        [
            [44, 15.1, 0.01, 0.6],
            [49, 57.2, 0.1, 0.7],
            [26, 26.3, 0.01, 0.6],
            [16, 73.4, 0.1, 0.5],
            [13, 56.5, 0.1, 0.6],
            [57, 29.6, 0.01, 0.5],
        ],
        columns=columns["no_onehot"],
    ),
}

y = Series([1, 0, 1, 0, 1, 0], name="outcome")

splits = {
    "train": [0, 1],
    "train_FO": [1],  # fully observed
    "val": [2, 3],
    "val_FO": [3],
    "test": [4, 5],
    "test_FO": [5],
}
