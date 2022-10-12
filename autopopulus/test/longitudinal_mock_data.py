from numpy import nan
from pandas import MultiIndex, DataFrame, Series

columns = {  # continuous, binary, and multiclass (onehot)
    # scr = serum creatinine, sob = short of breath, bp = blood pressure
    "columns": ["scr", "is_sob", "bp_low", "bp_high"],
    "ctn_cols": ["scr"],
    "onehot_prefix_names": ["bp"],
}
indices = {"ctn_cols": [0], "cat_cols": [1, 2, 3]}
discretization = {
    "cuts": [
        [(0, 0.7), (0.7, 0.94), (0.94, 2)],
    ],
}
discretization["discretizer_dict"] = {
    "scr": {"bins": discretization["cuts"][0], "indices": [3, 4, 5]},
}
columns["onehot_continuous"] = [
    f"{col}_{bin_range[0]} - {bin_range[1]}"
    for col, info in discretization["discretizer_dict"].items()
    for bin_range in info["bins"]
]
columns["discretized_columns"] = ["is_sob", "bp_low", "bp_high"] + columns[
    "onehot_continuous"
]

col_idxs_by_type = {"continuous": [0, 1], "categorical": [2, 3, 4, 5]}

groupby = {
    "categorical_only": {
        "discretize": {
            "categorical_onehots": {1: "bp", 2: "bp"},
            "binary_vars": {0: "is_sob"},
        },
        "no_discretize": {
            "categorical_onehots": {2: "bp", 3: "bp"},
            "binary_vars": {1: "is_sob"},
        },
    },
    "after_fit": {
        "discretize": {
            "categorical_onehots": {1: "bp", 2: "bp"},
            "binary_vars": {0: "is_sob"},
            "discretized_ctn_cols": {"data": {3: "age", 4: "age", 5: "age"}},
        },
        "no_discretize": {
            "categorical_onehots": {2: "bp", 3: "bp"},
            "binary_vars": {1: "is_sob"},
        },
    },
}
longitudinal_index = MultiIndex.from_arrays(
    [[0, 0, 0, 1, 1, 2], [0, 1, 2, 0, 1, 0]],
    names=("id", "time"),
)
X_longitudinal = {
    "X": DataFrame(
        [
            [nan, 1, 0, 1],  # ctn missing: 0
            [0.87, 1, 0, 1],  #: 0
            [1.03, 0, 1, 0],  #: 0
            [0.6, nan, 1, 0],  # binary missing: 1
            [0.65, 0, 1, 0],  #: 1
            [1.65, 1, nan, nan],  # onehot missing: 2
        ],
        columns=columns["columns"],
        index=longitudinal_index,
    ),
    "nomissing": DataFrame(
        [
            [0.8, 1, 0, 1],
            [0.87, 1, 0, 1],
            [1.03, 0, 1, 0],
            [0.6, 0, 1, 0],
            [0.65, 0, 1, 0],
            [1.65, 1, 0, 1],
        ],
        columns=columns["columns"],
        index=longitudinal_index,
    ),
    "disc": DataFrame(
        [
            [1, 0, 1, nan, nan, nan],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [nan, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [1, nan, nan, 0, 0, 1],
        ],
        columns=columns["discretized_columns"],
        index=longitudinal_index,
    ),
    "disc_true": DataFrame(
        [
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
        ],
        columns=columns["discretized_columns"],
        index=longitudinal_index,
    ),
    "uniform": DataFrame(
        [
            [1, 0, 1, 1 / 3, 1 / 3, 1 / 3],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [1 / 2, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 1 / 2, 1 / 2, 0, 0, 1],
        ],
        columns=columns["discretized_columns"],
        index=longitudinal_index,
    ),
}
y = Series([0, 1, 1], name="outcome")

splits = {
    "train": [0, 1, 2],
    "val": [3, 4],
    "test": [5],
}
