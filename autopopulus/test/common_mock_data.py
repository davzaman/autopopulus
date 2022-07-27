import pandas as pd
import numpy as np

seed = 0
columns = {
    "columns": ["age", "weight", "ismale", "fries_s", "fries_m", "fries_l"],
    "ctn_cols": ["age", "weight"],
    "onehot_prefix_names": ["fries"],
}
indices = {"ctn_cols": [0, 1], "cat_cols": [2, 3, 4, 5]}
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

col_indices_by_type = {"continuous": [0, 1], "categorical": [2, 3, 4, 5]}

groupby = {
    "categorical_only": {
        "discretize": {
            "categorical_onehots": {1: "fries", 2: "fries", 3: "fries"},
            "binary_vars": {0: "ismale"},
        },
        "no_discretize": {
            "categorical_onehots": {3: "fries", 4: "fries", 5: "fries"},
            "binary_vars": {2: "ismale"},
        },
    },
    "after_fit": {
        "discretize": {
            "categorical_onehots": {1: "fries", 2: "fries", 3: "fries"},
            "binary_vars": {0: "ismale"},
            "discretized_ctn_cols": {
                "data": {
                    4: "age",
                    5: "age",
                    6: "age",
                    7: "weight",
                    8: "weight",
                    9: "weight",
                    10: "weight",
                },
                "ground_truth": {
                    4: "age",
                    5: "age",
                    6: "age",
                    7: "weight",
                    8: "weight",
                    9: "weight",
                    10: "weight",
                },
            },
        },
        "no_discretize": {
            "categorical_onehots": {3: "fries", 4: "fries", 5: "fries"},
            "binary_vars": {2: "ismale"},
        },
    },
}

X = {
    "X": pd.DataFrame(
        [
            [44, np.nan, 0, 0, 1, 0],
            [39, 57.2, 1, 0, 0, 1],
            [26, 26.3, 0, np.nan, np.nan, np.nan],
            [16, 73.4, 1, 1, 0, 0],
            [np.nan, 56.5, 1, 0, 1, 0],
            [57, 29.6, 0, 1, 0, 0],
        ],
        columns=columns["columns"],
    ),
    "nomissing": pd.DataFrame(
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
    "scale": pd.DataFrame(
        [
            [1, np.nan, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [-2.6, -30.9, 0, np.nan, np.nan, np.nan],
            [-4.6, 16.2, 1, 1, 0, 0],
            [np.nan, -0.7, 1, 0, 1, 0],
            [3.6, -27.6, 0, 1, 0, 0],
        ],
        columns=columns["columns"],
    ),
    "disc": pd.DataFrame(
        [
            [0, 0, 1, 0, 0, 0, 1, np.nan, np.nan, np.nan, np.nan],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, np.nan, np.nan, np.nan, 0, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, np.nan, np.nan, np.nan, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        ],
        columns=columns["discretized_columns"],
    ),
    "disc_true": pd.DataFrame(
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
    "uniform": pd.DataFrame(
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
}
y = pd.Series([1, 0, 1, 0, 1, 0], name="outcome")

splits = {
    "train": [0, 1],
    "train_FO": [1],  # fully observed
    "val": [2, 3],
    "val_FO": [3],
    "test": [4, 5],
    "test_FO": [5],
}
