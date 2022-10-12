from typing import List, Tuple
import numpy as np
import pandas as pd
from Orange.data import Table, Domain
from Orange.data.variable import DiscreteVariable
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames

from autopopulus.data.utils import onehot_multicategorical_column


def build_onehot_from_hypothesis(
    df: pd.DataFrame, onehot_prefixes: List[str]
) -> pd.DataFrame:
    dfs = []
    for col in df.columns:
        if col in onehot_prefixes:
            # enforce df so its compatible witih this function for convenience
            dfs.append(onehot_multicategorical_column(col)(pd.DataFrame(df[col])))
        else:
            dfs.append(df[col])
    return pd.concat(dfs, axis=1)


def mock_disc_data(mock_MDL, df_disc_only: pd.DataFrame, y: pd.Series):
    domain = Domain(
        [DiscreteVariable(col) for col in df_disc_only.columns],
        class_vars=DiscreteVariable("class_var", values=["0", "1"]),
    )
    disc = None
    mock_MDL.return_value = (
        Table.from_numpy(domain, pd.concat([df_disc_only, y], axis=1).values),
        disc,
    )


def create_fake_disc_data(
    rng: np.random.Generator,
    nsamples: int,
    cuts: List[List[Tuple[float, float]]],
    category_names: List[List[str]],
    ctn_cols: List[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            # ensure this is properly onehot encoded via CategoricalDtype
            pd.get_dummies(
                pd.DataFrame(
                    # need the actual value placed to be category name for dummies to work so we use numerical -> name map
                    map(
                        lambda encoded_int: category_names[col_i][encoded_int],
                        rng.integers(0, len(cuts[0]), nsamples),
                    ),
                    columns=[colname],
                ).astype(pd.CategoricalDtype(category_names[col_i])),
                columns=[colname],
            )
            for col_i, colname in enumerate(ctn_cols)
        ],
        axis=1,
    )
