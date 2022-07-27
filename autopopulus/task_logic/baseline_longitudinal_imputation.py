from argparse import Namespace
from typing import Tuple
import pandas as pd
import numpy as np

from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    LOCF,
)

## Local Modules
from autopopulus.data import PAD_VALUE, CommonDataModule

# Alias
InputDataSplit = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def brits(args: Namespace, data: CommonDataModule) -> InputDataSplit:
    # TODO: add arg for this?
    brits = BRITS(data.n_features * 0.5)

    def multi_index_df_to_3d_numpy(df: pd.DataFrame) -> np.ndarray:
        """
        Reshape a 2d multiindex dataframe into a 3d numpy array.
        The longitudinal commondatamodule dataloader will collate everything to be padded to the same sequence length already.
        """
        seq_lens = df.groupby(level=0).size()
        # Ensure they're all the same seq length (dropna bc the diff will have the 1st one be NaN)
        assert (
            seq_lens.diff().dropna().eq(0).all()
        ), "Data passed has unequal sequence lengths."
        return df.values.reshape(-1, max(seq_lens), data.n_features)

    imputer = brits.fit(
        multi_index_df_to_3d_numpy(data.splits["data"]["normal"]["train"])
    )
    X_train = imputer.impute(
        multi_index_df_to_3d_numpy(data.splits["data"]["normal"]["train"])
    )
    X_val = imputer.impute(
        multi_index_df_to_3d_numpy(data.splits["data"]["normal"]["val"])
    )
    X_test = imputer.impute(
        multi_index_df_to_3d_numpy(data.splits["data"]["normal"]["test"])
    )

    return (X_train, X_val, X_test)
