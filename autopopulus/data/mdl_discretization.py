from typing import List, Dict, Optional, Tuple, Union
from sklearn import datasets
import pandas as pd
import numpy as np
from Orange.data import Domain, Table, ContinuousVariable, DiscreteVariable
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EntropyMDL

from autopopulus.data.utils import onehot_multicategorical_column

Bin = Tuple[float, float]
ColInfoDict = Dict[str, Dict[str, Union[List[Bin], List[int]]]]


class MDLDiscretizer:
    """
    Discretize with MDL using the Orange package.
    Transforms df and also has a dictionary available `continuous_to_categorical_mapping`:
    It maps a column name (str) to its categorical:
        - "bins" (a list of value ranges, List[Tuple[float, float]])
        - "labels" (strings representing the bins, List[str])
        -"indices" in the discretized df (List[int]).
    """

    def __init__(self, colname_to_idx: Optional[Dict[str, int]] = None):
        self.disc = None
        self.ctn_to_cat_map = []
        self.colname_to_idx = colname_to_idx

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        continuous_cols: List[str],
        force: bool = True,
    ):
        if self.colname_to_idx is None:
            self.colname_to_idx = {
                colname: idx for idx, colname in enumerate(df.columns)
            }

        cont_data = self.df2Orange(df, y, continuous_cols)
        mins = df[continuous_cols].min()
        maxs = df[continuous_cols].max()

        d_cont_data, self.disc = self.get_discretized_MDL_data(cont_data, force=force)
        cols = [attr.name for attr in cont_data.domain.attributes]
        list_of_values = [attr.values for attr in d_cont_data.domain.attributes]

        self.ctn_to_cat_map = {
            col_name: {
                # Bins for that column as a list of tuples
                "bins": bin_ranges,
                "labels": [
                    self.bin_tuple_to_str(bin_range) for bin_range in bin_ranges
                ],
            }
            for col_name, bin_ranges in zip(
                cols,
                self.bin_ranges_as_tuples(list_of_values, mins, maxs),
            )
        }

        # bookkeeping for getting indices of new discretized columns grouped by their original continuous column name.
        # since later cols will be pushed even further back by earlier columns
        index_adjustment = 0
        num_not_discretized_vars = df.shape[1] - len(self.ctn_to_cat_map)
        for i, (col, col_info_dict) in enumerate(self.ctn_to_cat_map.items()):
            bins = col_info_dict["bins"]
            # indices for each categorical indicator var (once discretized) in discretized df
            # self.colname_to_idx[col] + index_adjustment
            self.ctn_to_cat_map[col]["indices"] = (
                np.arange(len(bins)) + index_adjustment + num_not_discretized_vars
            )
            index_adjustment += len(bins)  # - 1
            # We ignore the nan dummy columns, they will be removed upon transform

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms df directly, no need to convert to Orange Table.
        Orange has an issue where transform will fit the data again, this is a workaround.
        """
        transform_df = df.copy()  # to keep original df intact

        for col, col_info_dict in self.ctn_to_cat_map.items():
            bins = col_info_dict["bins"]
            labels = col_info_dict["labels"]

            # pandas cut expects a list of the bin edges
            bin_points = np.unique(
                [range_point for range_tuple in bins for range_point in range_tuple]
            )
            # Ref: https://stackoverflow.com/a/45169454/1888794
            unbounded_bin_points = bin_points.copy().astype(float)  # float to allow inf
            unbounded_bin_points[0] = -np.inf
            unbounded_bin_points[-1] = np.inf
            # cut and create a pandas categorical column that includes info of all possible categories, even if it's not in the data itself.
            # https://stackoverflow.com/questions/37425961/dummy-variables-when-not-all-categories-are-present
            transform_df.loc[:, col] = pd.cut(
                transform_df[col],
                labels=labels,  # labels will reflect the 2-side bounded ranges learned by data
                bins=unbounded_bin_points,  # but our bins will actually be unbounded at the lower and upper bounds to accomodate for unseen values in ground_truth
                right=True,
                include_lowest=True,
            ).astype(pd.CategoricalDtype(categories=labels))

        return onehot_multicategorical_column(self.ctn_to_cat_map.keys())(transform_df)

    @staticmethod
    def bin_ranges_as_tuples(
        lst: List[Tuple[str, ...]], mins: pd.Series, maxs: pd.Series
    ) -> List[List[Tuple[float, float]]]:
        """Because we want to undiscretize later, we want to give explicit ranges for the edge bins.
        The edge bins are >/≥ a, a - b, or </≤ a. We want (a, b) instead.
        """
        edge_range_indicators = "<≤ >≥"
        list_of_ranges_per_col = []
        for ranges, min_col_value, max_col_value in zip(lst, mins, maxs):
            # Grab the decimals from the string describing the bin.
            end_of_first_bin = float(ranges[0].strip(edge_range_indicators))
            start_of_last_bin = float(ranges[-1].strip(edge_range_indicators))

            # add ranges as tuples to list
            column_ranges = [(min_col_value, end_of_first_bin)]
            # if there's more than 2 bins, split the "-" and add range tuple
            for range_str in ranges[1:-1]:
                start_range, end_range = range_str.replace(" ", "").split("-")
                column_ranges.append((float(start_range), float(end_range)))
            # min=max: don't want to add a bin back to the min, (it will be binary)
            if max_col_value > min_col_value:
                column_ranges.append((start_of_last_bin, max_col_value))

            # add list of all ranges for each column to a list
            list_of_ranges_per_col.append(column_ranges)

        return list_of_ranges_per_col

    @staticmethod
    def get_discretized_MDL_data(dataTable: Table, force: bool = True):
        disc = Discretize(method=EntropyMDL(force=force))
        return disc(dataTable), disc

    @staticmethod
    def orange2Df(
        data: Table, cols: List[str], dicts: Dict, mapped: bool = True
    ) -> pd.DataFrame:
        df = pd.DataFrame(data=data.X, columns=cols)
        if mapped:
            for col in cols:
                df[col] = df[col].map(dicts[col])
            return df
        return df

    @staticmethod
    def df2Orange(df: pd.DataFrame, y: pd.Series, continuous_cols: List[str]) -> Table:
        # label feature
        class_values = list(y.map(str).unique())
        class_var = DiscreteVariable("class_var", values=class_values)
        # continuous features
        domain = Domain(
            [ContinuousVariable(col) for col in continuous_cols], class_vars=class_var
        )

        data = Table.from_numpy(
            domain, pd.concat([df[continuous_cols], y], axis=1).values
        )
        return data

    @staticmethod
    def bin_tuple_to_str(bin: Tuple[float, float]) -> str:
        return f"{bin[0]} - {bin[1]}"


if __name__ == "__main__":
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])

    print("Original data: ")
    print(iris_df[:3])
    print("\n")
    print("Fitting data ...")
    print("\n")
    discretizer = MDLDiscretizer()
    discretizer.fit(iris_df, pd.Series(iris["target"]), iris["feature_names"])
    print("List of discretizations: ")
    print(discretizer.ctn_to_cat_map)
    print("\n")
    print("Transformed data: ")
    print(discretizer.transform(iris_df[:3], iris["feature_names"][:2]))
