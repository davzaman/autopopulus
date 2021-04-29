from typing import List, Dict, Union, Tuple
from sklearn import datasets
import Orange
import pandas as pd
import numpy as np
import re
from Orange.data import Table, ContinuousVariable, DiscreteVariable


def list2dict(lst: List, names: List[str] = []) -> Dict[Union[str, int], List]:
    if len(names) > 1:
        return {names[i]: lst[i] for i in range(len(lst))}
    else:
        return {i: lst[i] for i in range(len(lst))}


def range_for_edge_bins(
    lst: List[Tuple[str, ...]], mins: pd.Series, maxs: pd.Series
) -> List[Tuple[str, ...]]:
    """Because we want to undiscretize later, we want to give explicit ranges for the edge bins.
    The edge bins are >/≥ val, or </≤ val. We want a - b instead.
    """
    decimal_regex = r"\d*\.?\d*"
    lower_edge_regex = r"[<≤]"
    upper_edge_regex = r"[>≥]"
    new_ranges = []
    for ranges, mi, ma in zip(lst, mins, maxs):
        # Grab the decimals from the string describing the bin.
        high = float(
            re.search(
                r".*" + lower_edge_regex + r"\s(" + decimal_regex + r")", ranges[0]
            ).group(1)
        )
        low = float(
            re.search(
                r".*" + upper_edge_regex + r"\s(" + decimal_regex + r")", ranges[-1]
            ).group(1)
        )
        if len(ranges) > 2:
            new_ranges.append((f"{mi} - {high}", *ranges[1:-1], f"{low} - {ma}"))
        else:
            new_ranges.append((f"{mi} - {high}", f"{low} - {ma}"))

    return new_ranges


def get_discretized_MDL_data(dataTable: Table, force: bool = True):
    disc = Orange.preprocess.Discretize()
    disc.method = Orange.preprocess.discretize.EntropyMDL(force=force)
    return disc(dataTable), disc


def dfMapColumnValues(df: pd.DataFrame, cols: List[str], dicts: Dict) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].map(dicts[col])
    return df


def orange2Df(
    data: Table, cols: List[str], dicts: Dict, mapped: bool = True
) -> pd.DataFrame:
    X = data.X
    df = pd.DataFrame(data=X, columns=cols)
    return dfMapColumnValues(df, cols, dicts) if mapped else df


def dict2list(d: Dict[str, Dict[int, str]]) -> List[float]:
    """Generates pandas bins from Orange discretizer dicts object."""
    bins = []
    for i, val in enumerate(d.values()):
        bin1, bin2 = val.replace(" ", "").split("-")
        bins.extend([float(bin1), float(bin2)])
    return np.unique(bins)


def df2Orange(df: pd.DataFrame, y: pd.Series, continuous_cols: List[str]) -> Table:
    # label feature
    class_values = list(y.map(str).unique())
    class_var = DiscreteVariable("class_var", values=class_values)
    # continuous features
    domain = Orange.data.Domain(
        [ContinuousVariable(col) for col in continuous_cols], class_vars=class_var
    )

    data = Orange.data.Table.from_numpy(
        domain, pd.concat([df[continuous_cols], y], axis=1).values
    )
    return data


def merge_disc_vars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge newly discretizes vars with original dataframe."""
    # the new discretized df will have its index reset already
    return pd.concat([old.reset_index(), new], axis=1, ignore_index=False).drop(
        ["index"], axis=1
    )


class MDLDiscretizer:
    def __init__(self):
        self.cols = []
        self.disc = None
        self.list_of_values = []
        self.dicts = []

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        continuous_cols: List[str],
        force: bool = True,
    ):
        cont_data = df2Orange(df, y, continuous_cols)
        self.mins = df[continuous_cols].min()
        self.maxs = df[continuous_cols].max()
        self._fit(cont_data, force)
        return self

    def _fit(self, cont_data: Table, force: bool = True):
        d_cont_data, self.disc = get_discretized_MDL_data(cont_data, force=force)
        self.cols = [attr.name for attr in d_cont_data.domain.attributes]
        self.list_of_values = [attr.values for attr in d_cont_data.domain.attributes]
        self.dicts = list2dict(
            [
                list2dict(values)
                for values in range_for_edge_bins(
                    self.list_of_values, self.mins, self.maxs
                )
            ],
            self.cols,
        )

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        continuous_cols: List[str],
        force: bool = True,
        mapped: bool = True,
    ) -> pd.DataFrame:
        cont_data = df2Orange(df, y, continuous_cols)
        self.mins = df[continuous_cols].min()
        self.maxs = df[continuous_cols].max()
        df_disc = self._fit_transform(cont_data, force, mapped)
        # merge it back to match a normal transform that operates on the whole df
        return merge_disc_vars(df[df.columns.difference(continuous_cols)], df_disc)

    def _fit_transform(
        self, cont_data: Table, force: bool = True, mapped: bool = True
    ) -> pd.DataFrame:
        d_cont_data, self.disc = get_discretized_MDL_data(cont_data, force=force)
        self.cols = [attr.name for attr in d_cont_data.domain.attributes]
        self.list_of_values = [attr.values for attr in d_cont_data.domain.attributes]
        self.dicts = list2dict(
            [
                list2dict(values)
                for values in range_for_edge_bins(
                    self.list_of_values, self.mins, self.maxs
                )
            ],
            self.cols,
        )
        return orange2Df(
            data=d_cont_data, cols=self.cols, dicts=self.dicts, mapped=mapped
        )

    def transform(
        self, df: pd.DataFrame, continuous_cols: List[str], mapped: bool = True
    ) -> pd.DataFrame:
        """Transforms df directly, no need to convert to Orange Table.
        Orange has an issue where transform will fit the data again, this is a workaround.
        """
        cont_data = df.copy()  # to keep original df intact

        for col in continuous_cols:
            bins = dict2list(self.dicts[col])
            if mapped:
                labels = list(self.dicts[col].values())
            else:
                labels = list(range(len(bins) - 1))
            cont_data.loc[:, col] = pd.cut(
                cont_data[col],
                bins=bins,
                right=True,
                labels=labels,
                include_lowest=True,
            )

        return cont_data


if __name__ == "__main__":
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])

    print("Original data: ")
    print(iris_df[:3])
    print("\n")
    print("Fitting data ...")
    print("\n")
    discritizer = MDLDiscretizer()
    discritizer.fit(iris_df, pd.Series(iris["target"]), iris["feature_names"])
    print("List of discretizations: ")
    print(discritizer.dicts)
    print("\n")
    print("Transformed data: ")
    print(discritizer.transform(iris_df[:3], iris["feature_names"]))
