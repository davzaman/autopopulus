from enum import Enum
from typing import Dict, List, Tuple, Union
from pandas import DataFrame, Series, Index
from numpy import ndarray

# aliases
DataT = Union[ndarray, DataFrame]
LabelT = Union[ndarray, Series]
DataColumn = Union[Index, List[str]]
# everything treated as static, labels
StaticFeatureAndLabel = Tuple[DataFrame, Series]
# longitudinal, static, labels
LongitudinalFeatureAndLabel = Dict[str, DataFrame]


class DataTypeTimeDim(Enum):
    STATIC = 0
    LONGITUDINAL = 1
    STATIC_SUBSET = 2
    LONGITUDINAL_SUBSET = 3

    def is_longitudinal(self) -> bool:
        return self.value in (1, 3)
