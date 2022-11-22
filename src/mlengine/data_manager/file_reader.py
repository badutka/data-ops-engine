from typing import Iterable, Sequence, Mapping

import pandas as pd
from pandas.io.formats.info import DataFrameInfo


def get_df_details(df: pd.DataFrame) -> tuple[int, Iterable, Sequence[int], Mapping[str, int], str]:
    return (DataFrameInfo(df).col_count,
            DataFrameInfo(df).dtypes,
            DataFrameInfo(df).non_null_counts,
            DataFrameInfo(df).dtype_counts,
            DataFrameInfo(df).memory_usage_string)
