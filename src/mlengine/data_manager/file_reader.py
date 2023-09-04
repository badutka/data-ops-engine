from typing import Iterable, Sequence, Mapping

import pandas as pd
from pandas.io.formats.info import DataFrameInfo


def get_df_details(df: pd.DataFrame) -> tuple[int, Iterable, Sequence[int], Mapping[str, int], str]:
    df_info = DataFrameInfo(df)
    return (
        df_info.col_count,
        df_info.dtypes,
        df_info.non_null_counts,
        df_info.dtype_counts,
        df_info.memory_usage_string
    )
