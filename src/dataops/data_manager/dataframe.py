from typing import Iterable, Sequence, Mapping, List

import pandas as pd
from pandas.io.formats.info import DataFrameInfo

from dataops import settings


def get_df_details(df: pd.DataFrame) -> tuple[int, Iterable, Sequence[int], Mapping[str, int], str]:
    df_info = DataFrameInfo(df)
    return (
        df_info.col_count,
        df_info.dtypes,
        df_info.non_null_counts,
        df_info.dtype_counts,
        df_info.memory_usage_string
    )


def get_df_info(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    df_info = pd.DataFrame({"Dtype": DataFrameInfo(df).dtypes,
                            "Non-Null": DataFrameInfo(df).non_null_counts,
                            "Unique": df.nunique()}).reset_index(names="Column")

    if verbose:
        print(f"DataFrame Information:\n{44 * '-'}\n{df_info}")

    return df_info


def drop_columns(df: pd.DataFrame, columns: List, keep=False) -> pd.DataFrame:
    if keep:
        df = df[columns]
    else:
        df = df.drop(columns, axis=1)
    return df


def is_numeric(df):
    return df.apply(pd.api.types.is_numeric_dtype)


def is_below_nunique_limit(df):
    return df.nunique() <= settings.multiclass.max_nunique_for_column
