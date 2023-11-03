from typing import Iterable, Sequence, Mapping, List, Union

import pandas as pd
from pandas.io.formats.info import DataFrameInfo

from dataops import settings


def group_below_top_n(count_df, n, text):
    sorted_df = count_df.sort_values(by=text, ascending=False)
    top_n = sorted_df.iloc[:n]
    others_sum = sorted_df.iloc[n:].sum()
    others = pd.DataFrame({text: [others_sum]}, index=['Others'])
    pie_df = pd.concat([top_n, others], ignore_index=False)
    return pie_df


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


def get_features_index(columns, target):
    column_name_to_integer = {col: i for i, col in enumerate(columns) if col != target}
    # Get the remaining columns as integers in the original order
    return [column_name_to_integer[col] for col in columns if col != target]


def drop_columns(df: pd.DataFrame, columns: List, keep=False) -> pd.DataFrame:
    if keep:
        df = df[columns]
    else:
        df = df.drop(columns, axis=1)
    return df


def is_numeric(df: Union[pd.DataFrame, pd.Series]):
    if isinstance(df, pd.DataFrame):
        return df.apply(pd.api.types.is_numeric_dtype)
    elif isinstance(df, pd.Series):
        return pd.api.types.is_numeric_dtype(df)


def get_num_cols(df):
    return df.columns[is_numeric(df)]


def get_cat_cols(df):
    return df.columns[~is_numeric(df)]


def get_num_and_cat_feats(df, target):
    df = df.drop(target, axis=1)
    return get_num_cols(df), get_cat_cols(df)


def get_cat_columns(df):
    return df.select_dtypes(include='object').columns.tolist()  # exclude=[np.number]


def is_below_nunique_limit(df):
    return df.nunique() <= settings.multiclass.max_nunique_for_column
