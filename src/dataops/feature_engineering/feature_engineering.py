import logging

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.api.types import is_numeric_dtype
from collections import Counter
from sklearn.compose import make_column_transformer
import scipy
import numpy as np

import dataops.messages as messages
from dataops.utils.timing import timefunc


# def remove_single_instance_targets(df, count):
#     # Count the number of instances per class
#     class_counts = df['mfr'].value_counts()
#
#     # Identify classes with only one instance
#     single_instance_classes = class_counts[class_counts == count].index.tolist()
#
#     # Create a boolean mask to filter instances of single-instance classes
#     mask = df['mfr'].isin(single_instance_classes)
#
#     # Filter the dataset to remove instances of single-instance classes
#     df = df[~mask]
#
#     return df


# https://datascience.stackexchange.com/questions/39317/difference-between-ordinalencoder-and-labelencoder
# https://datagy.io/sklearn-one-hot-encode/

def encode_data(df, encoder, enc_columns):
    if encoder not in ['ohe', None]:
        raise ValueError(messages.FEN_EX_001_MSG)

    df = df.select_dtypes(exclude=[np.number])

    if encoder == 'ohe':
        transformer = make_column_transformer((OneHotEncoder(drop='first'), enc_columns), remainder='passthrough', verbose_feature_names_out=False)
        df = transformer.fit_transform(df)
        columns = transformer.get_feature_names_out()

        if isinstance(df, scipy.sparse._csr.csr_matrix):
            logging.getLogger('dataops-logger').warning(messages.FEN_EX_002_MSG)
            df = df.toarray()

        df = pd.DataFrame(df, columns=columns)

    return df


def one_hot_encode_column(df, column):
    # ONE HOT ENCODING MANUALLY -> removes original column but keeps all encoded columns
    feature_encoder = OneHotEncoder()
    encoded_features = feature_encoder.fit_transform(df[[column]])
    # print(feature_encoder.categories_)
    df[feature_encoder.categories_[0]] = encoded_features.toarray()


def one_hot_encode(X_train, X_test, columns):
    transformer = make_column_transformer((OneHotEncoder(drop='first'), columns), remainder='passthrough', verbose_feature_names_out=False)
    transformed_train = transformer.fit_transform(X_train)
    transformed_X_train = pd.DataFrame(transformed_train, columns=transformer.get_feature_names_out())
    transformed_test = transformer.fit_transform(X_test)
    transformed_X_test = pd.DataFrame(transformed_test, columns=transformer.get_feature_names_out())

    return transformed_X_train, transformed_X_test


def label_encode(y):
    label_encoder = LabelEncoder()
    transformed_y = label_encoder.fit_transform(y)

    return transformed_y


def aggregate_rare_categories_to_others(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    data_column = Counter(df['Purpose'])
    column_counter = pd.Series(data_column).sort_values(ascending=False)
    return pd.concat([pd.DataFrame({"Occurrences": column_counter.iloc[:threshold]}),
                      pd.DataFrame({"Occurrences": column_counter.iloc[threshold:].sum()},
                                   index=['Others'])],
                     ignore_index=False)


def encode_categorical_data(X_train, X_test, columns, y_train, y_test):
    if columns is not None:
        X_train, X_test = one_hot_encode(X_train, X_test, columns)
    if is_numeric_dtype(y_train):
        y_train = label_encode(y_train)
        y_test = label_encode(y_test)

    return X_train, X_test, y_train, y_test


def get_train_test_X_y(df, target, *args, **kwargs):
    y = df[target]
    X = df.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, *args, **kwargs)
    return X_train, X_test, y_train, y_test
