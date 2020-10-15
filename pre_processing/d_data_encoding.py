import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from pre_processing.b_data_profile import *
from pre_processing.c_data_cleaning import *

from sklearn.compose import ColumnTransformer
from pre_processing.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    """
    This method should generate a (sklearn version of a) label encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column

    """
    text_column_list = get_text_categorical_columns(df_column.to_frame())
    if df_column.name in text_column_list:
        labelencoder = LabelEncoder()
        labelencoder.fit(df_column)
        return labelencoder
    else:
        print("Given column is not a categorical column")
        return None





def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    """
    This method should generate a (sklearn version of a) one hot encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    text_column_list = get_text_categorical_columns(df_column.to_frame())
    if df_column.name in text_column_list:
        oenc = OneHotEncoder(handle_unknown='ignore')
        oenc.fit(df_column.to_numpy().reshape(-1,1))
        return oenc
    else:
        print("Given column is not a categorical column")
        return None






def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should replace the column of df with the label encoder's version of the column
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to replace the column
    :return: The df with the column replaced with the one from label encoder
    """
    # Reference : https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
    text_column_list = get_text_categorical_columns(df)
    df_copy=df.copy()
    if column in text_column_list:
        df_copy[column] = le.transform(df_copy[column])
        return df_copy
    else:
        print("Given column is not a categorical column")
        return None



def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder, ohe_column_names: List[str]) -> pd.DataFrame:
    """
    This method should replace the column of df with all the columns generated from the one hot's version of the encoder
    Feel free to do it manually or through a sklearn ColumnTransformer
    :param df: Dataset
    :param column: column to be replaced
    :param ohe: the one hot encoder to be used to replace the column
    :param ohe_column_names: the names to be used as the one hot encoded's column names
    :return: The df with the column replaced with the one from label encoder
    """
    # Reference: https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
    text_column_list = get_text_categorical_columns(df)
    df_copy = df.copy()
    if column in text_column_list:

        col_list = list(df_copy.columns.values)
        col_list.remove(column)
        enc_df = pd.DataFrame(ohe.fit_transform(df_copy[[column]]).toarray())
        df_copy = df_copy.drop(column, axis=1)
        df_copy = enc_df.join(df_copy)
        df_copy.columns = ohe_column_names + col_list
        return df_copy
    else:
        print("Given column is not a categorical column")
        return None



def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_label_encoder
    The column of df should be from the label encoder, and you should use the le to revert the column to the previous state
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to revert the column
    :return: The df with the column reverted from label encoder
    """
    df[column] = le.inverse_transform(df[column])
    return df



def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_one_hot_encoder
    The columns (one of the method's arguments) are the columns of df that were placed there by the OneHotEncoder.
    You should use the ohe to revert these columns to the previous state (single column) which was present previously
    :param df: Dataset
    :param columns: the one hot encoded columns to be replaced
    :param ohe: the one hot encoder to be used to revert the columns
    :param original_column_name: the original column name which was used before being replaced with the one hot encoded version of it
    :return: The df with the columns reverted from the one hot encoder
    """
    df[original_column_name] = ohe.inverse_transform(df[columns])
    for x in columns:
        df.drop(x, inplace=True, axis=1)
    return df


if __name__ == "__main__":
    df = pd.DataFrame({'a':[1,2,3,4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    le = generate_label_encoder(df.loc[:, 'c'])
    assert le is not None
    ohe = generate_one_hot_encoder(df.loc[:, 'c'])
    assert ohe is not None
    assert replace_with_label_encoder(df, 'c', le) is not None
    assert replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())) is not None
    assert replace_label_encoder_with_original_column(replace_with_label_encoder(df, 'c', le), 'c', le) is not None
    assert replace_one_hot_encoder_with_original_column(replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())),
                                                        list(ohe.get_feature_names()),
                                                        ohe,
                                                        'c') is not None
    print("ok")