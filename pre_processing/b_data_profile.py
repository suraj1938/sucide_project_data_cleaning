from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from pre_processing.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df:
        print("No Column With the Given Name Exists")
        return 0
    column = df[column_name]
    max_value = np.nanmax(column)
    return max_value


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df:
        print("No Column With the Given Name Exists")
        return 0
    column = df[column_name]
    min_value = np.nanmin(column)
    return min_value


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df:
        print("No Column With the Given Name Exists")
        return 0
    column = df[column_name]
    if column.dtype == np.object:
        print("Cannot find mean of the column since it is not an int or float")
        return 0
    mean_value=np.nanmean(column)
    return mean_value


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    """
    This is also known as the number of 'missing values'
    """
    if column_name not in df:
        print("No Column With the Given Name Exists")
        return 0
    column = df[column_name]
    count_nan=column.isna().sum()
    return count_nan



def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df:
        print("No Column With the Given Name Exists")
        return 0
    duplicate_count=len(df[column_name]) - len(df[column_name].drop_duplicates())
    return duplicate_count


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    # Returning only numeric columns by removing binary columns
    bool_cols = get_binary_columns(df)
    cols = df.select_dtypes([np.number]).columns.tolist()
    numeric_cols = list(set(cols) - set(bool_cols))
    return numeric_cols


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    bool_cols = [col for col in df
                 if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    return bool_cols


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    all_cols = df.columns.tolist()
    num_cols = get_numeric_columns(df)
    binary_cols = get_binary_columns(df)
    categorical_columns = list(set(all_cols) - (set(num_cols) | set(binary_cols)))
    return categorical_columns


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate and return the pearson correlation between two columns
    """
    # Reference : https://realpython.com/numpy-scipy-pandas-correlation-python/
    if col1 not in df or col2 not in df:
        print("No Column With the Given Name Exists")
        return 0
    text_column_names=get_text_categorical_columns(df)
    if col1 in text_column_names or col2 in text_column_names:
        print("Cannot find co-relation for categorical columns")
        return 0
    column1 = df[col1]
    column2 = df[col2]
    column1 = column1.to_numpy()
    column2 = column2.to_numpy()

    correlation_matrix = np.corrcoef(column1, column2)
    pearson_coef = correlation_matrix[0,1]
    return pearson_coef



if __name__ == "__main__":
    df = read_dataset(Path('..', '..', 'iris.csv'))

    a = pandas_profile(df)

    assert get_column_max(df, df.columns[0]) is not None
    assert get_column_min(df, df.columns[0]) is not None
    assert get_column_mean(df, df.columns[0]) is not None
    assert get_column_count_of_nan(df, df.columns[0]) is not None
    assert get_column_number_of_duplicates(df, df.columns[4]) is not None
    assert get_numeric_columns(df) is not None
    assert get_binary_columns(df) is not None
    assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[2], df.columns[3]) is not None
    print("ok")
