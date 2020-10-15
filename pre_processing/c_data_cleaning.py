import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum
from pandas.api.types import is_numeric_dtype
import math
from sklearn.metrics import jaccard_score

import pandas as pd
import numpy as np

from pre_processing.b_data_profile import *
from pre_processing.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    numeric_columns = get_numeric_columns(df)
    if column in numeric_columns:
        col = df[column]
        col_array = col.to_numpy()
        if must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
            for value in np.nditer(col_array, op_flags=['readwrite']):
                if not(math.isnan(value)) and value <= 0:
                    value[...] = np.nan
        elif must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
            for value in np.nditer(col_array, op_flags=['readwrite']):
                if not(math.isnan(value)) and value >= 0:
                    value[...] = np.nan
        elif must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
            for value in np.nditer(col_array, op_flags=['readwrite']):
                if not(math.isnan(value)) and value <= must_be_rule_optional_parameter:
                    value[...] = np.nan
        elif must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
            for value in np.nditer(col_array, op_flags=['readwrite']):
                if not(math.isnan(value)) and value >= must_be_rule_optional_parameter:
                    value[...] = np.nan

        df[column] = col_array
        return df
    else:
        print("Not A NUMERIC Column")
        return df









def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """
    # Reference : https://www.pluralsight.com/guides/cleaning-up-data-from-outliers
    # Using inter quartile range to identify outliers and replacing them
    numeric_columns = get_numeric_columns(df)
    if column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        df[column] = np.where(df[column] < (Q1 - 1.5 * IQR),Q1, df[column])
        df[column] = np.where(df[column] > (Q3 + 1.5 * IQR),Q3, df[column])
        return df

    else:
        print("Not a numeric column to perform Outlier analysis")
        return df




def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    # Filling nans of a numeric column with mean of the column
    numeric_columns = get_numeric_columns(df)
    if column in numeric_columns:
        df[column].fillna(value=get_column_mean(df,column), inplace=True)
        return df
    else:
        print("Column is not numeric to fill NAN")
        df = df.dropna(subset=[column])
        return df




def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    if is_numeric_dtype(df_column):
        min_max_df_column = (df_column - df_column.min()) / (df_column.max() - df_column.min())
        return min_max_df_column
    else:
        print("Not a numeric column to normalize")
        return df_column



def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    if is_numeric_dtype(df_column):
        normalized_df_column = (df_column - df_column.mean()) / df_column.std()
        return normalized_df_column
    else:
        print("Not a numeric column to standardize")
        return df_column



def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series, distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    #Reference :https://www.geeksforgeeks.org/pandas-compute-the-euclidean-distance-between-two-series/\
    #Reference: https://medium.com/analytics-vidhya/various-types-of-distance-metrics-machine-learning-cc9d4698c2da#:~:text=Manhattan%20distance%20is%20a%20metric,%2Dcoordinates%20and%20y%2Dcoordinates.

    if is_numeric_dtype(df_column_1) and is_numeric_dtype(df_column_2):
        if len(df_column_1) == len(df_column_2):
            if distance_metric == DistanceMetric.EUCLIDEAN:
                eucledean_dist = np.sqrt((df_column_1 - df_column_2) * (df_column_1 - df_column_2))
                return eucledean_dist
            elif distance_metric == DistanceMetric.MANHATTAN:
                manhattan_dist = abs(df_column_1 - df_column_2)
                return manhattan_dist
        else:
            print("Columns are not of same length")
            return pd.Series([np.int64])
    else:
        print("Either of columns are not numeric to perform distance measuremnet")
        return pd.Series([np.int64])


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    binary_columns_1 = get_binary_columns(df_column_1.to_frame())
    binary_columns_2 = get_binary_columns(df_column_2.to_frame())

    if df_column_1.name in binary_columns_1 and df_column_2.name in binary_columns_2:
        if len(df_column_1) == len(df_column_2):
            df_column_1 = df_column_1.to_numpy()
            df_column_2 = df_column_2.to_numpy()
            distance_array=np.zeros(len(df_column_1))
            for idx, x in np.ndenumerate(df_column_1):
                if df_column_1[idx] == df_column_2[idx]:
                    distance_array[idx] = False
                else:
                    distance_array[idx] = True
            distance_series = pd.Series(distance_array)
            return distance_series
        else:
            print("Given columns are not of same length")
            return pd.Series([np.int64])
    else:
        print("Either of columns are not binary  to perform binary distance")
        return pd.Series([np.int64])




if __name__ == "__main__":
    df = pd.DataFrame({'a':[1,2,3,None], 'd':[False,False,True,False ],'b': [True, True, False, True], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    d2 = read_dataset(Path('..', '..', 'iris.csv'))
    assert fix_outliers(d2, d2.columns[0]) is not None
    assert fix_nans(d2, d2.columns[4]) is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(d2["sepal_length"], d2["sepal_width"], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(d2["sepal_length"], d2["sepal_width"], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'd']) is not None
    print("ok")
