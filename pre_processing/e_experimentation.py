import collections
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum
import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from pre_processing.b_data_profile import *
from pre_processing.c_data_cleaning import *
from pre_processing.d_data_encoding import *
from pre_processing.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_who_sucide_data() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 4 columns:
    three numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    df = read_dataset(Path('who_suicide_statistics.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)



    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = normalize_column(df.loc[:, nc])

    distances = pd.DataFrame()

    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))
    df.to_csv("processed_suicide.csv")
    print(df.head)
    return df





if __name__ == "__main__":
    assert process_who_sucide_data() is not None
    print("ok")
