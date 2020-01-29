# this file contains codes for generating missing values in a dataset

import sys
import math
import numpy as np
import pandas as pd
from utils.data import *

# missingness complete at random
# take a Dataset object as input
# take a random_ratio as input
# Implementation:
# select random rows based on random_ratio
# then for each of these rows, select random cols based on random_ratio
# replace these entries with value of NaN
# return the converted dataset object
def gen_complete_random(data, random_ratio=0.3):
    if random_ratio > 0.5:
        print("Warning: gen_complete_random, random missing ratio > 0.5")
    X_data = data.X.copy()
    if len(X_data.shape) != 2:
        print("Error: gen_complete_random only support dataset with rank of 2")
        sys.exit(1)
    num_rows, num_cols = X_data.shape
    row_rand = np.random.permutation(num_rows)
    row_rand = row_rand[:math.floor(num_rows*random_ratio)]
    for row in row_rand:
        col_rand = np.random.permutation(num_cols)
        col_rand = col_rand[:math.floor(num_cols*random_ratio)]
        for col in col_rand:
            X_data.iloc[row, col] = np.nan
    print("gen_complete_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    data.X = X_data
    return data

# missingness at random
def gen_random(data):
    pass

# missingness that depends on unobserved predictors
def gen_unobserved(data):
    pass

# missingness that depends on missing value itself
def gen_by_itself(data):
    pass
