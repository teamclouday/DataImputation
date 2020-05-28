# this file contains functions for generating missing values in a dataset

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
# IMPORTANT: leave at least one value for each column
# IMPORTANT: skip the protected features
# return the converted dataset object
def gen_complete_random(data, random_ratio=0.2, print_time=False, print_all=True):
    if print_time:
        tt = time.process_time()
    if random_ratio > 0.5:
        if print_all:
            print("Warning: gen_complete_random, random missing ratio > 0.5")
    X_data = data.X.copy()
    if data.protected:
        X_data.drop(columns=data.protected, inplace=True)
        X_data_protected = data.X[data.protected].copy()
    if len(X_data.shape) != 2:
        print("Error: gen_complete_random only support dataset with rank of 2\nYour input has rank of {0}".format(len(X_data.shape)))
        sys.exit(1)
    random_ratio = random_ratio ** 0.5
    num_rows, num_cols = X_data.shape
    row_rand = np.random.permutation(num_rows)
    row_rand = row_rand[:math.floor(num_rows*random_ratio)]
    for row in row_rand:
        col_rand = np.random.permutation(num_cols)
        col_rand = col_rand[:math.floor(num_cols*random_ratio)]
        for col in col_rand:
            if X_data.iloc[:, col].isnull().sum() < (num_rows - 1):
                X_data.iloc[row, col] = np.nan
    if print_all:
        print("gen_complete_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if data.protected:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# missingness at random
# take a Dataset object as input
# take random_ratio as input
# take user-defined random_cols as input
# Implementation:
# generate random_cols if not specified
# calculate the ratio for selecting random rows
# for each random row
# for each selected col on that row
# replace the value of the entry with NaN
# IMPORTANT: leave at least one value for each column
# IMPORTANT: skip the protected features
# return the Dataset object
def gen_random(data, random_ratio=0.2, random_cols=[], print_time=False, print_all=True):
    if print_time:
        tt = time.process_time()
    if random_ratio > 0.5:
        if print_all:
            print("Warning: gen_random, random missing ratio > 0.5")
    X_data = data.X.copy()
    if data.protected:
        X_data.drop(columns=data.protected, inplace=True)
        X_data_protected = data.X[data.protected].copy()
    if len(X_data.shape) != 2:
        print("Error: gen_random only support dataset with rank of 2\nYour input has rank of {0}".format(len(X_data.shape)))
        sys.exit(1)
    random_ratio = random_ratio ** 0.5
    num_rows, num_cols = X_data.shape
    if random_cols == []:
        random_cols = np.random.permutation(num_cols)
        random_cols = random_cols[:math.floor(num_cols*random_ratio)]
    ratio_rows = random_ratio ** 2 / (len(random_cols) / num_cols)
    random_rows = np.random.permutation(num_rows)
    random_rows = random_rows[:math.floor(num_rows*ratio_rows)]
    for row in random_rows:
        for col in random_cols:
            if X_data.iloc[:, col].isnull().sum() < (num_rows - 1):
                X_data.iloc[row, col] = np.nan
    if print_all:
        print("gen_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if data.protected:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# missingness that depends on unobserved predictors
def gen_unobserved(data):
    pass

# missingness that depends on missing value itself
def gen_by_itself(data):
    pass
