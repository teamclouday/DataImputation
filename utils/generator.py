# this file contains functions for generating missing values in a dataset

import math
import numpy as np
import pandas as pd
from utils.data import *

def gen_complete_random(data, random_ratio=0.2, print_time=False, print_all=True):
    """Missing Complete At Random (MCAR)

    ### Args
    
    1. `data` - type of `Dataset`
    2. `random_ratio` - defines the missingness across rows and columns
    3. `print_time` - print time to evaluate performance
    4. `print_all` - print all messages, including warnings

    ### Returns
    Converted `Dataset` object with missing values

    -----

    ### Implementation
    1. Select random rows based on random_ratio
    2. For each row, select random cols based on random_ratio
    3. Leave at least one value for each column
    4. Skip protected features

    """
    if print_time:
        tt = time.process_time()
    if random_ratio > 0.5:
        if print_all:
            print("Warning: gen_complete_random, random missing ratio > 0.5")
    X_data = data.X.copy()
    if len(data.protected_features) > 0:
        X_data.drop(columns=data.protected_features, inplace=True)
        X_data_protected = data.X[data.protected_features].copy()
    if len(X_data.shape) != 2:
        print("Error: gen_complete_random only support dataset with rank of 2\nYour input has rank of {0}".format(len(X_data.shape)))
        sys.exit(1)
    random_ratio = random_ratio ** 0.5
    num_rows, num_cols = X_data.shape
    row_rand = np.random.permutation(num_rows)
    row_rand = row_rand[:math.floor(num_rows*random_ratio)]
    for row in row_rand[:-1]:
        col_rand = np.random.permutation(num_cols)
        col_rand = col_rand[:math.floor(num_cols*random_ratio)]
        X_data.iloc[row, col_rand] = np.nan
    row = row_rand[-1]
    col_rand = np.random.permutation(num_cols)
    col_rand = col_rand[:math.floor(num_cols*random_ratio)]
    for col in col_rand:
        if X_data.iloc[:, col].isnull().sum() < (num_rows - 1):
            X_data.iloc[row, col] = np.nan
    if print_all:
        print("gen_complete_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if len(data.protected_features) > 0:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

def gen_random(data, n_columns_observed=2, print_time=False, print_all=True):
    """Missing At Random (MAR)

    ### Args
    
    1. `data` - type of `Dataset`
    2. `n_columns_observed` - decides how many columns without missing value
    3. `print_time` - print time to evaluate performance
    4. `print_all` - print all messages, including warnings

    ### Returns
    Converted `Dataset` object with missing values

    ------

    ### Implementation
    1. For each column `K` to convert:
        a. Draw (`n_columns_observed`+1) scalar values from a random standard normal distribution N(-1, 1)

        b. Construct `M` by multiplying the scalar values on the observed features

        c. Use the standard logistic cumulative distribution function to convert `M` to probability `p`

        d. For each `p`, draw a value randomly from the Binomial(1, `p`) distribution

        e. For each row `i`, if `p`[`i`] = 1, then `K`[`i`] is replaced by NaN
    2. Leave at least one value for each column
    3. Skip protected features
    """
    if print_time:
        tt = time.process_time()
    X_data = data.X.copy()
    if len(data.protected_features) > 0:
        X_data.drop(columns=data.protected_features, inplace=True)
        X_data_protected = data.X[data.protected_features].copy()
    if len(X_data.shape) != 2:
        print("Error: gen_random only support dataset with rank of 2\nYour input has rank of {0}".format(len(X_data.shape)))
        sys.exit(1)
    
    def convert_single_feature(K_df, observed_df):
        scalars = np.random.standard_normal(n_columns_observed)
        scalar_a = np.random.standard_normal()
        observed = observed_df.to_numpy()
        M = scalar_a + (scalars * observed).sum(axis=1)
        p = 1 / (1 + np.exp(-M))
        p_bi = np.random.binomial(1, p).astype(np.bool)
        K_df = K_df.where(~p_bi, other=np.nan)
        return K_df



    if print_all:
        print("gen_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if len(data.protected_features) > 0:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

def gen_not_random(data, print_time=False, print_all=True):
    """Missing Not At Random (MNAR)

    ### Args
    
    1. `data` - type of `Dataset`
    3. `print_time` - print time to evaluate performance
    4. `print_all` - print all messages, including warnings

    ### Returns
    Converted `Dataset` object with missing values

    ------

    ### Implementation
    1. 
    2. Leave at least one value for each column
    3. Skip protected features
    """
    if print_time:
        tt = time.process_time()
    X_data = data.X.copy()
    if len(data.protected_features) > 0:
        X_data.drop(columns=data.protected_features, inplace=True)
        X_data_protected = data.X[data.protected_features].copy()
    if len(X_data.shape) != 2:
        print("Error: gen_not_random only support dataset with rank of 2\nYour input has rank of {0}".format(len(X_data.shape)))
        sys.exit(1)
    
    

    if print_all:
        print("gen_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if len(data.protected_features) > 0:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data