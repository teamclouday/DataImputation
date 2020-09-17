# this file contains functions for generating missing values in a dataset

import math
import numpy as np
import pandas as pd
from utils.data import *
from sklearn.preprocessing import StandardScaler

def gen_complete_random(data, random_ratio=0.2, print_time=False, print_all=True):
    """Missing Complete At Random (MCAR)

    ### Args
    
    1. `data` - type of `Dataset`
    2. `random_ratio` - defines the missingness across rows and columns
    3. `print_time` - print time to evaluate performance
    4. `print_all` - print all messages, including warnings

    ### Returns
    Converted `Dataset` object with missing values

    ### Exception
    Will raise an exception if `data.X` is not rank 2

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
    if random_ratio == 0.0:
        return data.copy()
    if len(data.protected_features) > 0:
        X_data.drop(columns=data.protected_features, inplace=True)
        X_data_protected = data.X[data.protected_features].copy()
    if len(X_data.shape) != 2:
        raise Exception("Error: gen_complete_random only support dataset with rank of 2, but your input has rank of {0}".format(len(X_data.shape)))

    random_ratio = random_ratio ** 0.5
    num_rows, num_cols = X_data.shape
    row_rand = np.random.permutation(num_rows)
    row_rand = row_rand[:math.floor(num_rows*random_ratio)]
    for row in row_rand[:-1]:
        col_rand = np.random.permutation(num_cols)
        col_rand = col_rand[:math.floor(num_cols*random_ratio)]
        X_data.iloc[row, col_rand] = np.nan
    # process the last row to ensure no column is completely missing
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

def gen_random(data, columns_observed=[], print_time=False, print_all=True, range_min=0.1, range_max=0.3, scalar_0=0.05, scalar_1=2):
    """Missing At Random (MAR)

    ### Args
    
    1. `data` - type of `Dataset`
    2. `columns_observed` - feature names in a list (to be observed from)
    3. `print_time` - print time to evaluate performance
    4. `print_all` - print all messages, including warnings
    5. `range_min` - missingness percentage bound min, default 0.1
    6. `range_max` - missingness percentage bound max, default 0.3
    7. `scalar_0` - scalar coefficient 0 (affects variance), default 0.05
    8. `scalar_1` - scalar coefficient 1 (affects baseline), default 2

    ### Returns
    Converted `Dataset` object with missing values

    ### Exception
    Will raise an exception if `data.X` is not rank 2

    Will raise an exception if `columns_observed` is empty

    Will raise an exception if generation failed in internal function `convert_single_feature`

    ------

    ### Implementation
    1. For each column `K` to convert:
        a. Draw (n_columns_observed+1) scalar values from a random standard normal distribution N(0, 1)

        b. Construct `M` by multiplying the scalar values on the observed features

        c. Use the standard logistic cumulative distribution function to convert `M` to probability `p`

        d. For each `p`, draw a value randomly from the Binomial(1, `p`) distribution

        e. For each row `i`, if `p[i]` = 1, then `K[i]` is replaced by NaN
    2. Skip protected features
    """
    if print_time:
        tt = time.process_time()
    X_data = data.X.copy()
    if len(data.protected_features) > 0:
        X_data.drop(columns=data.protected_features, inplace=True)
        X_data_protected = data.X[data.protected_features].copy()
    if len(X_data.shape) != 2:
        raise Exception("Error: gen_random only support dataset with rank of 2, but your input has rank of {0}".format(len(X_data.shape)))
    if len(columns_observed) <= 0 and print_all:
        raise Exception("Error: gen_random has no observed features selected")
    for feature in columns_observed:
        if (feature not in data.X.columns.tolist()) or (feature in data.protected_features):
            raise Exception("Error: gen_random columns_observed contain invalid feature: {}".format(feature))

    scaler = StandardScaler()
    def convert_single_feature(K_df, observed_df, range_min=0.1, range_max=0.3, scalar_0=0.05, scalar_1=2):
        """Convert a single feature using Missing At Random

        Missingness in `K_df` depends on `observed_df`
        """
        max_iter = 10
        while(max_iter > 0):
            scalars = np.random.standard_normal(observed_df.shape[1]) * scalar_0
            scalar_a = np.random.standard_normal() - scalar_1
            observed = observed_df.to_numpy()
            observed = scaler.fit_transform(observed)
            M = scalar_a + (scalars * observed).sum(axis=1)
            p = 1 / (1 + np.exp(-M))
            ratio = p.sum() / K_df.shape[0]
            if ratio > range_min and ratio < range_max:
                break
            max_iter -= 1
        if max_iter <= 0:
            return None
        p_bi = np.random.binomial(1, p).astype(np.bool)
        K_df = K_df.where(~p_bi, other=np.nan)
        return K_df

    observed_df = X_data[columns_observed].copy()
    for feature in X_data.columns.tolist():
        if feature not in columns_observed:
            new_df = convert_single_feature(X_data[feature], observed_df, range_min=range_min, range_max=range_max, scalar_0=scalar_0, scalar_1=scalar_1)
            if new_df is None:
                raise Exception("Error: gen_random failed to generate missing values, with range ({}, {}), scalar0={}, scalar1={}".format(range_min, range_max, scalar_0, scalar_1))
            else:
                X_data[feature] = new_df

    if print_all:
        print("gen_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if len(data.protected_features) > 0:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

def gen_not_random(data, print_time=False, print_all=True, range_min=0.1, range_max=0.3, scalar_0=0.05, scalar_1=2):
    """Missing Not At Random (MNAR)

    ### Args
    
    1. `data` - type of `Dataset`
    2. `print_time` - print time to evaluate performance
    3. `print_all` - print all messages, including warnings
    4. `range_min` - missingness percentage bound min, default 0.1
    5. `range_max` - missingness percentage bound max, default 0.3
    6. `scalar_0` - scalar coefficient 0 (affects variance), default 0.05
    7. `scalar_1` - scalar coefficient 1 (affects baseline), default 2

    ### Returns
    Converted `Dataset` object with missing values

    ### Exception
    Will raise an exception if `data.X` is not rank 2

    Will raise an exception if generation failed in internal function `convert_multiple_features`

    ------

    ### Implementation
    1. For the whole input data:
        a. Draw (n_columns_data+1) scalar values from a random standard normal distribution N(0, 1)

        b. Construct `M` by multiplying the scalar values on the data itself

        c. Use the standard logistic cumulative distribution function to convert `M` to probability `p`

        d. For each `p`, draw a value randomly from the Binomial(1, `p`) distribution

        e. For each row `i`, if `p[i]` = 1, then `K[i]` is replaced by NaN
    2. Skip protected features
    """
    if print_time:
        tt = time.process_time()
    X_data = data.X.copy()
    if len(data.protected_features) > 0:
        X_data.drop(columns=data.protected_features, inplace=True)
        X_data_protected = data.X[data.protected_features].copy()
    if len(X_data.shape) != 2:
        raise Exception("Error: gen_not_random only support dataset with rank of 2, but your input has rank of {0}".format(len(X_data.shape)))

    scaler = StandardScaler()
    def convert_multiple_features(K_df, range_min=0.1, range_max=0.3, scalar_0=0.05, scalar_1=2):
        """Convert multiple features using Missing Not At Random

        Missingness in `K_df` depends on itself
        """
        max_iter = 10
        while(max_iter > 0):
            scalars = np.random.standard_normal(K_df.shape) * scalar_0
            scalar_a = np.random.standard_normal() - scalar_1
            K = K_df.to_numpy()
            K = scaler.fit_transform(K)
            M = scalar_a + (scalars * K)
            p = 1 / (1 + np.exp(-M))
            ratio = p.mean(axis=0)
            if ratio.min() > range_min and ratio.max() < range_max:
                break
            max_iter -= 1
        if max_iter <= 0:
            return None
        p_bi = np.random.binomial(1, p).astype(np.bool)
        K_df = K_df.where(~p_bi, other=np.nan)
        return K_df

    X_data = convert_multiple_features(X_data, range_max=range_max, range_min=range_min, scalar_0=scalar_0, scalar_1=scalar_1)
    if X_data is None:
        raise Exception("Error: gen_not_random failed to generate missing values, with range ({}, {}), scalar0={}, scalar1={}".format(range_min, range_max, scalar_0, scalar_1))

    if print_all:
        print("gen_random: {0} NaN values have been inserted".format(X_data.isnull().sum().sum()))
    if len(data.protected_features) > 0:
        X_data = pd.concat([X_data, X_data_protected], axis=1)
    data = data.copy()
    data.X = X_data
    if print_time and print_all:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data