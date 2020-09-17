# This file contains functions that complete dataset with missing entries

import pandas as pd
import numpy as np
from utils.data import *

# Method 1
def complete_by_value(data, value=0, print_time=False):
    """
    Replace NaN with `value` passed as argument
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected_features].copy()
    data_unprotected = data.X.drop(columns=data.protected_features).copy()
    data_unprotected = data_unprotected.fillna(value).astype(data.types.drop(data.protected_features))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 2
def complete_by_mean_col(data, print_time=False):
    """
    Fill missing entries using the mean of that column
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected_features].copy()
    data_unprotected = data.X.drop(columns=data.protected_features).copy()
    data_unprotected = data_unprotected.fillna(data_unprotected.mean()).astype(data.types.drop(data.protected_features))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 2 version 2
def complete_by_mean_col_v2(data, print_time=False, target_feature=None):
    """
    Fill missing entries using the mean of the column from opposite group (defined by `target_feature`)

    For example, entries for `race`="African-American" will be imputed from rows whose `race` is not "African-American"
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()

    if target_feature:
        assert target_feature in data.protected_features
        target_unique_values = data.X[target_feature].unique().tolist()
        assert len(target_unique_values) > 0
        if len(target_unique_values) < 2:
            print("Warning: complete_by_mean_col_v2: only one unique value found for target feature")
            return complete_by_mean_col(data, print_time=print_time)
        imputed_parts = []
        for value in target_unique_values:
            data_train = data.X[data.X[target_feature] != value].drop(columns=data.protected_features).copy()
            data_protected = data.X[data.X[target_feature] == value][data.protected_features].copy()
            data_unprotected = data.X[data.X[target_feature] == value].drop(columns=data.protected_features).copy()
            data_unprotected = data_unprotected.fillna(data_train.mean()).astype(data.types.drop(data.protected_features))
            imputed_parts.append(pd.concat([data_unprotected, data_protected], axis=1))
        data_X = imputed_parts[0]
        idx = 1
        while idx < len(imputed_parts):
            data_X = pd.concat([data_X, imputed_parts[idx]], axis=0)
            idx += 1
        assert data_X.shape == data.X.shape
        data.X = data_X.sort_index()
    else:
        print("Warning: You're using V2 mean imputation, but didn't set a value for target_feature. Will perform V1.")
        return complete_by_mean_col(data, print_time=print_time)

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 3
def complete_by_nearby_row(data, print_time=False):
    """
    Fill the missing entries by nearby values
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected_features].copy()
    data_unprotected = data.X.drop(columns=data.protected_features).copy()
    data_unprotected = data_unprotected.fillna(method="ffill")
    data_unprotected = data_unprotected.fillna(method="bfill").astype(data.types.drop(data.protected_features))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 4
def complete_by_similar_row(data, print_time=False, K=5):
    """
    Fill the missing entries by values from most similar rows, found by KNN
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected_features].copy()
    data_unprotected = data.X.drop(columns=data.protected_features).copy()

    imputer = KNNImputer(n_neighbors=K, weights="uniform") # by default use euclidean distance
    data_unprotected = pd.DataFrame(imputer.fit_transform(data_unprotected), columns=data_unprotected.columns).astype(data.types.drop(data.protected_features))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 4 version 2
def complete_by_similar_row_v2(data, print_time=False, K=5, target_feature=None):
    """
    Fill the missing entries by values from most similar rows, found by KNN

    KNN is fit on opposite group data, where opposite group is defined by `target_feature`

    For example, entries for `race`="African-American" will be imputed from rows whose `race` is not "African-American"
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()

    if target_feature:
        assert target_feature in data.protected_features
        target_unique_values = data.X[target_feature].unique().tolist()
        assert len(target_unique_values) > 0
        if len(target_unique_values) < 2:
            print("Warning: complete_by_similar_row_v2: only one unique value found for target feature")
            return complete_by_similar_row(data, print_time=print_time, K=K)
        imputed_parts = []
        for value in target_unique_values:
            imputer = KNNImputer(n_neighbors=K, weights="uniform")
            data_train = data.X[data.X[target_feature] != value].drop(columns=data.protected_features).copy()
            imputer.fit(data_train)
            data_protected = data.X[data.X[target_feature] == value][data.protected_features].copy()
            data_unprotected = data.X[data.X[target_feature] == value].drop(columns=data.protected_features).copy()
            data_unprotected = pd.DataFrame(imputer.transform(data_unprotected), columns=data_unprotected.columns, index=data_unprotected.index).astype(data.types.drop(data.protected_features))
            imputed_parts.append(pd.concat([data_unprotected, data_protected], axis=1))
        data_X = imputed_parts[0]
        idx = 1
        while idx < len(imputed_parts):
            data_X = pd.concat([data_X, imputed_parts[idx]], axis=0)
            idx += 1
        assert data_X.shape == data.X.shape
        data.X = data_X.sort_index()
    else:
        print("Warning: You're using V2 similar imputation, but didn't set a value for target_feature. Will perform V1.")
        return complete_by_similar_row(data, print_time=print_time, K=K)

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 5
def complete_by_most_freq(data, print_time=False):
    """
    Fill the missing entries by the most frequent value from that column
    """
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected_features].copy()
    data_unprotected = data.X.drop(columns=data.protected_features).copy()
    data_unprotected.fillna(data_unprotected.mode().iloc[0], inplace=True)
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 6
def complete_by_multi(data, print_time=False, num_outputs=10, verbose=0):
    """
    Fill the missing entries by running multiple imputation (MICE)

    Return a list of `Dataset` objects instead of single object
    """
    if print_time:
        tt = time.process_time()
    data_new = []
    imputer = IterativeImputer(max_iter=1, sample_posterior=True, verbose=verbose)
    for _ in range(num_outputs):
        data_copy = data.copy()
        data_protected = data_copy.X[data_copy.protected_features].copy()
        data_unprotected = data_copy.X.drop(columns=data_copy.protected_features).copy()
        data_unprotected = pd.DataFrame(imputer.fit_transform(data_unprotected), columns=data_unprotected.columns).astype(data.types.drop(data.protected_features))
        data_copy.X = pd.concat([data_unprotected, data_protected], axis=1)
        data_new.append(data_copy)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data_new

# Method 6 version 2
def complete_by_multi_v2(data, print_time=False, num_outputs=10, target_feature=None, verbose=0):
    """
    Fill the missing entries by running multiple imputation (MICE)

    Return a list of `Dataset` objects instead of single object

    Imputer learns from opposite group data, where opposite group is defined by `target_feature`, and transforms current group data

    For example, entries for `race`="African-American" will be imputed from rows whose `race` is not "African-American"
    """
    if print_time:
        tt = time.process_time()

    data_new = []

    if target_feature:
        assert target_feature in data.protected_features
        target_unique_values = data.X[target_feature].unique().tolist()
        assert len(target_unique_values) > 0
        if len(target_unique_values) < 2:
            print("Warning: complete_by_multi_v2: only one unique value found for target feature")
            return complete_by_multi(data, print_time=print_time, num_outputs=num_outputs)
        imputers = [IterativeImputer(max_iter=1, sample_posterior=True, verbose=verbose) for _ in range(len(target_unique_values))]
        for _ in range(num_outputs):
            data_copy = data.copy()
            imputed_parts = []
            for i in range(len(target_unique_values)):
                value = target_unique_values[i]
                data_train = data_copy.X[data_copy.X[target_feature] != value].drop(columns=data_copy.protected_features).copy()
                imputers[i].fit(data_train)
                data_protected = data_copy.X[data_copy.X[target_feature] == value][data_copy.protected_features].copy()
                data_unprotected = data_copy.X[data_copy.X[target_feature] == value].drop(columns=data_copy.protected_features).copy()
                data_unprotected = pd.DataFrame(imputers[i].transform(data_unprotected), columns=data_unprotected.columns, index=data_unprotected.index).astype(data.types.drop(data.protected_features))
                imputed_parts.append(pd.concat([data_unprotected, data_protected], axis=1))
            data_X = imputed_parts[0]
            idx = 1
            while idx < len(imputed_parts):
                data_X = pd.concat([data_X, imputed_parts[idx]], axis=0)
                idx += 1
            assert data_X.shape == data_copy.X.shape
            data_copy.X = data_X.sort_index()
            data_new.append(data_copy)
    else:
        print("Warning: You're using V2 multiple imputation, but didn't set a value for target_feature. Will perform V1.")
        return complete_by_multi(data, print_time=print_time, num_outputs=num_outputs)

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data_new