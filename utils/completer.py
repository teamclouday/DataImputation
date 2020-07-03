# This file contains functions that complete dataset with missing entries

import pandas as pd
import numpy as np
from utils.data import *

# Method 1
# fill all missing entries with one value
def complete_by_value(data, value=0, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected].copy()
    data_unprotected = data.X.drop(columns=data.protected).copy()
    data_unprotected = data_unprotected.fillna(value).astype(data.types.drop(data.protected))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 2
# complete missing entries using the mean of that column
def complete_by_mean_col(data, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected].copy()
    data_unprotected = data.X.drop(columns=data.protected).copy()
    data_unprotected = data_unprotected.fillna(data_unprotected.mean()).astype(data.types.drop(data.protected))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 2 version 2
# same idea as similar imputation version 2
def complete_by_mean_col_v2(data, print_time=False, target_feature=None):
    if print_time:
        tt = time.process_time()
    data = data.copy()

    if target_feature:
        assert target_feature in data.protected
        target_unique_values = data.X[target_feature].unique().tolist()
        assert len(target_unique_values) > 0
        if len(target_unique_values) < 2:
            print("Warning: complete_by_mean_col_v2: only one unique value found for target feature")
            return complete_by_mean_col(data, print_time=print_time)
        imputed_parts = []
        for value in target_unique_values:
            data_train = data.X[data.X[target_feature] != value].drop(columns=data.protected).copy()
            data_protected = data.X[data.X[target_feature] == value][data.protected].copy()
            data_unprotected = data.X[data.X[target_feature] == value].drop(columns=data.protected).copy()
            data_unprotected = data_unprotected.fillna(data_train.mean()).astype(data.types.drop(data.protected))
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
# complete missing entries using value from previous row
# if previous rows are all NaN, then fill with value from next row
def complete_by_nearby_row(data, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected].copy()
    data_unprotected = data.X.drop(columns=data.protected).copy()
    data_unprotected = data_unprotected.fillna(method="ffill")
    data_unprotected = data_unprotected.fillna(method="bfill").astype(data.types.drop(data.protected))
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 4
# complete missing entries by values from most similar row
def complete_by_similar_row(data, print_time=False, K=5):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected].copy()
    data_unprotected = data.X.drop(columns=data.protected).copy()
    # isnull_matrix = data.X.isnull()
    # # compute similarity matrix
    # matrix = np.zeros((len(data.X), len(data.X)))
    # data_X = data.X.to_numpy()
    # for i in range(len(data_X)-1):
    #     for j in range(i+1, len(data_X)):
    #         print(i, j)
    #         sim = 0
    #         count = 0
    #         # for col_name in data.X.columns:
    #         #     if not isnull_matrix[col_name][i] and not isnull_matrix[col_name][j]:
    #         #         sim += (data.X[col_name][i] - data.X[col_name][j])**2 # (x-y)^2
    #         #         count += 1
    #         sim = sim ** 0.5
    #         sim /= count
    #         matrix[i][j] = sim
    #         matrix[j][i] = sim
    # # fill in nan values
    # for col_name in data.X.columns:
    #     for i in range(len(data.X)):
    #         if isnull_matrix[col_name][i]:
    #             possible_rows = {a:x for (a,x) in enumerate(matrix[i]) if x > 0 and not isnull_matrix[col_name][a]}
    #             possible_rows = sorted(possible_rows.items(), key=lambda x: x[1])
    #             if len(possible_rows) <= 0: possible_rows = [(0,0)]
    #             new_data.X[col_name][i] = data.X[col_name][possible_rows[0][0]]

    imputer = KNNImputer(n_neighbors=K, weights="uniform") # by default use euclidean distance
    data_unprotected = pd.DataFrame(imputer.fit_transform(data_unprotected), columns=data_unprotected.columns)
    data.X = pd.concat([data_unprotected, data_protected], axis=1)

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 4 version 2
# complete missing entries by values from most similar row
# if target_feature (protected feature) is provided
# the imputation will be implemented by groups
# for example, entries for race="African-American" will be imputed from rows whose race is not this
# else if target_feature not provided
# it will perform the version 1 similar imputation
def complete_by_similar_row_v2(data, print_time=False, K=5, target_feature=None):
    if print_time:
        tt = time.process_time()
    data = data.copy()

    if target_feature:
        assert target_feature in data.protected
        target_unique_values = data.X[target_feature].unique().tolist()
        assert len(target_unique_values) > 0
        if len(target_unique_values) < 2:
            print("Warning: complete_by_similar_row_v2: only one unique value found for target feature")
            return complete_by_similar_row(data, print_time=print_time, K=K)
        imputed_parts = []
        for value in target_unique_values:
            imputer = KNNImputer(n_neighbors=K, weights="uniform")
            data_train = data.X[data.X[target_feature] != value].drop(columns=data.protected).copy()
            imputer.fit(data_train)
            data_protected = data.X[data.X[target_feature] == value][data.protected].copy()
            data_unprotected = data.X[data.X[target_feature] == value].drop(columns=data.protected).copy()
            data_unprotected = pd.DataFrame(imputer.transform(data_unprotected), columns=data_unprotected.columns, index=data_unprotected.index)
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
# fill with the most frequent value in that column
def complete_by_most_freq(data, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data_protected = data.X[data.protected].copy()
    data_unprotected = data.X.drop(columns=data.protected).copy()
    data_unprotected.fillna(data_unprotected.mode().iloc[0], inplace=True)
    data.X = pd.concat([data_unprotected, data_protected], axis=1)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 6
# multiple imputation
def complete_by_multi(data, print_time=False, num_outputs=5):
    if print_time:
        tt = time.process_time()
    data_new = []
    imputer = IterativeImputer(max_iter=50, sample_posterior=True)
    for i in range(num_outputs):
        data_copy = data.copy()
        data_protected = data_copy.X[data_copy.protected].copy()
        data_unprotected = data_copy.X.drop(columns=data_copy.protected).copy()
        data_unprotected = pd.DataFrame(imputer.fit_transform(data_unprotected), columns=data_unprotected.columns)
        data_copy.X = pd.concat([data_unprotected, data_protected], axis=1)
        data_new.append(data_copy)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data_new

# Method 6 version 2
# multiple imputation
# same idea as similar imputation version 2
# apply imputation based on the data from opposite groups
def complete_by_multi_v2(data, print_time=False, num_outputs=5, target_feature=None):
    if print_time:
        tt = time.process_time()

    data_new = []

    if target_feature:
        assert target_feature in data.protected
        target_unique_values = data.X[target_feature].unique().tolist()
        assert len(target_unique_values) > 0
        if len(target_unique_values) < 2:
            print("Warning: complete_by_multi_v2: only one unique value found for target feature")
            return complete_by_multi(data, print_time=print_time, num_outputs=num_outputs)
        imputer = IterativeImputer(max_iter=50, sample_posterior=True)
        for i in range(num_outputs):
            data_copy = data.copy()
            imputed_parts = []
            for value in target_unique_values:
                data_train = data_copy.X[data_copy.X[target_feature] != value].drop(columns=data_copy.protected).copy()
                imputer.fit(data_train)
                data_protected = data_copy.X[data_copy.X[target_feature] == value][data_copy.protected].copy()
                data_unprotected = data_copy.X[data_copy.X[target_feature] == value].drop(columns=data_copy.protected).copy()
                data_unprotected = pd.DataFrame(imputer.transform(data_unprotected), columns=data_unprotected.columns, index=data_unprotected.index)
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