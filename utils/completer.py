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
    data.X = data.X.fillna(value).astype(data.types)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 2
# complete missing entries using the mean of that column
def complete_by_mean_col(data, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data.X = data.X.fillna(data.X.mean()).astype(data.types)
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
    data.X = data.X.fillna(method="ffill")
    data.X = data.X.fillna(method="bfill").astype(data.types)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 4
# complete missing entries by values from most similar row
def complete_by_similar_row(data, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
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

    imputer = KNNImputer(n_neighbors=5, weights="uniform") # by default use euclidean distance
    data.X = imputer.fit_transform(data.X)

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 5
# fill with the most frequent value in that column
def complete_by_most_freq(data, print_time=False):
    if print_time:
        tt = time.process_time()
    data = data.copy()
    data.X.fillna(data.X.mode().iloc[0], inplace=True)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data

# Method 6
# multiple imputation
def complete_by_multi(data, print_time=False, num_outputs=5):
    if print_time:
        tt = time.process_time()
    data_new = []
    imputer = IterativeImputer(max_iter=10) # do not set random state here
    for _ in num_outputs:
        data_copy = data.copy()
        data_copy.X = imputer.fit_transform(data_copy.X)
        data_new.append(data_copy)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return data_new