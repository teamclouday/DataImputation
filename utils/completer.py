# This file contains functions that complete dataset with missing entries

import sys
import pandas as pd
import numpy as np
from utils.data import *

# Method 1
# fill all missing entries with one value
def complete_by_value(data, value=0):
    data = data.copy()
    data.X.fillna(value, inplace=True)
    return data

# Method 2
# complete missing entries using the mean of that column
def complete_by_mean_col(data):
    data = data.copy()
    names = data.X.columns
    values = {}
    for name in names:
        values[name] = data.X[name].mean()
    data.X.fillna(values, inplace=True)
    return data

# Method 3
# complete missing entries using value from previous row
def complete_by_nearby_row(data):
    data = data.copy()
    data.X.fillna(method="ffill", inplace=True)
    return data

# Method 4
# train a regression model and predict the target missing value
def complete_by_model(data):
    pass
