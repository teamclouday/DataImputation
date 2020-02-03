# This file contains functions that can train a model for prediction

import numpy as np
import pandas as pd
from utils.data import *
from utils.model_header import *

# helper function to split the dataset
def _Help_Split(data):
    pass

# Method 1: K-Nearest Neighbor
def KNN(data, train_test_split=0.3, random_seed=None, **kwargs):
    data = data.copy()
    knn = KNeighborsClassifier(kwargs)