# This file contains functions that can train a model for prediction

import numpy as np
import pandas as pd
from utils.data import *
from utils.model_header import *  

# Method 1: K-Nearest Neighbor
def KNN(data, cv=5):
    knn = KNeighborsClassifier()
    score = cross_val_score(knn, data.X, data.y, cv=cv)
    return score.mean()

# Method 2: Stochastic Gradient Descent
def SGD(data, cv=5):
    sgd = SGDClassifier()
    score = cross_val_score(sgd, data.X, data.y, cv=cv)
    return score.mean()

# Method 3: Decision Tree
def DecisionTree(data, cv=5):
    tree = DecisionTreeClassifier()
    score = cross_val_score(tree, data.X, data.y, cv=cv)
    return score.mean()