# This file contains functions that can train a model for prediction

import numpy as np
import pandas as pd
from utils.data import *
from utils.model_header import *  

# Method 1: K-Nearest Neighbor
def KNN(data, cv):
    knn = KNeighborsClassifier()
    score = cross_val_score(knn, data.X, data.y, cv=cv)
    return score.mean()

# Method 2: Stochastic Gradient Descent
def SGD(data, cv):
    sgd = SGDClassifier()
    score = cross_val_score(sgd, data.X, data.y, cv=cv)
    return score.mean()

# Method 3: Decision Tree
def DecisionTree(data, cv):
    tree = DecisionTreeClassifier()
    score = cross_val_score(tree, data.X, data.y, cv=cv)
    return score.mean()

# Method 4: SVM
def SVM(data, cv):
    svm = SVC()
    score = cross_val_score(svm, data.X, data.y, cv=cv)
    return score.mean()

# Method 5: Random Forest
def Forest(data, cv):
    forest = RandomForestClassifier()
    score = cross_val_score(forest, data.X, data.y, cv=cv)
    return score.mean()