# This file contains functions that can train a model for prediction

import numpy as np
import pandas as pd
from utils.data import *
from utils.model_header import *

# Method 1: K-Nearest Neighbor
def KNN(data, cv, print_time=False, grid_search=False, n_jobs=1, return_model=False):
    if print_time:
        tt = time.process_time()
    knn = KNeighborsClassifier()
    if grid_search:
        params = {
            "n_neighbors": [2, 5, 8],
            "p": [1, 2],
            "algorithm": ["ball_tree", "kd_tree", "brute"]
        }
        grid = GridSearchCV(knn, param_grid=params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
        grid.fit(data.X, data.y)
        score = grid.best_score_
    else:
        scores = cross_validate(knn, data.X, data.y, cv=cv, scoring="accuracy", n_jobs=n_jobs, return_estimator=True)
        score = sum(scores["test_score"]) / len(scores["test_score"])
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    if return_model:
        if grid_search:
            return [score, grid.best_estimator_]
        else:
            return [score, scores["estimator"]]
    return score

# Method 2: Stochastic Gradient Descent
def SGD(data, cv, print_time=False, grid_search=False, n_jobs=1, return_model=False):
    if print_time:
        tt = time.process_time()
    sgd = SGDClassifier()
    if grid_search:
        params = {
            "alpha": [0.00005, 0.0001, 0.001, 0.01],
            "max_iter": [500, 1000, 4000],
            "learning_rate": ["optimal", "invscaling", "adaptive"],
            "eta0": [0.0001, 0.001, 0.01, 0.1]
        }
        grid = GridSearchCV(sgd, param_grid=params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
        grid.fit(data.X, data.y)
        score = grid.best_score_
    else:
        scores = cross_validate(sgd, data.X, data.y, cv=cv, scoring="accuracy", n_jobs=n_jobs, return_estimator=True)
        score = sum(scores["test_score"]) / len(scores["test_score"])
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    if return_model:
        if grid_search:
            return [score, grid.best_estimator_]
        else:
            return [score, scores["estimator"]]
    return score

# Method 3: Decision Tree
def DecisionTree(data, cv, print_time=False, grid_search=False, n_jobs=1, return_model=False):
    if print_time:
        tt = time.process_time()
    tree = DecisionTreeClassifier()
    if grid_search:
        params = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 100],
            "min_samples_split": [2, 5, 10],
            "max_leaf_nodes": [None, 100, 500, 1000]
        }
        grid = GridSearchCV(tree, param_grid=params, scoring="accuracy", n_jobs=n_jobs, cv=cv)
        grid.fit(data.X, data.y)
        score = grid.best_score_
    else:
        scores = cross_validate(tree, data.X, data.y, cv=cv, scoring="accuracy", n_jobs=n_jobs, return_estimator=True)
        score = sum(scores["test_score"]) / len(scores["test_score"])
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    if return_model:
        if grid_search:
            return [score, grid.best_estimator_]
        else:
            return [score, scores["estimator"]]
    return score

# Method 4: SVM
def SVM(data, cv, print_time=False, grid_search=False, n_jobs=1, return_model=False):
    if print_time:
        tt = time.process_time()
    svm = SVC()
    if grid_search:
        params = {
            "C": [0.01, 0.1, 1.0, 5.0],
            "tol": [1e-4, 1e-3, 1e-2],
            "kernel": ["poly", "rbf", "sigmoid"]
        }
        grid = GridSearchCV(svm, param_grid=params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
        grid.fit(data.X, data.y)
        score = grid.best_score_
    else:
        scores = cross_validate(svm, data.X, data.y, cv=cv, scoring="accuracy", n_jobs=n_jobs, return_estimator=True)
        score = sum(scores["test_score"]) / len(scores["test_score"])
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    if return_model:
        if grid_search:
            return [score, grid.best_estimator_]
        else:
            return [score, scores["estimator"]]
    return score

# Method 5: Random Forest
def Forest(data, cv, print_time=False, grid_search=False, n_jobs=1, return_model=False):
    if print_time:
        tt = time.process_time()
    forest = RandomForestClassifier()
    if grid_search:
        params = {
            "n_estimators": [50, 100, 200, 800],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 100],
            "min_samples_split": [2, 5, 10],
            "max_leaf_nodes": [None, 100, 500, 1000]
        }
        grid = GridSearchCV(forest, param_grid=params, scoring="accuracy", cv=cv, n_jobs=n_jobs)
        grid.fit(data.X, data.y)
        score = grid.best_score_
    else:
        scores = cross_validate(forest, data.X, data.y, cv=cv, scoring="accuracy", n_jobs=n_jobs, return_estimator=True)
        score = sum(scores["test_score"]) / len(scores["test_score"])
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    if return_model:
        if grid_search:
            return [score, grid.best_estimator_]
        else:
            return [score, scores["estimator"]]
    return score