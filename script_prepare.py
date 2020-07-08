# script for preparing necessary data for single tasks

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
from utils.data import Dataset, create_adult_dataset, create_compas_dataset, create_titanic_dataset

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC , SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SVMSMOTE

# define enum
target_metrics = {
    "acc": 0,
    "f1": 1
}

# prepare datasets for parameter searching
def prepare_datasets():
    results = {
        "adult": None,
        "compas": None,
        "titanic": None
    }
    # compas dataset
    c_data = create_compas_dataset()
    tmp_concat = pd.concat([c_data.X, pd.DataFrame(c_data.y, columns=["_TARGET_"])], axis=1)
    tmp_concat.dropna(inplace=True)
    tmp_concat.reset_index(drop=True, inplace=True)
    c_data.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
    c_data.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
    results["compas"] = c_data
    # adult dataset
    a_data = create_adult_dataset()
    results["adult"] = a_data
    # titanic dataset
    t_data = create_titanic_dataset()
    tmp_concat = pd.concat([t_data.X, pd.DataFrame(t_data.y, columns=["_TARGET_"])], axis=1)
    tmp_concat.dropna(inplace=True)
    tmp_concat.reset_index(drop=True, inplace=True)
    t_data.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
    t_data.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
    results["titanic"] = t_data
    return results

# run parameter searching and save data
def param_search(datasets, metrics, json_file=None):
    params = {
        "KNN": {
            "n_neighbors": [2, 5, 10, 50, 100, 500],
            "leaf_size": [5, 10, 30, 50, 100],
        },
        "LinearSVC": {
            "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [1000, 5000, 10000],
        },
        "SVC": {
            "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [1000, 5000, 10000, -1],
        },
        "Forest": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 50, 100, 200],
            "min_samples_leaf": [1, 5, 10, 50],
        },
        "LogReg": {
            "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "C": [1e-2, 1e-1, 1, 1e1, 1e2],
            "max_iter": [100, 500, 1000, 5000],
        },
        "Tree": {
            "max_depth": [None, 10, 50, 100, 200],
            "max_leaf_nodes": [None, 10, 100, 1000],
            "min_samples_leaf": [1, 5, 10, 50],
        },
        "MLP": {
            "hidden_layer_sizes": [(10,), (100,), (1000,)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
            "max_iter": [200, 500, 1000],
        },
    }
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "LinearSVC": LinearSVC(dual=False),
        "SVC": SVC(),
        "Forest": RandomForestClassifier(),
        "LogReg": LogisticRegression(),
        "Tree": DecisionTreeClassifier(),
        "MLP": MLPClassifier(),
    }
    results = {}
    smote = SVMSMOTE(random_state=22)
    if metrics == target_metrics["acc"]: scoring = "accuracy"
    elif metrics == target_metrics["f1"]: scoring = "f1"
    else: raise ValueError("metrics is not the correct value")
    print("Target metric: {}".format(scoring))
    for d_name, d_value in datasets.items():
        results[d_name] = {}
        print("Now running on {} dataset".format(d_name))
        for clf in classifiers.keys():
            model = classifiers[clf]
            X = d_value.X.drop(columns=d_value.protected).copy().to_numpy()
            y = d_value.y.copy()
            X_res, y_res = smote.fit_resample(X, y)
            print("Parameter searching for {}".format(model.__class__.__name__))
            search = GridSearchCV(model, params[clf], n_jobs=-1, cv=10, scoring=scoring)
            start_time = time.time()
            search.fit(X_res, y_res)
            print("Search finished in {:.2f}min, best score = {}".format((time.time() - start_time) / 60, search.best_score_))
            results[d_name][clf] = search.best_params_
    if json_file:
        with open(json_file, "w") as outFile:
            json.dump(results, outFile)
    return results

if __name__=="__main__":
    datasets = prepare_datasets()
    param_search(datasets, metrics=target_metrics["acc"], json_file="params_acc.json")
    param_search(datasets, metrics=target_metrics["f1"], json_file="params_f1.json")