# script for each single task for condor-submit

# this script is currently for computing random ratios on compas analysis

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from utils.data import create_compas_dataset, Dataset
from utils.generator import gen_complete_random
from utils.completer import complete_by_mean_col, complete_by_mean_col_v2, complete_by_multi, complete_by_multi_v2, complete_by_similar_row, complete_by_similar_row_v2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC , SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SVMSMOTE

#define functions

# [TN_AA, FP_AA, FN_AA, TP_AA, TN_C, FP_C, FN_C, TP_C]
def bias1(data):
    # input should be data from compute_confusion_matrix
    # bias 1 = |(FPR_AA/FNR_AA) - (FPR_C/FNR_C)|
    FPR_AA = data[1] / (data[1] + data[0])
    FNR_AA = data[2] / (data[2] + data[3])
    FPR_C  = data[5] / (data[5] + data[4])
    FNR_C  = data[6] / (data[6] + data[7])
    if FNR_AA == 0 or FNR_C == 0: return -1 # mark error situation
    bias = (FPR_AA / FNR_AA) - (FPR_C / FNR_C)
    return abs(bias)

def bias2(data):
    # input should be data from compute_confusion_matrix
    # bias 2 = |(FPR_AA/FPR_C) - (FNR_AA/FNR_C)|
    FPR_AA = data[1] / (data[1] + data[0])
    FNR_AA = data[2] / (data[2] + data[3])
    FPR_C  = data[5] / (data[5] + data[4])
    FNR_C  = data[6] / (data[6] + data[7])
    if FNR_C == 0 or FPR_C == 0: return -1 # mark error situation
    bias = (FPR_AA / FPR_C) - (FNR_AA / FNR_C)
    return abs(bias)

def acc(data):
    # input should be data from compute_confusion_matrix
    # acc = (TP + TN) / (TP + TN + FP + FN)
    TP = data[3] + data[7]
    TN = data[0] + data[4]
    accuracy = (TP + TN) / sum(data)
    return accuracy

def helper_freq(array):
    """simple helper function to return the most frequent number in an array"""
    count = np.bincount(array)
    return array[np.argmax(count)]

def compute_confusion_matrix(X_train, y_train, X_test, y_test, clf, protected_features, multi=False):
    # X are pandas dataframe
    # y are numpy array
    # clf is a sklearn classifier
    # protected_features is list
    smote = SVMSMOTE(random_state=22)
    if not multi:
        X_train = X_train.drop(columns=protected_features).copy().to_numpy()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        clf.fit(X_train_res, y_train_res)
        X_test_AA = X_test[X_test["race"] == "African-American"].drop(columns=protected_features).to_numpy()
        X_test_C = X_test[X_test["race"] == "Caucasian"].drop(columns=protected_features).to_numpy()
        y_test_AA = y_test[X_test[X_test["race"] == "African-American"].index.tolist()]
        y_test_C = y_test[X_test[X_test["race"] == "Caucasian"].index.tolist()]
        matrix_AA = confusion_matrix(y_test_AA, clf.predict(X_test_AA))
        matrix_C = confusion_matrix(y_test_C, clf.predict(X_test_C))
    else:
        prediction_AA = []
        prediction_C = []
        X_test_first = X_test[0]
        y_test_AA = y_test[X_test_first[X_test_first["race"] == "African-American"].index.tolist()]
        y_test_C = y_test[X_test_first[X_test_first["race"] == "Caucasian"].index.tolist()]
        for X_train_m in X_train:
            X_train_m = X_train_m.drop(columns=protected_features).copy().to_numpy()
            X_train_res, y_train_res = smote.fit_resample(X_train_m, y_train)
            clf.fit(X_train_res, y_train_res)
            for X_test_m in X_test:
                X_test_AA = X_test_m[X_test_m["race"] == "African-American"].drop(columns=protected_features).to_numpy()
                X_test_C = X_test_m[X_test_m["race"] == "Caucasian"].drop(columns=protected_features).to_numpy()
                prediction_AA.append(clf.predict(X_test_AA))
                prediction_C.append(clf.predict(X_test_C))
        # compute final predictions by voting
        prediction_AA = np.apply_along_axis(helper_freq, 0, np.array(prediction_AA))
        prediction_C = np.apply_along_axis(helper_freq, 0, np.array(prediction_C))
        matrix_AA = confusion_matrix(y_test_AA, prediction_AA)
        matrix_C = confusion_matrix(y_test_C, prediction_C)
    # [TN_AA, FP_AA, FN_AA, TP_AA, TN_C, FP_C, FN_C, TP_C]
    result = matrix_AA.ravel().tolist() + matrix_C.ravel().tolist()
    return result

def test_imputation(X, y, protected_features, completer_func=None, multi=False):
    # X is pandas dataframe
    # y is numpy array,
    # protected_features is list
    # completer func is the imputation function
    clfs = { # define all the classifiers with best parameters
        "KNN": KNeighborsClassifier(n_neighbors=2, leaf_size=5),
        "LinearSVC": LinearSVC(dual=False, tol=0.001, C=0.1, max_iter=1000),
        "SVC": SVC(tol=0.0001, C=10, max_iter=-1),
        "Forest": RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_leaf=5),
        "LogReg": LogisticRegression(tol=1e-5, C=10, max_iter=100),
        "Tree": DecisionTreeClassifier(max_depth=10, max_leaf_nodes=100, min_samples_leaf=1),
        "MLP": MLPClassifier(alpha=0.0001, learning_rate_init=0.01, max_iter=500),
    }
    acc_cv = { # save each accuracy output cv
        "KNN": [],
        "LinearSVC": [],
        "SVC": [],
        "Forest": [],
        "LogReg": [],
        "Tree": [],
        "MLP": [],
    }
    bias1_cv = { # save each bias 1 outputs cv
        "KNN": [],
        "LinearSVC": [],
        "SVC": [],
        "Forest": [],
        "LogReg": [],
        "Tree": [],
        "MLP": [],
    }
    bias2_cv = { # save each bias 2 outputs cv
        "KNN": [],
        "LinearSVC": [],
        "SVC": [],
        "Forest": [],
        "LogReg": [],
        "Tree": [],
        "MLP": [],
    }
    kf = KFold(n_splits=10, shuffle=True)
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_test = X_test.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        if completer_func:
        # do imputations on training set and test set individually
            data_incomplete = Dataset("tmp", X_train, y_train, auto_convert=False, protected_features=protected_features)
            data_complete = completer_func(data_incomplete)
            X_train = [m.X.copy() for m in data_complete] if multi else data_complete.X.copy()
            y_train = data_complete[0].y.copy() if multi else data_complete.y.copy()
            data_incomplete = Dataset("tmp", X_test, y_test, auto_convert=False, protected_features=protected_features)
            data_complete = completer_func(data_incomplete)
            X_test = [m.X.copy() for m in data_complete] if multi else data_complete.X.copy()
            y_test = data_complete[0].y.copy() if multi else data_complete.y.copy()
        # get result for each classifier
        for clf_name, clf in clfs.items():
            result = compute_confusion_matrix(X_train, y_train, X_test, y_test, clf, protected_features, multi=multi)
            acc_cv[clf_name].append(acc(result))
            bias1_cv[clf_name].append(bias1(result))
            bias2_cv[clf_name].append(bias2(result))
        fold += 1
    return (acc_cv, bias1_cv, bias2_cv)

# prepare data

data = create_compas_dataset()
data_compas_complete = data.copy()
tmp_concat = pd.concat([data_compas_complete.X, pd.DataFrame(data_compas_complete.y, columns=["_TARGET_"])], axis=1)
tmp_concat.dropna(inplace=True)
tmp_concat.reset_index(drop=True, inplace=True)
data_compas_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
data_compas_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()

random_ratios = np.linspace(0.0, 1.0, num=20, endpoint=False)

if __name__ == "__main__":
    if not os.path.exists("condor_outputs"):
        os.makedirs("condor_outputs")
    if len(sys.argv) < 2:
        raise Exception("should have argument for task id")

    final_result = {}
    start_time = time.time()

    # run mean version 1
    print("Now running mean imputation version 1")
    final_result["mean_v1"] = []
    for ratio in random_ratios:
        print("Current Ratio: {:.2f}".format(ratio))
        data_sim = gen_complete_random(data_compas_complete, random_ratio=ratio, print_all=False)
        final_result["mean_v1"].append(test_imputation(data_sim.X.copy(), data_sim.y.copy(), data_sim.protected, complete_by_mean_col, multi=False))
    print("Task finished in {:.3f}s\n".format(time.time() - start_time))
    start_time = time.time()

    # run mean version 2
    print("Now running mean imputation version 2")
    final_result["mean_v2"] = []
    for ratio in random_ratios:
        print("Current Ratio: {:.2f}".format(ratio))
        data_sim = gen_complete_random(data_compas_complete, random_ratio=ratio, print_all=False)
        final_result["mean_v2"].append(test_imputation(data_sim.X.copy(), data_sim.y.copy(), data_sim.protected, partial(complete_by_mean_col_v2, target_feature="race"), multi=False))
    print("Task finished in {:.3f}s\n".format(time.time() - start_time))
    start_time = time.time()

    # run similar version 1
    print("Now running similar imputation version 1")
    final_result["similar_v1"] = []
    for ratio in random_ratios:
        print("Current Ratio: {:.2f}".format(ratio))
        data_sim = gen_complete_random(data_compas_complete, random_ratio=ratio, print_all=False)
        final_result["similar_v1"].append(test_imputation(data_sim.X.copy(), data_sim.y.copy(), data_sim.protected, complete_by_similar_row, multi=False))
    print("Task finished in {:.3f}s\n".format(time.time() - start_time))
    start_time = time.time()

    # run similar version 2
    print("Now running similar imputation version 2")
    final_result["similar_v2"] = []
    for ratio in random_ratios:
        print("Current Ratio: {:.2f}".format(ratio))
        data_sim = gen_complete_random(data_compas_complete, random_ratio=ratio, print_all=False)
        final_result["similar_v2"].append(test_imputation(data_sim.X.copy(), data_sim.y.copy(), data_sim.protected, partial(complete_by_similar_row_v2, target_feature="race"), multi=False))
    print("Task finished in {:.3f}s\n".format(time.time() - start_time))
    start_time = time.time()

    # run multi version 1
    print("Now running multiple imputation version 1")
    final_result["multi_v1"] = []
    for ratio in random_ratios:
        print("Current Ratio: {:.2f}".format(ratio))
        data_sim = gen_complete_random(data_compas_complete, random_ratio=ratio, print_all=False)
        final_result["multi_v1"].append(test_imputation(data_sim.X.copy(), data_sim.y.copy(), data_sim.protected, complete_by_multi, multi=True))
    print("Task finished in {:.3f}s\n".format(time.time() - start_time))
    start_time = time.time()

    # run multi version 2
    print("Now running multiple imputation version 2")
    final_result["multi_v2"] = []
    for ratio in random_ratios:
        print("Current Ratio: {:.2f}".format(ratio))
        data_sim = gen_complete_random(data_compas_complete, random_ratio=ratio, print_all=False)
        final_result["multi_v2"].append(test_imputation(data_sim.X.copy(), data_sim.y.copy(), data_sim.protected, partial(complete_by_multi_v2, target_feature="race"), multi=True))
    print("Task finished in {:.3f}s\n".format(time.time() - start_time))
    start_time = time.time()

    # save outputs
    with open(os.path.join("condor_outputs", "output_{:0>4}.pkl".format(sys.argv[1])), "wb") as outFile:
        pickle.dump(final_result, outFile)