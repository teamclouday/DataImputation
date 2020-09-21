# Extended version of script single task
# For experiments on
# Missing At Random
# Missing Not At Random

import os
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from utils.data import create_compas_dataset, create_adult_dataset, create_titanic_dataset, create_communities_dataset, create_german_dataset, create_bank_dataset, Dataset
from utils.generator import gen_random, gen_not_random
from utils.completer import complete_by_mean_col, complete_by_mean_col_v2, complete_by_multi, complete_by_multi_v2, complete_by_similar_row, complete_by_similar_row_v2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC , SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE

RUN_MEAN_V1     = True
RUN_MEAN_V2     = True
RUN_SIMILAR_V1  = True
RUN_SIMILAR_V2  = True
RUN_MULTI_V1    = True
RUN_MULTI_V2    = True

NAME_DATA = {
    "adult": 1,
    "compas": 2,
    "titanic": 3,
    "german": 4,
    "communities": 5,
    "bank": 6,
}
NAME_MISSING = {
    "MAR": 1,
    "MNAR": 2,
}
PARAMS = None
PARAMS_DATA = None
N_SPLITS = 10

def actualAcc(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

def helper_freq(array):
    """simple helper function to return the most frequent number in an array"""
    count = np.bincount(array)
    return array[np.argmax(count)]

def compute_confusion_matrix(X_train, y_train, X_test, y_test, clf, protected_features, multi=False):
    # X are pandas dataframe
    # y are numpy array
    # clf is a sklearn classifier
    # protected_features is list
    global PARAMS_DATA
    smote = SMOTE()
    scaler = StandardScaler()
    result_acc = -1
    if not multi:
        X_train = X_train.drop(columns=protected_features).copy().to_numpy()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        X_train_scaled = scaler.fit_transform(X_train_res)
        clf.fit(X_train_scaled, y_train_res)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.drop(columns=protected_features)), columns=X_test.drop(columns=protected_features).columns)
        X_test_scaled = pd.concat([X_test_scaled, X_test[protected_features]], axis=1)
        X_test_A = X_test_scaled[X_test_scaled[PARAMS_DATA["target"]] == PARAMS_DATA["A"]].drop(columns=protected_features).to_numpy()
        X_test_B = X_test_scaled[X_test_scaled[PARAMS_DATA["target"]] == PARAMS_DATA["B"]].drop(columns=protected_features).to_numpy()
        y_test_A = y_test[X_test_scaled[X_test_scaled[PARAMS_DATA["target"]] == PARAMS_DATA["A"]].index.tolist()]
        y_test_B = y_test[X_test_scaled[X_test_scaled[PARAMS_DATA["target"]] == PARAMS_DATA["B"]].index.tolist()]
        matrix_A = confusion_matrix(y_test_A, clf.predict(X_test_A))
        matrix_B = confusion_matrix(y_test_B, clf.predict(X_test_B))
        result_acc = actualAcc(clf.predict(X_test_scaled.drop(columns=protected_features).to_numpy()), y_test)
    else:
        prediction_A = []
        prediction_B = []
        predictions = []
        X_test_first = X_test[0]
        y_test_A = y_test[X_test_first[X_test_first[PARAMS_DATA["target"]] == PARAMS_DATA["A"]].index.tolist()]
        y_test_B = y_test[X_test_first[X_test_first[PARAMS_DATA["target"]] == PARAMS_DATA["B"]].index.tolist()]
        for X_train_m in X_train:
            X_train_m = X_train_m.drop(columns=protected_features).copy().to_numpy()
            X_train_res, y_train_res = smote.fit_resample(X_train_m, y_train)
            X_train_scaled = scaler.fit_transform(X_train_res)
            clf.fit(X_train_scaled, y_train_res)
            for X_test_m in X_test:
                X_test_scaled = pd.DataFrame(scaler.transform(X_test_m.drop(columns=protected_features)), columns=X_test_m.drop(columns=protected_features).columns)
                X_test_scaled = pd.concat([X_test_scaled, X_test_m[protected_features]], axis=1)
                X_test_A = X_test_scaled[X_test_scaled[PARAMS_DATA["target"]] == PARAMS_DATA["A"]].drop(columns=protected_features).to_numpy()
                X_test_B = X_test_scaled[X_test_scaled[PARAMS_DATA["target"]] == PARAMS_DATA["B"]].drop(columns=protected_features).to_numpy()
                prediction_A.append(clf.predict(X_test_A))
                prediction_B.append(clf.predict(X_test_B))
                predictions.append(clf.predict(X_test_scaled.drop(columns=protected_features).to_numpy()))
        # compute final predictions by voting
        predictions = np.apply_along_axis(helper_freq, 0, np.array(predictions))
        prediction_A = np.apply_along_axis(helper_freq, 0, np.array(prediction_A))
        prediction_B = np.apply_along_axis(helper_freq, 0, np.array(prediction_B))
        matrix_A = confusion_matrix(y_test_A, prediction_A)
        matrix_B = confusion_matrix(y_test_B, prediction_B)
        result_acc = actualAcc(predictions, y_test)
    # [TN_A, FP_A, FN_A, TP_A, TN_B, FP_B, FN_B, TP_B]
    result = matrix_A.ravel().tolist() + matrix_B.ravel().tolist()
    return [result, result_acc]

def test_imputation(data, completer_func=None, multi=False, verboseID=""):
    # data is Dataset object
    # completer func is the imputation function
    # multi determines whether completer_func is a multiple imputation method
    # verboseID is the name of the completer_func function
    global PARAMS
    global N_SPLITS
    clfs = { # define all the classifiers with best parameters
        "KNN": KNeighborsClassifier(n_neighbors=PARAMS["KNN"]["n_neighbors"], leaf_size=PARAMS["KNN"]["leaf_size"]),
        "LinearSVC": LinearSVC(dual=False, tol=PARAMS["LinearSVC"]["tol"], C=PARAMS["LinearSVC"]["C"], max_iter=PARAMS["LinearSVC"]["max_iter"]),
        # "SVC": SVC(tol=PARAMS["SVC"]["tol"], C=PARAMS["SVC"]["C"], max_iter=PARAMS["SVC"]["max_iter"]),
        "Forest": RandomForestClassifier(n_estimators=PARAMS["Forest"]["n_estimators"], max_depth=PARAMS["Forest"]["max_depth"], min_samples_leaf=PARAMS["Forest"]["min_samples_leaf"]),
        "LogReg": LogisticRegression(tol=PARAMS["LogReg"]["tol"], C=PARAMS["LogReg"]["C"], max_iter=PARAMS["LogReg"]["max_iter"]),
        "Tree": DecisionTreeClassifier(max_depth=PARAMS["Tree"]["max_depth"], max_leaf_nodes=PARAMS["Tree"]["max_leaf_nodes"], min_samples_leaf=PARAMS["Tree"]["min_samples_leaf"]),
        "MLP": MLPClassifier(alpha=PARAMS["MLP"]["alpha"], learning_rate_init=PARAMS["MLP"]["learning_rate_init"], max_iter=PARAMS["MLP"]["max_iter"], hidden_layer_sizes=PARAMS["MLP"]["hidden_layer_sizes"], early_stopping=True, n_iter_no_change=5),
    }
    rawdata_cv = { # save each raw confusion matrix output cv
        "KNN": [],
        "LinearSVC": [],
        # "SVC": [],
        "Forest": [],
        "LogReg": [],
        "Tree": [],
        "MLP": [],
    }
    kf = StratifiedShuffleSplit(n_splits=N_SPLITS)
    fold = 1
    X = data.X
    y = data.y
    for train_idx, test_idx in kf.split(X, y):
        print("Fold {}".format(fold))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_test = X_test.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        if completer_func:
        # do imputations on training set and test set individually
            data_incomplete = Dataset("tmp", X_train, y_train, types=data.types, 
                protected_features=data.protected_features, categorical_features=data.categorical_features,
                encoders=[data.X_encoders, data.y_encoder])
            try:
                data_complete = completer_func(data_incomplete)
            except Exception as e:
                # catch exception and skip
                print("Exception occurred in completer function '{}': {}".format(verboseID, e))
                for clf_name in clfs.keys():
                    rawdata_cv[clf_name].append([])
                fold += 1
                continue
            if ((not multi) and data_complete.X.isnull().sum().sum() > 0) or (multi and sum([dd.X.isnull().sum().sum() for dd in data_complete]) > 0):
                # if completed dataset still contains missing values, skip
                print("Completer function '{}' produces missing values, skipped".format(verboseID))
                for clf_name in clfs.keys():
                    rawdata_cv[clf_name].append([])
                fold += 1
                continue
            # apply one-hot-encoding
            if multi:
                _ = [m.preprocess() for m in data_complete]
            else:
                data_complete.preprocess()
            X_train = [m.X_encoded.copy() for m in data_complete] if multi else data_complete.X_encoded.copy()
            y_train = data_complete[0].y.copy() if multi else data_complete.y.copy()

            data_incomplete = Dataset("tmp", X_test, y_test, types=data.types, 
                protected_features=data.protected_features, categorical_features=data.categorical_features,
                encoders=[data.X_encoders, data.y_encoder])
            try:
                data_complete = completer_func(data_incomplete)
            except Exception as e:
                print("Exception occurred in completer function '{}': {}".format(verboseID, e))
                for clf_name in clfs.keys():
                    rawdata_cv[clf_name].append([])
                fold += 1
                continue
            if ((not multi) and data_complete.X.isnull().sum().sum() > 0) or (multi and sum([dd.X.isnull().sum().sum() for dd in data_complete]) > 0):
                print("Completer function '{}' produces missing values, skipped".format(verboseID))
                for clf_name in clfs.keys():
                    rawdata_cv[clf_name].append([])
                fold += 1
                continue
            # apply one-hot-encoding
            if multi:
                _ = [m.preprocess() for m in data_complete]
            else:
                data_complete.preprocess()
            X_test = [m.X_encoded.copy() for m in data_complete] if multi else data_complete.X_encoded.copy()
            y_test = data_complete[0].y.copy() if multi else data_complete.y.copy()
        # get result for each classifier
        for clf_name, clf in clfs.items():
            print(clf_name)
            result = compute_confusion_matrix(X_train, y_train, X_test, y_test, clf, data.protected_features, multi=multi)
            rawdata_cv[clf_name].append(result)
        fold += 1
    return rawdata_cv

def single_task_MAR(data_complete, complete_func, multi=False, verboseID=""):
    data_sim = gen_random(data_complete, print_all=False, columns_observed=data_complete.observed_features)
    missing_ratio = data_sim.get_missing_ratio()
    result1 = test_imputation(data_sim, complete_func, multi=multi, verboseID=verboseID)
    result2 = test_imputation(data_complete, None, multi=multi, verboseID=verboseID)
    return [result1, result2, missing_ratio]

def single_task_MNAR(data_complete, complete_func, multi=False, verboseID=""):
    data_sim = gen_not_random(data_complete, print_all=False)
    missing_ratio = data_sim.get_missing_ratio()
    result1 = test_imputation(data_sim, complete_func, multi=multi, verboseID=verboseID)
    result2 = test_imputation(data_complete, None, multi=multi, verboseID=verboseID)
    return [result1, result2, missing_ratio]

# argv[1] = process id
# argv[2] = missing generater id
if __name__=="__main__":
    if not os.path.exists("condor_outputs"):
        os.makedirs("condor_outputs")
    for tt in list(NAME_MISSING.keys()):
        if not os.path.exists(os.path.join("condor_outputs", tt)):
            os.makedirs(os.path.join("condor_outputs", tt))
        for dd in list(NAME_DATA.keys()):
            if not os.path.exists(os.path.join("condor_outputs", tt, dd)):
                os.makedirs(os.path.join("condor_outputs", tt, dd))

    if len(sys.argv) < 3:
        raise Exception("should have argument for task id, generater id")

    id_process = int(sys.argv[1])
    id_generater = int(sys.argv[2])

    with open("params_acc.json", "r") as inFile:
        params_clf = json.load(inFile)
    with open("params_datasets.json", "r") as inFile:
        params_data = json.load(inFile)

    print("Process ID: {}".format(id_process))

    global_time = time.time()

    for dataName in NAME_DATA.keys():
        final_result = {}
        PARAMS_DATA = params_data[dataName]
        PARAMS = params_clf[dataName]

        if dataName == "adult":
            data_complete = create_adult_dataset()
        elif dataName == "compas":
            data_complete = create_compas_dataset()
            tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
            tmp_concat.dropna(inplace=True)
            tmp_concat.reset_index(drop=True, inplace=True)
            data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
            data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
        elif dataName == "titanic":
            data_complete = create_titanic_dataset()
            tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
            tmp_concat.dropna(inplace=True)
            tmp_concat.reset_index(drop=True, inplace=True)
            data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
            data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
            N_SPLITS = 5
        elif dataName == "german":
            data_complete = create_german_dataset()
            N_SPLITS = 5
        elif dataName == "communities":
            data_complete = create_communities_dataset()
            tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
            tmp_concat.dropna(inplace=True)
            tmp_concat.reset_index(drop=True, inplace=True)
            data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
            data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
        elif dataName == "bank":
            data_complete = create_bank_dataset()
        else: raise ValueError("Cannot recognize dataset: {}".format(dataName))

        if id_generater == NAME_MISSING["MAR"]:
            single_task = single_task_MAR
            task_name = "MAR"
        elif id_generater == NAME_MISSING["MNAR"]:
            single_task = single_task_MNAR
            task_name = "MNAR"
        else: raise ValueError("Cannot recognize generator ID: {}".format(id_generater))

        start_time = time.time()

        if RUN_MEAN_V1:
            print("Now running {} on {}".format("mean_v1", dataName))
            final_result["mean_v1"] = single_task(data_complete, complete_by_mean_col, False, "mean_v1")
            print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
            start_time = time.time()

        if RUN_MEAN_V2:
            print("Now running {} on {}".format("mean_v2", dataName))
            final_result["mean_v2"] = single_task(data_complete, partial(complete_by_mean_col_v2, target_feature=PARAMS_DATA["target"]), False, "mean_v2")
            print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
            start_time = time.time()

        if RUN_SIMILAR_V1:
            print("Now running {} on {}".format("similar_v1", dataName))
            final_result["similar_v1"] = single_task(data_complete, complete_by_similar_row, False, "similar_v1")
            print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
            start_time = time.time()

        if RUN_SIMILAR_V2:
            print("Now running {} on {}".format("similar_v2", dataName))
            final_result["similar_v2"] = single_task(data_complete, partial(complete_by_similar_row_v2, target_feature=PARAMS_DATA["target"]), False, "similar_v2")
            print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
            start_time = time.time()

        if RUN_MULTI_V1:
            print("Now running {} on {}".format("multi_v1", dataName))
            final_result["multi_v1"] = single_task(data_complete, complete_by_multi, True, "multi_v1")
            print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
            start_time = time.time()

        if RUN_MULTI_V2:
            print("Now running {} on {}".format("multi_v2", dataName))
            final_result["multi_v2"] = single_task(data_complete, partial(complete_by_multi_v2, target_feature=PARAMS_DATA["target"]), True, "multi_v2")
            print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
            start_time = time.time()

        with open(os.path.join("condor_outputs", task_name, dataName, "output_{:0>4}.pkl".format(id_process)), "wb") as outFile:
            pickle.dump(final_result, outFile)

    print("All tasks complete in {:.2f}hr".format((time.time() - global_time) / 3600))