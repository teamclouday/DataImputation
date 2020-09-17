# script for each single task for condor-submit

# this script is currently for computing random ratios on compas analysis

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
import multiprocessing
from multiprocessing.pool import Pool
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from utils.data import create_compas_dataset, create_adult_dataset, create_titanic_dataset, create_communities_dataset, create_german_dataset, create_bank_dataset, Dataset
from utils.generator import gen_complete_random
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

RUN_DEBUG       = False

NAME_DATA = {
    "adult": 1,
    "compas": 2,
    "titanic": 3,
    "german": 4,
    "communities": 5,
    "bank": 6,
}
NAME_TARGET = {
    "acc": 1,
    "f1": 2,
}
PARAMS = None
PARAMS_DATA = None
N_SPLITS = 10 # set to 5 when running on titanic

#define functions

# [TN_A, FP_A, FN_A, TP_A, TN_B, FP_B, FN_B, TP_B]
def bias1(data):
    # input should be data from compute_confusion_matrix
    # bias 1 = |(FPR_A/FNR_A) - (FPR_B/FNR_B)|
    FPR_A = data[1] / (data[1] + data[0])
    FNR_A = data[2] / (data[2] + data[3])
    FPR_B  = data[5] / (data[5] + data[4])
    FNR_B  = data[6] / (data[6] + data[7])
    if FNR_A == 0 or FNR_B == 0: return -1 # mark error situation
    bias = (FPR_A / FNR_A) - (FPR_B / FNR_B)
    return abs(bias)

def bias2(data):
    # input should be data from compute_confusion_matrix
    # bias 2 = |(FPR_A/FPR_B) - (FNR_A/FNR_B)|
    FPR_A = data[1] / (data[1] + data[0])
    FNR_A = data[2] / (data[2] + data[3])
    FPR_B  = data[5] / (data[5] + data[4])
    FNR_B  = data[6] / (data[6] + data[7])
    if FNR_B == 0 or FPR_B == 0: return -1 # mark error situation
    bias = (FPR_A / FPR_B) - (FNR_A / FNR_B)
    return abs(bias)

def acc(data):
    # input should be data from compute_confusion_matrix
    # acc = (TP + TN) / (TP + TN + FP + FN)
    TP = data[3] + data[7]
    TN = data[0] + data[4]
    accuracy = (TP + TN) / sum(data)
    return accuracy

def f1score(data):
    # input should be data from compute_confusion_matrix
    # precision = TP / (TP + FP)
    # recall    = TP / (TP + FN)
    # f1 score  = 2 * (precision * recall) / (recall + precision)
    precision_A = data[3] / (data[3] + data[1]) if (data[3] + data[1]) != 0 else 0
    precision_B = data[7] / (data[7] + data[5]) if (data[7] + data[5]) != 0 else 0
    recall_A    = data[3] / (data[3] + data[2])
    recall_B    = data[7] / (data[7] + data[6])
    if (recall_A + precision_A) == 0 or (recall_B + precision_B) == 0:
        return [None] # mark error situation
    f1_A = 2 * (precision_A * recall_A) / (recall_A + precision_A)
    f1_B = 2 * (precision_B * recall_B) / (recall_B + precision_B)
    return [f1_A, f1_B]

def newBias(data, A=1, B=1):
    # Pr(AA is labeled as low risk when he is actually high risk) = Pr(Caucasian is labeled as low risk when actually high risk) 
    # Pr(AA is labeled as high risk when he is low risk) =  Pr(Caucasian is labeled as high risk when actually low risk)
    # bias = |LHS - RHS|
    # A*|LHS - RHS of first type| + B*|LHS - RHS of second type|
    # A*|FPR_A - FPR_B| + B*|FNR_A - FNR_C|
    FPR_A = data[1] / (data[1] + data[0])
    FNR_A = data[2] / (data[2] + data[3])
    FPR_B  = data[5] / (data[5] + data[4])
    FNR_B  = data[6] / (data[6] + data[7])
    bias = A*abs(FPR_A - FPR_B) + B*abs(FNR_A - FNR_B)
    return bias

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
        "SVC": SVC(tol=PARAMS["SVC"]["tol"], C=PARAMS["SVC"]["C"], max_iter=PARAMS["SVC"]["max_iter"]),
        "Forest": RandomForestClassifier(n_estimators=PARAMS["Forest"]["n_estimators"], max_depth=PARAMS["Forest"]["max_depth"], min_samples_leaf=PARAMS["Forest"]["min_samples_leaf"]),
        "LogReg": LogisticRegression(tol=PARAMS["LogReg"]["tol"], C=PARAMS["LogReg"]["C"], max_iter=PARAMS["LogReg"]["max_iter"]),
        "Tree": DecisionTreeClassifier(max_depth=PARAMS["Tree"]["max_depth"], max_leaf_nodes=PARAMS["Tree"]["max_leaf_nodes"], min_samples_leaf=PARAMS["Tree"]["min_samples_leaf"]),
        "MLP": MLPClassifier(alpha=PARAMS["MLP"]["alpha"], learning_rate_init=PARAMS["MLP"]["learning_rate_init"], max_iter=PARAMS["MLP"]["max_iter"], hidden_layer_sizes=PARAMS["MLP"]["hidden_layer_sizes"]),
    }
    rawdata_cv = { # save each raw confusion matrix output cv
        "KNN": [],
        "LinearSVC": [],
        "SVC": [],
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
            result = compute_confusion_matrix(X_train, y_train, X_test, y_test, clf, data.protected_features, multi=multi)
            rawdata_cv[clf_name].append(result)
        fold += 1
    return rawdata_cv

# prepare data

data_complete = None
if RUN_DEBUG:
    data = create_compas_dataset()
    data_complete = data.copy()
    tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
    tmp_concat.dropna(inplace=True)
    tmp_concat.reset_index(drop=True, inplace=True)
    data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
    data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
    with open("params_acc.json", "r") as inFile:
        PARAMS = json.load(inFile)
    PARAMS = PARAMS["compas"]

random_ratios = np.linspace(0.0, 1.0, num=20, endpoint=False)

MAX_PROCESS_COUNT = multiprocessing.cpu_count()

# define single task functions

def complete_mean_task(idx):
    global data_complete
    data_sim = gen_complete_random(data_complete, random_ratio=random_ratios[idx], print_all=False)
    result = test_imputation(data_sim, complete_by_mean_col, multi=False, verboseID="mean_v1")
    return result

def complete_mean_v2_task(idx):
    global data_complete, PARAMS_DATA
    data_sim = gen_complete_random(data_complete, random_ratio=random_ratios[idx], print_all=False)
    result = test_imputation(data_sim, partial(complete_by_mean_col_v2, target_feature=PARAMS_DATA["target"]), multi=False, verboseID="mean_v2")
    return result

def complete_similar_task(idx):
    global data_complete
    data_sim = gen_complete_random(data_complete, random_ratio=random_ratios[idx], print_all=False)
    result = test_imputation(data_sim, complete_by_similar_row, multi=False, verboseID="similar_v1")
    return result

def complete_similar_v2_task(idx):
    global data_complete, PARAMS_DATA
    data_sim = gen_complete_random(data_complete, random_ratio=random_ratios[idx], print_all=False)
    result = test_imputation(data_sim, partial(complete_by_similar_row_v2, target_feature=PARAMS_DATA["target"]), multi=False, verboseID="similar_v2")
    return result

def complete_multi_task(idx):
    global data_complete
    data_sim = gen_complete_random(data_complete, random_ratio=random_ratios[idx], print_all=False)
    result = test_imputation(data_sim, complete_by_multi, multi=True, verboseID="multi_v1")
    return result

def complete_multi_v2_task(idx):
    global data_complete, PARAMS_DATA
    data_sim = gen_complete_random(data_complete, random_ratio=random_ratios[idx], print_all=False)
    result = test_imputation(data_sim, partial(complete_by_multi_v2, target_feature=PARAMS_DATA["target"]), multi=True, verboseID="multi_v2")
    return result

# argv[1] = process id
# argv[2] = data id
# argv[3] = target id
if __name__ == "__main__":
    if not RUN_DEBUG:
        if not os.path.exists("condor_outputs"):
            os.makedirs("condor_outputs")
        for tt in list(NAME_TARGET.keys()):
            if not os.path.exists(os.path.join("condor_outputs", tt)):
                os.makedirs(os.path.join("condor_outputs", tt))
            for dd in list(NAME_DATA.keys()):
                if not os.path.exists(os.path.join("condor_outputs", tt, dd)):
                    os.makedirs(os.path.join("condor_outputs", tt, dd))
        if len(sys.argv) < 4:
            raise Exception("should have argument for task id, data id, target id")
        id_data = int(sys.argv[2])
        id_target = int(sys.argv[3])
        if id_target == NAME_TARGET["acc"]:
            targetName = "acc"
            fileName = "params_acc.json"
        elif id_target == NAME_TARGET["f1"]:
            targetName = "f1"
            fileName = "params_f1.json"
        else: raise ValueError("target id is wrong")
        if id_data == NAME_DATA["adult"]:
            dataName = "adult"
            data_complete = create_adult_dataset()
        elif id_data == NAME_DATA["compas"]:
            dataName = "compas"
            data_complete = create_compas_dataset()
            tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
            tmp_concat.dropna(inplace=True)
            tmp_concat.reset_index(drop=True, inplace=True)
            data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
            data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
        elif id_data == NAME_DATA["titanic"]:
            dataName = "titanic"
            data_complete = create_titanic_dataset()
            tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
            tmp_concat.dropna(inplace=True)
            tmp_concat.reset_index(drop=True, inplace=True)
            data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
            data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
            N_SPLITS = 5
        elif id_data == NAME_DATA["german"]:
            dataName = "german"
            data_complete = create_german_dataset()
            N_SPLITS = 5
        elif id_data == NAME_DATA["communities"]:
            dataName = "communities"
            data_complete = create_communities_dataset()
            tmp_concat = pd.concat([data_complete.X, pd.DataFrame(data_complete.y, columns=["_TARGET_"])], axis=1)
            tmp_concat.dropna(inplace=True)
            tmp_concat.reset_index(drop=True, inplace=True)
            data_complete.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
            data_complete.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
        elif id_data == NAME_DATA["bank"]:
            dataName = "bank"
            data_complete = create_bank_dataset()
        else: raise ValueError("data id is wrong")
        with open(fileName, "r") as inFile:
            PARAMS = json.load(inFile)
        PARAMS = PARAMS[dataName]
        with open("params_datasets.json", "r") as inFile:
            PARAMS_DATA = json.load(inFile)
        PARAMS_DATA = PARAMS_DATA[dataName]
        print("Script ID: {}, Dataset Name: {}, Target Name: {}".format(sys.argv[1], dataName, targetName))

    final_result = {}
    start_time = time.time()
    total_time = time.time()

    # run mean version 1
    if RUN_MEAN_V1:
        print("Now running mean imputation version 1")
        final_result["mean_v1"] = []
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result["mean_v1"] = list(pool.imap(complete_mean_task, range(len(random_ratios))))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
        start_time = time.time()

    # run mean version 2
    if RUN_MEAN_V2:
        print("Now running mean imputation version 2")
        final_result["mean_v2"] = []
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result["mean_v2"] = list(pool.imap(complete_mean_v2_task, range(len(random_ratios))))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
        start_time = time.time()

    # run similar version 1
    if RUN_SIMILAR_V1:
        print("Now running similar imputation version 1")
        final_result["similar_v1"] = []
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result["similar_v1"] = list(pool.imap(complete_similar_task, range(len(random_ratios))))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
        start_time = time.time()

    # run similar version 2
    if RUN_SIMILAR_V2:
        print("Now running similar imputation version 2")
        final_result["similar_v2"] = []
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result["similar_v2"] = list(pool.imap(complete_similar_v2_task, range(len(random_ratios))))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
        start_time = time.time()

    # run multi version 1
    if RUN_MULTI_V1:
        print("Now running multiple imputation version 1")
        final_result["multi_v1"] = []
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result["multi_v1"] = list(pool.imap(complete_multi_task, range(len(random_ratios))))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
        start_time = time.time()

    # run multi version 2
    if RUN_MULTI_V2:
        print("Now running multiple imputation version 2")
        final_result["multi_v2"] = []
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result["multi_v2"] = list(pool.imap(complete_multi_v2_task, range(len(random_ratios))))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))
        start_time = time.time()

    # save outputs
    if not RUN_DEBUG:
        with open(os.path.join("condor_outputs", targetName, dataName, "output_{:0>4}.pkl".format(sys.argv[1])), "wb") as outFile:
            pickle.dump(final_result, outFile)

        print("All tasks complete in {:.2f}hr".format((time.time() - total_time) / 3600))

    if RUN_DEBUG:
        import tqdm
        print("Now running debug task (mean_v1)")
        MAX_PROCESS_COUNT = (MAX_PROCESS_COUNT - 1) if MAX_PROCESS_COUNT > 1 else 1
        with Pool(processes=MAX_PROCESS_COUNT) as pool:
            final_result = list(tqdm.tqdm(pool.imap(complete_mean_task, range(len(random_ratios))), total=len(random_ratios)))
        print("Task complete in {:.2f}min".format((time.time() - start_time) / 60))

        with open("debug_data.pkl", "wb") as outFile:
            pickle.dump(final_result, outFile)