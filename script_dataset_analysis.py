# script for generate analysis for datasets

import os
import json
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from utils.data import Dataset
from utils.data import create_adult_dataset, create_bank_dataset
from utils.data import create_communities_dataset, create_compas_dataset
from utils.data import create_german_dataset, create_titanic_dataset

from utils.generator import gen_complete_random
from utils.completer import complete_by_mean_col, complete_by_mean_col_v2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

PLOT_DROP           = False
PLOT_IMPUTE         = False
PLOT_SCATTER        = False
PLOT_DROP_SINGLE    = False
PLOT_DROP_EXCLUDE   = False

PLOT_ADULT          = True
PLOT_COMPAS         = True
PLOT_TITANIC        = True
PLOT_GERMAN         = True
PLOT_COMMUNITIES    = True
PLOT_BANK           = True

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

def cross_val(data_original: Dataset, data_config, clf_config, complete_function=None, selected_cols=[]):
    bias = []
    acc = []
    smote = SMOTE()
    scaler = StandardScaler()
    for i in range(20):
        if complete_function: data = gen_complete_random(data_original, random_ratio=0.4, selected_cols=selected_cols)
        else: data = data_original
        print("Running Cross Validation {}".format(i))
        bias_cv = []
        acc_cv = []
        for train_idx, test_idx in StratifiedShuffleSplit(n_splits=20).split(data.X, data.y):
            X_train, X_test = data.X.iloc[train_idx].copy(), data.X.iloc[test_idx].copy()
            Y_train, Y_test = data.y[train_idx], data.y[test_idx]
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)

            if complete_function:
                data_incomplete = Dataset("tmp", X_train, Y_train, types=data.types, 
                    protected_features=data.protected_features, categorical_features=data.categorical_features,
                    encoders=[data.X_encoders, data.y_encoder])
                try:
                    data_complete = complete_function(data_incomplete)
                except Exception as e:
                    print("Error: {}. Skipped".format(e))
                    continue
                if data_complete.X.isnull().sum().sum() > 0:
                    print("Complete function error, skipped")
                    continue
                X_train = data_complete.X.copy()
                Y_train = data_complete.y.copy()
            X_train.drop(columns=data.protected_features, inplace=True)

            if complete_function:
                data_incomplete = Dataset("tmp", X_test, Y_test, types=data.types, 
                    protected_features=data.protected_features, categorical_features=data.categorical_features,
                    encoders=[data.X_encoders, data.y_encoder])
                try:
                    data_complete = complete_function(data_incomplete)
                except Exception as e:
                    print("Error: {}. Skipped".format(e))
                    continue
                if data_complete.X.isnull().sum().sum() > 0:
                    print("Complete function error, skipped")
                    continue
                X_test = data_complete.X.copy()
                Y_test = data_complete.y.copy()
            
            X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
            X_scaled = scaler.fit_transform(X_train_res)
            clf = LogisticRegression(max_iter=clf_config["max_iter"], C=clf_config["C"], tol=clf_config["tol"])
            clf.fit(X_scaled, Y_train_res)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test.drop(columns=data.protected_features)), columns=X_test.drop(columns=data.protected_features).columns)
            X_test_scaled = pd.concat([X_test_scaled, X_test[data.protected_features]], axis=1)
            X_test_A = X_test_scaled[X_test_scaled[data_config["target"]] == data_config["A"]].drop(columns=data.protected_features).to_numpy()
            X_test_B = X_test_scaled[X_test_scaled[data_config["target"]] == data_config["B"]].drop(columns=data.protected_features).to_numpy()
            Y_test_A = Y_test[X_test_scaled[X_test_scaled[data_config["target"]] == data_config["A"]].index.tolist()]
            Y_test_B = Y_test[X_test_scaled[X_test_scaled[data_config["target"]] == data_config["B"]].index.tolist()]
            matrix_A = confusion_matrix(Y_test_A, clf.predict(X_test_A)).ravel().tolist()
            matrix_B = confusion_matrix(Y_test_B, clf.predict(X_test_B)).ravel().tolist()
            try:
                bias_cv.append(newBias(matrix_A+matrix_B))
            except Exception as e:
                print("\tError: {}, skipped".format(e))
            acc_cv.append(accuracy_score(clf.predict(X_test_scaled.drop(columns=data.protected_features).to_numpy()), Y_test))
        bias.append(np.mean(bias_cv))
        acc.append(np.mean(acc_cv))
    return (np.mean(bias), np.mean(acc))

def drop_na(data: Dataset) -> Dataset:
    data = data.copy()
    tmp_concat = pd.concat([data.X, pd.DataFrame(data.y, columns=["_TARGET_"])], axis=1)
    tmp_concat.dropna(inplace=True)
    tmp_concat.reset_index(drop=True, inplace=True)
    data.X = tmp_concat.drop(columns=["_TARGET_"]).copy()
    data.y = tmp_concat["_TARGET_"].copy().to_numpy().ravel()
    return data

def convert_protected(data: Dataset) -> Tuple[Dataset, LabelEncoder]:
    data = data.copy()
    encoder = LabelEncoder()
    for feature in data.protected_features:
        data.X[feature] = encoder.fit_transform(data.X[feature])
    return data, encoder

def concat(data: Dataset) -> pd.DataFrame:
    data = data.copy()
    return pd.concat([data.X, pd.DataFrame(data.y, columns=["_TARGET_"])], axis=1)

def analysis_drop_correlated_features(data_fn, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("params_datasets.json", "r") as inFile:
        dataConfig = json.load(inFile)
    with open("params_acc.json", "r") as inFile:
        clfConfig = json.load(inFile)
    data: Dataset = drop_na(data_fn())
    cdata, encoder = convert_protected(data)
    ccdata = concat(cdata)
    print("Dataset: {}".format(data.name))
    dataConfig = dataConfig[data.name]
    clfConfig = clfConfig[data.name]["LogReg"]
    correlation = ccdata.corr()[data.protected_features[0]]
    correlation = correlation[correlation.abs().sort_values(ascending=False).head(15).index]
    if "_TARGET_" in correlation.keys():
        del correlation["_TARGET_"]
    del correlation[data.protected_features[0]]
    print("Correlation with protected attribute:")
    for key, val in correlation.items():
        print("\tFeature: {:<30} Corr: {:.4f}".format(key, val))
    print("Computing accuracy and bias by drop")
    correlated_features = np.array(correlation.keys())
    plot_bias = []
    plot_acc = []
    for i in range(len(correlated_features)):
        print("Drop {:<2} most correlated features".format(i))
        data_tmp = data.copy()
        data_tmp.X.drop(columns=correlated_features[range(i)], inplace=True)
        # print(data_tmp.X.columns)
        bias, acc = cross_val(data_tmp, dataConfig, clfConfig)
        plot_bias.append(bias)
        plot_acc.append(acc)
    print("Generating plots")
    fig, axes = plt.subplots(3, 1, figsize=(5, 16))
    plot_pos = np.arange(len(correlated_features))
    axes[0].barh(plot_pos, correlation.abs(), align='center', height=0.5, tick_label=correlated_features)
    axes[0].set_xlabel("Absolute Correlation with Protected Attribute")

    axes[1].bar(plot_pos, plot_bias, tick_label=[str(i) for i in range(len(correlated_features))], width=0.8, align="center")
    axes[1].set_xlabel("Features Dropped")
    axes[1].set_ylabel("Bias")
    axes[1].set_ylim([0.0, 1.0])
    for i, val in enumerate(plot_bias):
        axes[1].text(i, val+0.05, "{:.2f}".format(val), horizontalalignment='center')

    axes[2].bar(plot_pos, plot_acc, tick_label=[str(i) for i in range(len(correlated_features))], width=0.8, align="center")
    axes[2].set_xlabel("Features Dropped")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim([0.0, 1.2])
    for i, val in enumerate(plot_acc):
        axes[2].text(i, val+0.05, "{:.2f}".format(val), horizontalalignment='center')
    fig.suptitle("Most Correlated Feature Drop Experiment ({})".format(data.name))
    plt.subplots_adjust(top=0.94)
    fig.savefig(os.path.join(folder, filename), transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    print("Done")

def analysis_impute_correlated_features(data_fn, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("params_datasets.json", "r") as inFile:
        dataConfig = json.load(inFile)
    with open("params_acc.json", "r") as inFile:
        clfConfig = json.load(inFile)
    data: Dataset = drop_na(data_fn())
    cdata, encoder = convert_protected(data)
    ccdata = concat(cdata)
    print("Dataset: {}".format(data.name))
    dataConfig = dataConfig[data.name]
    clfConfig = clfConfig[data.name]["LogReg"]
    correlation = ccdata.corr()[data.protected_features[0]]
    correlation = correlation[correlation.abs().sort_values(ascending=False).head(15).index]
    if "_TARGET_" in correlation.keys():
        del correlation["_TARGET_"]
    del correlation[data.protected_features[0]]
    print("Correlation with protected attribute:")
    for key, val in correlation.items():
        print("\tFeature: {:<30} Corr: {:.4f}".format(key, val))
    print("Computing accuracy and bias by MCAR and Mean Imputation")
    correlated_features = np.array(correlation.keys())
    plot_bias = []
    plot_acc = []
    for i in range(len(correlated_features)):
        print("Impute on {:<2} most correlated features".format(i))
        data_tmp = data.copy()
        # data_tmp.X.drop(columns=correlated_features[range(i)], inplace=True)
        # data_tmp = gen_complete_random(data_tmp, random_ratio=0.2, selected_cols=correlated_features[range(i)])
        bias, acc = cross_val(data_tmp, dataConfig, clfConfig, complete_by_mean_col, correlated_features[range(i)])
        plot_bias.append(bias)
        plot_acc.append(acc)
    print("Generating plots")
    fig, axes = plt.subplots(3, 1, figsize=(5, 16))
    plot_pos = np.arange(len(correlated_features))
    axes[0].barh(plot_pos, correlation.abs(), align='center', height=0.5, tick_label=correlated_features)
    axes[0].set_xlabel("Absolute Correlation with Protected Attribute")

    axes[1].bar(plot_pos, plot_bias, tick_label=[str(i) for i in range(len(correlated_features))], width=0.8, align="center")
    axes[1].set_xlabel("Features Imputed")
    axes[1].set_ylabel("Bias")
    axes[1].set_ylim([0.0, 1.0])
    for i, val in enumerate(plot_bias):
        axes[1].text(i, val+0.05, "{:.2f}".format(val), horizontalalignment='center')

    axes[2].bar(plot_pos, plot_acc, tick_label=[str(i) for i in range(len(correlated_features))], width=0.8, align="center")
    axes[2].set_xlabel("Features Imputed")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim([0.0, 1.2])
    for i, val in enumerate(plot_acc):
        axes[2].text(i, val+0.05, "{:.2f}".format(val), horizontalalignment='center')
    fig.suptitle("Most Correlated Feature Impute Experiment ({})".format(data.name))
    plt.subplots_adjust(top=0.94)
    fig.savefig(os.path.join(folder, filename), transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    print("Done")

def analysis_scatter_correlated_features(data_fn, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    data: Dataset = drop_na(data_fn())
    cdata, encoder = convert_protected(data)
    ccdata = concat(cdata)
    print("Dataset: {}".format(data.name))
    print("Plotting correlation of features with target & protected")
    correlation_protected = ccdata.corr()[data.protected_features[0]]
    del correlation_protected["_TARGET_"]
    del correlation_protected[data.protected_features[0]]
    correlation_target = ccdata.corr()["_TARGET_"]
    del correlation_target["_TARGET_"]
    del correlation_target[data.protected_features[0]]
    plotX = []
    plotY = []
    for key in correlation_protected.keys():
        plotX.append(correlation_protected[key])
        plotY.append(correlation_target[key])
    plotX = np.abs(np.array(plotX))
    plotY = np.abs(np.array(plotY))
    plt.figure(figsize=(10, 10))
    plt.scatter(plotX, plotY, s=200, alpha=0.8)
    for i, val in enumerate(correlation_protected.keys()):
        plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
    plt.xlabel("Protected Feature Correlation")
    plt.ylabel("Target Correlation")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.title("Correlation Plot ({})".format(data.name))
    plt.savefig(os.path.join(folder, filename), transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    with open(os.path.join(folder, "{}.txt".format(filename)), "w") as outFile:
        for i, val in enumerate(correlation_protected.keys()):
            print("{:<2}: {}".format(i+1, val), file=outFile)
    print("Done")

def analysis_drop_correlated_features_single(data_fn, folder, filename, draw_original=True, draw_impute_v1=True, draw_impute_v2=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("params_datasets.json", "r") as inFile:
        dataConfig = json.load(inFile)
    with open("params_acc.json", "r") as inFile:
        clfConfig = json.load(inFile)
    data: Dataset = drop_na(data_fn())
    cdata, encoder = convert_protected(data)
    ccdata = concat(cdata)
    print("Dataset: {}".format(data.name))
    dataConfig = dataConfig[data.name]
    clfConfig = clfConfig[data.name]["LogReg"]
    print("Computing bias & accuracy for each single feature (10 most correlated with protected)")
    correlation = ccdata.corr()[data.protected_features[0]]
    del correlation["_TARGET_"]
    del correlation[data.protected_features[0]]
    correlation = correlation[correlation.abs().sort_values(ascending=False).head(10).index]
    if draw_original:
        plotX = [] # bias
        plotY = [] # accuracy
        for key in correlation.keys():
            print("Compute on single feature {}".format(key))
            data_tmp = data.copy()
            data_tmp.X.drop(columns=[f for f in correlation.keys() if f != key], inplace=True)
            bias, acc = cross_val(data_tmp, dataConfig, clfConfig)
            plotX.append(acc)
            plotY.append(bias)
        plotY_compare, plotX_compare = cross_val(data.copy(), dataConfig, clfConfig)
        plotX = np.abs(np.array(plotX))
        plotY = np.abs(np.array(plotY))
        plt.figure(figsize=(10, 10))
        plt.scatter(plotX, plotY, s=200, alpha=0.8)
        plt.scatter([plotX_compare], [plotY_compare], s=200, alpha=1.0, c="orange")
        for i, _ in enumerate(correlation.keys()):
            plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
        plt.ylabel("Bias")
        plt.xlabel("Accuracy")
        plt.xlim([0.4, 1.0])
        plt.title("Most Correlated Single Feature Experiment ({})".format(data.name))
        plt.savefig(os.path.join(folder, filename+".png"), transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    if draw_impute_v1:
        plotX = [] # bias
        plotY = [] # accuracy
        for key in correlation.keys():
            print("Compute on single feature {} (Impute V1)".format(key))
            data_tmp = data.copy()
            # data_tmp = gen_complete_random(data_tmp, random_ratio=0.4, selected_cols=[f for f in correlation.keys() if f != key])
            # data_tmp.X.drop(columns=[f for f in correlation.keys() if f != key], inplace=True)
            bias, acc = cross_val(data_tmp, dataConfig, clfConfig, complete_by_mean_col, [f for f in correlation.keys() if f != key])
            plotX.append(acc)
            plotY.append(bias)
        plotY_compare, plotX_compare = cross_val(data.copy(), dataConfig, clfConfig)
        plotX = np.abs(np.array(plotX))
        plotY = np.abs(np.array(plotY))
        plt.figure(figsize=(10, 10))
        plt.scatter(plotX, plotY, s=200, alpha=0.8)
        plt.scatter([plotX_compare], [plotY_compare], s=200, alpha=1.0, c="orange")
        for i, _ in enumerate(correlation.keys()):
            plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
        plt.ylabel("Bias")
        plt.xlabel("Accuracy")
        plt.xlim([0.4, 1.0])
        plt.title("Most Correlated Single Feature Experiment ({}) (Mean V1)".format(data.name))
        plt.savefig(os.path.join(folder, filename+"_v1.png"), transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    if draw_impute_v2:
        plotX = [] # bias
        plotY = [] # accuracy
        for key in correlation.keys():
            print("Compute on single feature {} (Impute V2)".format(key))
            data_tmp = data.copy()
            # data_tmp = gen_complete_random(data_tmp, random_ratio=0.4, selected_cols=[f for f in correlation.keys() if f != key])
            # data_tmp.X.drop(columns=[f for f in correlation.keys() if f != key], inplace=True)
            bias, acc = cross_val(data_tmp, dataConfig, clfConfig, partial(complete_by_mean_col_v2, target_feature=dataConfig["target"]), [f for f in correlation.keys() if f != key])
            plotX.append(acc)
            plotY.append(bias)
        plotY_compare, plotX_compare = cross_val(data.copy(), dataConfig, clfConfig)
        plotX = np.abs(np.array(plotX))
        plotY = np.abs(np.array(plotY))
        plt.figure(figsize=(10, 10))
        plt.scatter(plotX, plotY, s=200, alpha=0.8)
        plt.scatter([plotX_compare], [plotY_compare], s=200, alpha=1.0, c="orange")
        for i, _ in enumerate(correlation.keys()):
            plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
        plt.ylabel("Bias")
        plt.xlabel("Accuracy")
        plt.xlim([0.4, 1.0])
        plt.title("Most Correlated Single Feature Experiment ({}) (Mean V2)".format(data.name))
        plt.savefig(os.path.join(folder, filename+"_v2.png"), transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    print("Done")

def analysis_drop_correlated_features_exclude(data_fn, folder, filename, draw_original=True, draw_impute_v1=True, draw_impute_v2=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("params_datasets.json", "r") as inFile:
        dataConfig = json.load(inFile)
    with open("params_acc.json", "r") as inFile:
        clfConfig = json.load(inFile)
    data: Dataset = drop_na(data_fn())
    cdata, encoder = convert_protected(data)
    ccdata = concat(cdata)
    print("Dataset: {}".format(data.name))
    dataConfig = dataConfig[data.name]
    clfConfig = clfConfig[data.name]["LogReg"]
    print("Computing bias & accuracy excluding single feature (10 most correlated with protected)")
    correlation = ccdata.corr()[data.protected_features[0]]
    del correlation["_TARGET_"]
    del correlation[data.protected_features[0]]
    correlation = correlation[correlation.abs().sort_values(ascending=False).head(10).index]
    if draw_original:
        plotX = [] # bias
        plotY = [] # accuracy
        for key in correlation.keys():
            print("Compute by excluding feature {}".format(key))
            data_tmp = data.copy()
            data_tmp.X.drop(columns=[key], inplace=True)
            bias, acc = cross_val(data_tmp, dataConfig, clfConfig)
            plotX.append(acc)
            plotY.append(bias)
        plotY_compare, plotX_compare = cross_val(data.copy(), dataConfig, clfConfig)
        plotX = np.abs(np.array(plotX))
        plotY = np.abs(np.array(plotY))
        plt.figure(figsize=(10, 10))
        plt.scatter(plotX, plotY, s=200, alpha=0.8)
        plt.scatter([plotX_compare], [plotY_compare], s=200, alpha=1.0, c="orange")
        for i, _ in enumerate(correlation.keys()):
            plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
        plt.ylabel("Bias")
        plt.xlabel("Accuracy")
        plt.xlim([0.4, 1.0])
        plt.title("Most Correlated Excluding Feature Experiment ({})".format(data.name))
        plt.savefig(os.path.join(folder, filename+".png"), transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    if draw_impute_v1:
        plotX = [] # bias
        plotY = [] # accuracy
        for key in correlation.keys():
            print("Compute by excluding feature {} (Impute V1)".format(key))
            data_tmp = data.copy()
            # data_tmp.X.drop(columns=[key], inplace=True)
            # data_tmp = gen_complete_random(data_tmp, random_ratio=0.4, selected_cols=[key])
            bias, acc = cross_val(data_tmp, dataConfig, clfConfig, complete_by_mean_col, [key])
            plotX.append(acc)
            plotY.append(bias)
        plotY_compare, plotX_compare = cross_val(data.copy(), dataConfig, clfConfig)
        plotX = np.abs(np.array(plotX))
        plotY = np.abs(np.array(plotY))
        plt.figure(figsize=(10, 10))
        plt.scatter(plotX, plotY, s=200, alpha=0.8)
        plt.scatter([plotX_compare], [plotY_compare], s=200, alpha=1.0, c="orange")
        for i, _ in enumerate(correlation.keys()):
            plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
        plt.ylabel("Bias")
        plt.xlabel("Accuracy")
        plt.xlim([0.4, 1.0])
        plt.title("Most Correlated Excluding Feature Experiment ({}) (Mean V1)".format(data.name))
        plt.savefig(os.path.join(folder, filename+"_v1.png"), transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    if draw_impute_v2:
        plotX = [] # bias
        plotY = [] # accuracy
        for key in correlation.keys():
            print("Compute by excluding feature {} (Impute V2)".format(key))
            data_tmp = data.copy()
            # data_tmp.X.drop(columns=[key], inplace=True)
            # data_tmp = gen_complete_random(data_tmp, random_ratio=0.4, selected_cols=[key])
            bias, acc = cross_val(data_tmp, dataConfig, clfConfig, partial(complete_by_mean_col_v2, target_feature=dataConfig["target"]), [key])
            plotX.append(acc)
            plotY.append(bias)
        plotY_compare, plotX_compare = cross_val(data.copy(), dataConfig, clfConfig)
        plotX = np.abs(np.array(plotX))
        plotY = np.abs(np.array(plotY))
        plt.figure(figsize=(10, 10))
        plt.scatter(plotX, plotY, s=200, alpha=0.8)
        plt.scatter([plotX_compare], [plotY_compare], s=200, alpha=1.0, c="orange")
        for i, _ in enumerate(correlation.keys()):
            plt.text(plotX[i], plotY[i], "{}".format(i+1), horizontalalignment='center', verticalalignment='center')
        plt.ylabel("Bias")
        plt.xlabel("Accuracy")
        plt.xlim([0.4, 1.0])
        plt.title("Most Correlated Excluding Feature Experiment ({}) (Mean V2)".format(data.name))
        plt.savefig(os.path.join(folder, filename+"_v2.png"), transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    print("Done")

if __name__=="__main__":
    if PLOT_DROP:
        if PLOT_ADULT:
            analysis_drop_correlated_features(create_adult_dataset, os.path.join("dataset_analysis_plots", "DROP"), "adult.png")
        if PLOT_COMPAS:
            analysis_drop_correlated_features(create_compas_dataset, os.path.join("dataset_analysis_plots", "DROP"), "compas.png")
        if PLOT_TITANIC:
            analysis_drop_correlated_features(create_titanic_dataset, os.path.join("dataset_analysis_plots", "DROP"), "titanic.png")
        if PLOT_GERMAN:
            analysis_drop_correlated_features(create_german_dataset, os.path.join("dataset_analysis_plots", "DROP"), "german.png")
        if PLOT_BANK:
            analysis_drop_correlated_features(create_bank_dataset, os.path.join("dataset_analysis_plots", "DROP"), "bank.png")
        if PLOT_COMMUNITIES:
            analysis_drop_correlated_features(create_communities_dataset, os.path.join("dataset_analysis_plots", "DROP"), "communities.png")
    if PLOT_IMPUTE:
        if PLOT_ADULT:
            analysis_impute_correlated_features(create_adult_dataset, os.path.join("dataset_analysis_plots", "IMPUTE"), "adult.png")
        if PLOT_COMPAS:
            analysis_impute_correlated_features(create_compas_dataset, os.path.join("dataset_analysis_plots", "IMPUTE"), "compas.png")
        if PLOT_TITANIC:
            analysis_impute_correlated_features(create_titanic_dataset, os.path.join("dataset_analysis_plots", "IMPUTE"), "titanic.png")
        if PLOT_GERMAN:
            analysis_impute_correlated_features(create_german_dataset, os.path.join("dataset_analysis_plots", "IMPUTE"), "german.png")
        if PLOT_BANK:
            analysis_impute_correlated_features(create_bank_dataset, os.path.join("dataset_analysis_plots", "IMPUTE"), "bank.png")
        if PLOT_COMMUNITIES:
            analysis_impute_correlated_features(create_communities_dataset, os.path.join("dataset_analysis_plots", "IMPUTE"), "communities.png")
    if PLOT_SCATTER:
        if PLOT_ADULT:
            analysis_scatter_correlated_features(create_adult_dataset, os.path.join("dataset_analysis_plots", "CORR"), "adult.png")
        if PLOT_COMPAS:
            analysis_scatter_correlated_features(create_compas_dataset, os.path.join("dataset_analysis_plots", "CORR"), "compas.png")
        if PLOT_TITANIC:
            analysis_scatter_correlated_features(create_titanic_dataset, os.path.join("dataset_analysis_plots", "CORR"), "titanic.png")
        if PLOT_GERMAN:
            analysis_scatter_correlated_features(create_german_dataset, os.path.join("dataset_analysis_plots", "CORR"), "german.png")
        if PLOT_BANK:
            analysis_scatter_correlated_features(create_bank_dataset, os.path.join("dataset_analysis_plots", "CORR"), "bank.png")
        if PLOT_COMMUNITIES:
            analysis_scatter_correlated_features(create_communities_dataset, os.path.join("dataset_analysis_plots", "CORR"), "communities.png")
    if PLOT_DROP_SINGLE:
        if PLOT_ADULT:
            analysis_drop_correlated_features_single(create_adult_dataset, os.path.join("dataset_analysis_plots", "DROP_SINGLE"), "adult")
        if PLOT_COMPAS:
            analysis_drop_correlated_features_single(create_compas_dataset, os.path.join("dataset_analysis_plots", "DROP_SINGLE"), "compas")
        if PLOT_TITANIC:
            analysis_drop_correlated_features_single(create_titanic_dataset, os.path.join("dataset_analysis_plots", "DROP_SINGLE"), "titanic")
        if PLOT_GERMAN:
            analysis_drop_correlated_features_single(create_german_dataset, os.path.join("dataset_analysis_plots", "DROP_SINGLE"), "german")
        if PLOT_BANK:
            analysis_drop_correlated_features_single(create_bank_dataset, os.path.join("dataset_analysis_plots", "DROP_SINGLE"), "bank")
        if PLOT_COMMUNITIES:
            analysis_drop_correlated_features_single(create_communities_dataset, os.path.join("dataset_analysis_plots", "DROP_SINGLE"), "communities")
    if PLOT_DROP_EXCLUDE:
        if PLOT_ADULT:
            analysis_drop_correlated_features_exclude(create_adult_dataset, os.path.join("dataset_analysis_plots", "DROP_EXCLUDE"), "adult")
        if PLOT_COMPAS:
            analysis_drop_correlated_features_exclude(create_compas_dataset, os.path.join("dataset_analysis_plots", "DROP_EXCLUDE"), "compas")
        if PLOT_TITANIC:
            analysis_drop_correlated_features_exclude(create_titanic_dataset, os.path.join("dataset_analysis_plots", "DROP_EXCLUDE"), "titanic")
        if PLOT_GERMAN:
            analysis_drop_correlated_features_exclude(create_german_dataset, os.path.join("dataset_analysis_plots", "DROP_EXCLUDE"), "german")
        if PLOT_BANK:
            analysis_drop_correlated_features_exclude(create_bank_dataset, os.path.join("dataset_analysis_plots", "DROP_EXCLUDE"), "bank")
        if PLOT_COMMUNITIES:
            analysis_drop_correlated_features_exclude(create_communities_dataset, os.path.join("dataset_analysis_plots", "DROP_EXCLUDE"), "communities")