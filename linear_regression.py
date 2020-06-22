# A Linear Regression function to find the coefficients based on ratio analysis outputs
# mainly to find the trade off between bias (bias1 or bias2) and target value (accuracy or f1 score)

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from script_single_task import random_ratios

CREATE_INPUT_DATASET                = False
TRAIN_LINEAR_REGRESSION_TOGETHER    = False
TRAIN_LINEAR_REGRESSION_SEPARATE    = True

iter_per_ratio = 300

classifiers = ["KNN", "LinearSVC", "SVC", "Forest", "LogReg", "Tree", "MLP"]
methods = ["mean_v1", "mean_v2", "similar_v1", "similar_v2", "multi_v1", "multi_v2"]
data_columns = ["random ratio", "iter number", "model name", "method name", "bias1", "bias2", "accuracy", "f1 score"]
model_columns = ["model name", "method name", "beta", "delta1", "delta2", "constant"]

# collect data for models
def prepare_dataset(file_name=None):
    global classifiers, methods, data_columns
    data = []
    for method in methods:
        if not os.path.exists("{}.pkl".format(method)):
            raise Exception("Required pkl not found: {}.pkl".format(method))
        with open("{}.pkl".format(method), "rb") as inFile:
            pkl_data = pickle.load(inFile)
        d_acc   = [x[0] for x in pkl_data]
        d_bias1 = [x[1] for x in pkl_data]
        d_bias2 = [x[2] for x in pkl_data]
        d_f1    = [x[3] for x in pkl_data]
        for i in range(iter_per_ratio):
            for j in range(len(random_ratios)):
                i_acc   = d_acc[i + j * iter_per_ratio]
                i_bias1 = d_bias1[i + j * iter_per_ratio]
                i_bias2 = d_bias2[i + j * iter_per_ratio]
                i_f1    = d_f1[i + j * iter_per_ratio]
                for clf in classifiers:
                    data_processed = [[], [], [], []] # [[acc avg], [bias1], [bias2], [f1 score]], remove -1, [None] cases
                    for x,y,z,w in zip(i_acc[clf], i_bias1[clf], i_bias2[clf], i_f1[clf]):
                        if (y > 0) and (z > 0) and len(w) == 2:
                            data_processed[0].append(x)
                            data_processed[1].append(y)
                            data_processed[2].append(z)
                            data_processed[3].append(np.mean(w))
                    row_data = [round(random_ratios[j], 2), i, clf, method, np.mean(data_processed[1]), np.mean(data_processed[2]), np.mean(data_processed[0]), np.mean(data_processed[3])]
                    data.append(row_data)
    data = pd.DataFrame(data, columns=data_columns)
    if file_name:
        data.to_csv(file_name, index=False)
    return data

# train corresponding models
def train_models(separate=False, save_data=True, save_models=False):
    global classifiers, methods, model_columns
    if not os.path.exists(os.path.join("ratio_analysis_plots", "collected_dataset.csv")):
        raise Exception("Required dataset not created: {}".format(os.path.join("ratio_analysis_plots", "collected_dataset.csv")))
    data = pd.read_csv(os.path.join("ratio_analysis_plots", "collected_dataset.csv"))
    if not separate:
        model_data_acc = {'bias1': [], 'bias2': []}
        coeff_data_acc = {'bias1': [], 'bias2': []}
        model_data_f1 = {'bias1': [], 'bias2': []}
        coeff_data_f1 = {'bias1': [], 'bias2': []}
        for method in methods:
            for clf in classifiers:
                data_copy = data.copy().drop(columns=["random ratio", "iter number"])
                data_copy["model name"] = data_copy["model name"].apply(lambda x: 1 if x == clf else 0)
                data_copy["method name"] = data_copy["method name"].apply(lambda x: 1 if x == method else 0)
                X_train_bias1 = data_copy[["bias1", "model name", "method name"]].to_numpy()
                X_train_bias2 = data_copy[["bias2", "model name", "method name"]].to_numpy()
                y_train_acc = data_copy["accuracy"]
                y_train_f1 = data_copy["f1 score"]
                # train the linear regression for 2 targets individually
                reg1 = LinearRegression()
                reg1.fit(X_train_bias1, y_train_acc)
                coeff_data_acc["bias1"].append([clf, method, reg1.coef_[0], reg1.coef_[1], reg1.coef_[2], reg1.intercept_])
                model_data_acc["bias1"].append([clf, method, reg1])
                reg2 = LinearRegression()
                reg2.fit(X_train_bias1, y_train_f1)
                coeff_data_f1["bias1"].append([clf, method, reg2.coef_[0], reg2.coef_[1], reg2.coef_[2], reg2.intercept_])
                model_data_f1["bias1"].append([clf, method, reg2])
                reg3 = LinearRegression()
                reg3.fit(X_train_bias2, y_train_acc)
                coeff_data_acc["bias2"].append([clf, method, reg3.coef_[0], reg3.coef_[1], reg3.coef_[2], reg3.intercept_])
                model_data_acc["bias2"].append([clf, method, reg3])
                reg4 = LinearRegression()
                reg4.fit(X_train_bias2, y_train_f1)
                coeff_data_f1["bias2"].append([clf, method, reg4.coef_[0], reg4.coef_[1], reg4.coef_[2], reg4.intercept_])
                model_data_f1["bias2"].append([clf, method, reg4])
        if save_data:
            data_acc_bias1 = pd.DataFrame(coeff_data_acc["bias1"], columns=model_columns)
            data_acc_bias2 = pd.DataFrame(coeff_data_acc["bias2"], columns=model_columns)
            data_f1_bias1 = pd.DataFrame(coeff_data_f1["bias1"], columns=model_columns)
            data_f1_bias2 = pd.DataFrame(coeff_data_f1["bias2"], columns=model_columns)
            writer = pd.ExcelWriter(os.path.join("ratio_analysis_plots", "coefficients.xlsx"), engine="xlsxwriter")
            data_acc_bias1.to_excel(writer, sheet_name="acc bias1")
            data_acc_bias2.to_excel(writer, sheet_name="acc bias2")
            data_f1_bias1.to_excel(writer, sheet_name="f1 bias1")
            data_f1_bias2.to_excel(writer, sheet_name="f1 bias2")
            writer.save()
        if save_models:
            with open(os.path.join("ratio_analysis_plots", "models.pkl"), "wb") as outFile:
                pickle.dump([model_data_acc, model_data_f1], outFile)
    else:
        model_data_acc = {'bias1': {}, 'bias2': {}}
        coeff_data_acc = {'bias1': {}, 'bias2': {}}
        model_data_f1 = {'bias1': {}, 'bias2': {}}
        coeff_data_f1 = {'bias1': {}, 'bias2': {}}
        for method in methods:
            model_data_acc["bias1"][method] = []
            model_data_acc["bias2"][method] = []
            coeff_data_acc["bias1"][method] = []
            coeff_data_acc["bias2"][method] = []
            model_data_f1["bias1"][method] = []
            model_data_f1["bias2"][method] = []
            coeff_data_f1["bias1"][method] = []
            coeff_data_f1["bias2"][method] = []
            for clf in classifiers:
                data_copy = data.copy().drop(columns=["random ratio", "iter number"])
                data_copy = data_copy[data_copy["method name"] == method].drop(columns=["method name"])
                data_copy["model name"] = data_copy["model name"].apply(lambda x: 1 if x == clf else 0)
                X_train_bias1 = data_copy[["bias1", "model name"]].to_numpy()
                X_train_bias2 = data_copy[["bias2", "model name"]].to_numpy()
                y_train_acc = data_copy["accuracy"]
                y_train_f1 = data_copy["f1 score"]
                # train the linear regression for 2 targets individually
                reg1 = LinearRegression()
                reg1.fit(X_train_bias1, y_train_acc)
                coeff_data_acc["bias1"][method].append([clf, reg1.coef_[0], reg1.coef_[1], reg1.intercept_])
                model_data_acc["bias1"][method].append([clf, reg1])
                reg2 = LinearRegression()
                reg2.fit(X_train_bias1, y_train_f1)
                coeff_data_f1["bias1"][method].append([clf, reg2.coef_[0], reg2.coef_[1], reg2.intercept_])
                model_data_f1["bias1"][method].append([clf, reg2])
                reg3 = LinearRegression()
                reg3.fit(X_train_bias2, y_train_acc)
                coeff_data_acc["bias2"][method].append([clf, reg3.coef_[0], reg3.coef_[1], reg3.intercept_])
                model_data_acc["bias2"][method].append([clf, reg3])
                reg4 = LinearRegression()
                reg4.fit(X_train_bias2, y_train_f1)
                coeff_data_f1["bias2"][method].append([clf, reg4.coef_[0], reg4.coef_[1], reg4.intercept_])
                model_data_f1["bias2"][method].append([clf, reg4])
        if save_data:
            separate_columns = [x for x in model_columns if x not in ["method name", "delta2"]]
            for method in methods:
                data_acc_bias1 = pd.DataFrame(coeff_data_acc["bias1"][method], columns=separate_columns)
                data_acc_bias2 = pd.DataFrame(coeff_data_acc["bias2"][method], columns=separate_columns)
                data_f1_bias1 = pd.DataFrame(coeff_data_f1["bias1"][method], columns=separate_columns)
                data_f1_bias2 = pd.DataFrame(coeff_data_f1["bias2"][method], columns=separate_columns)
                writer = pd.ExcelWriter(os.path.join("ratio_analysis_plots", "coefficients_{}.xlsx".format(method)), engine="xlsxwriter")
                data_acc_bias1.to_excel(writer, sheet_name="acc bias1")
                data_acc_bias2.to_excel(writer, sheet_name="acc bias2")
                data_f1_bias1.to_excel(writer, sheet_name="f1 bias1")
                data_f1_bias2.to_excel(writer, sheet_name="f1 bias2")
                writer.save()
        if save_models:
            with open(os.path.join("ratio_analysis_plots", "models_separate.pkl"), "wb") as outFile:
                pickle.dump([model_data_acc, model_data_f1], outFile)

if __name__=="__main__":
    if CREATE_INPUT_DATASET:
        print("Creating input dataset")
        prepare_dataset(os.path.join("ratio_analysis_plots", "collected_dataset.csv"))
    if TRAIN_LINEAR_REGRESSION_TOGETHER:
        print("Training linear regression models together")
        train_models()
    if TRAIN_LINEAR_REGRESSION_SEPARATE:
        print("Training linear regression models for each imputation method")
        train_models(separate=True)
