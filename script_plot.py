# script for generating plots from local data

# this script is currently for computing random ratios on compas analysis

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from script_single_task import random_ratios
from script_single_task import RUN_MEAN_V1, RUN_MEAN_V2, RUN_MULTI_V1, RUN_MULTI_V2, RUN_SIMILAR_V1, RUN_SIMILAR_V2
from script_single_task import acc, f1score, bias1, bias2, newBias

iter_per_ratio = 200

NAME_DATASETS = ["adult", "compas", "titanic", "german", "communities", "bank"]
NAME_TARGETS  = ["acc", "f1"]

INCOMPLETE_MODE = False

PLOT_ADULT_ACC          = False
PLOT_ADULT_F1           = False
PLOT_COMPAS_ACC         = False
PLOT_COMPAS_F1          = False
PLOT_TITANIC_ACC        = False
PLOT_TITANIC_F1         = False
PLOT_GERMAN_ACC         = False
PLOT_GERMAN_F1          = False
PLOT_COMMUNITIES_ACC    = False
PLOT_COMMUNITIES_F1     = False
PLOT_BANK_ACC           = False
PLOT_BANK_F1            = False

TRANSFORM_OUTPUTS       = True

PLOT_CREATE_MEAN_V1     = True
PLOT_CREATE_MEAN_V2     = True
PLOT_CREATE_SIMILAR_V1  = True
PLOT_CREATE_SIMILAR_V2  = True
PLOT_CREATE_MULTI_V1    = True
PLOT_CREATE_MULTI_V2    = True

PLOT_PARETO_FRONTIER_ACC     = True
PLOT_PARETO_FRONTIER_F1      = True
PLOT_PARETO_FRONTIER_REALACC = True

PLOT_DEBUG_FUNCTION     = False

def plot_func(data, method_name, file_name=None, yscale=None, plot_error=True):
    if not INCOMPLETE_MODE:
        assert len(data) == (iter_per_ratio * len(random_ratios))
    classifiers = ["KNN", "LinearSVC", "Forest", "LogReg", "Tree", "MLP"]
    plot_bias1 = {}
    plot_bias2 = {}
    plot_acc = {}
    plot_f1 = {}
    plot_realacc = {}
    plot_newbias = {}
    counter = 0
    for clf in classifiers:
        plot_bias1[clf]   = [[], []] # [[actual averaged data], [data for error bar]]
        plot_bias2[clf]   = [[], []]
        plot_acc[clf]     = [[], []]
        plot_f1[clf]      = [[], []]
        plot_realacc[clf] = [[], []]
        plot_newbias[clf] = [[], []]
    iterations = int(len(data) / len(random_ratios))
    for i in range(0, len(data), iterations):
        i_data = data[i:(i+iterations)]
        for clf in classifiers:
            clf_data = [x[clf] for x in i_data]
            data_processed = [[], [], [], [], [], []] # [[acc avg], [bias1], [bias2], [f1 score], [real acc], [new bias]], remove -1, [None] cases
            for cf_matrices in clf_data:
                tmp_data_processed = [[], [], [], [], [], []] #  [[acc each fold], [bias1], [bias2], [f1 score], [real acc], [new bias]], remove -1, [None] cases
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        counter += 1
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_data_processed[0].append(x)
                        tmp_data_processed[1].append(y)
                        tmp_data_processed[2].append(z)
                        tmp_data_processed[3].append(np.mean(w))
                        tmp_data_processed[4].append(acc_m)
                        tmp_data_processed[5].append(k)
                    else:
                        counter += 1
                data_processed[0].append(np.mean(tmp_data_processed[0]))
                data_processed[1].append(np.mean(tmp_data_processed[1]))
                data_processed[2].append(np.mean(tmp_data_processed[2]))
                data_processed[3].append(np.mean(tmp_data_processed[3]))
                data_processed[4].append(np.mean(tmp_data_processed[4]))
                data_processed[5].append(np.mean(tmp_data_processed[5]))
            plot_acc[clf][0].append(np.mean(data_processed[0]))
            plot_acc[clf][1].append(np.std(data_processed[0]))
            plot_bias1[clf][0].append(np.mean(data_processed[1]))
            plot_bias1[clf][1].append(np.std(data_processed[1]))
            plot_bias2[clf][0].append(np.mean(data_processed[2]))
            plot_bias2[clf][1].append(np.std(data_processed[2]))
            plot_f1[clf][0].append(np.mean(data_processed[3]))
            plot_f1[clf][1].append(np.std(data_processed[3]))
            plot_realacc[clf][0].append(np.mean(data_processed[4]))
            plot_realacc[clf][1].append(np.std(data_processed[4]))
            plot_newbias[clf][0].append(np.mean(data_processed[5]))
            plot_newbias[clf][1].append(np.std(data_processed[5]))
    if counter > 0:
        print("Warning: out of {} folds, {} are dropped".format(len(data)*len(classifiers)*10, counter))
    plot_gap = 0.002
    fig, axes = plt.subplots(6, figsize=(10, 30))
    # axes[0] shows bias1
    axes[0].set_title("Bias1")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[0].errorbar(random_ratios+(i-3)*plot_gap, plot_bias1[clf][0], yerr=plot_bias1[clf][1], label=clf)
        else:
            axes[0].plot(random_ratios+(i-3)*plot_gap, plot_bias1[clf][0], label=clf)
        axes[0].scatter(random_ratios+(i-3)*plot_gap, plot_bias1[clf][0], s=2)
    axes[0].legend(loc="best")
    axes[0].set_xticks(np.arange(0.0, 1.0, 0.05))
    if yscale:
        axes[0].set_yscale(yscale)
    # axes[1] shows bias2
    axes[1].set_title("Bias2")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[1].errorbar(random_ratios+(i-3)*plot_gap, plot_bias2[clf][0], yerr=plot_bias2[clf][1], label=clf)
        else:
            axes[1].plot(random_ratios+(i-3)*plot_gap, plot_bias2[clf][0], label=clf)
        axes[1].scatter(random_ratios+(i-3)*plot_gap, plot_bias2[clf][0], s=2)
    axes[1].legend(loc="best")
    axes[1].set_xticks(np.arange(0.0, 1.0, 0.05))
    if yscale:
        axes[1].set_yscale(yscale)
    # axes[2] shows accuracy
    axes[2].set_title("Confusion Matrix Accuracy")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[2].errorbar(random_ratios+(i-3)*plot_gap, plot_acc[clf][0], yerr=plot_acc[clf][1], label=clf)
        else:
            axes[2].plot(random_ratios+(i-3)*plot_gap, plot_acc[clf][0], label=clf)
        axes[2].scatter(random_ratios+(i-3)*plot_gap, plot_acc[clf][0], s=2)
    axes[2].legend(loc="best")
    axes[2].set_xticks(np.arange(0.0, 1.0, 0.05))
    if yscale:
        axes[2].set_yscale(yscale)
    # axes[3] shows f1 score
    axes[3].set_title("Confusion Matrix F1 Score")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[3].errorbar(random_ratios+(i-3)*plot_gap, plot_f1[clf][0], yerr=plot_f1[clf][1], label=clf)
        else:
            axes[3].plot(random_ratios+(i-3)*plot_gap, plot_f1[clf][0], label=clf)
        axes[3].scatter(random_ratios+(i-3)*plot_gap, plot_f1[clf][0], s=2)
    axes[3].legend(loc="best")
    axes[3].set_xticks(np.arange(0.0, 1.0, 0.05))
    if yscale:
        axes[3].set_yscale(yscale)
    # axes[4] show real acc
    axes[4].set_title("Real Accuracy")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[4].errorbar(random_ratios+(i-3)*plot_gap, plot_realacc[clf][0], yerr=plot_realacc[clf][1], label=clf)
        else:
            axes[4].plot(random_ratios+(i-3)*plot_gap, plot_realacc[clf][0], label=clf)
        axes[4].scatter(random_ratios+(i-3)*plot_gap, plot_realacc[clf][0], s=2)
    axes[4].legend(loc="best")
    axes[4].set_xticks(np.arange(0.0, 1.0, 0.05))
    if yscale:
        axes[4].set_yscale(yscale)
    # axes[5] show new bias
    axes[5].set_title("New Bias")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[5].errorbar(random_ratios+(i-3)*plot_gap, plot_newbias[clf][0], yerr=plot_newbias[clf][1], label=clf)
        else:
            axes[5].plot(random_ratios+(i-3)*plot_gap, plot_newbias[clf][0], label=clf)
        axes[5].scatter(random_ratios+(i-3)*plot_gap, plot_newbias[clf][0], s=2)
    axes[5].legend(loc="best")
    axes[5].set_xticks(np.arange(0.0, 1.0, 0.05))
    if yscale:
        axes[5].set_yscale(yscale)
    fig.tight_layout()
    fig.suptitle("Imputation Method: {}".format(method_name))
    plt.subplots_adjust(top=0.94)
    if file_name:
        fig.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def plot_func_pareto_front(data, title, file_name=None, y_scale=None, switch=False, x_axis="acc"):
    # extract data
    data_mean_v1    = data["mean_v1"]
    data_mean_v2    = data["mean_v2"]
    data_similar_v1 = data["similar_v1"]
    data_similar_v2 = data["similar_v2"]
    data_multi_v1   = data["multi_v1"]
    data_multi_v2   = data["multi_v2"]
    classifiers = ["KNN", "LinearSVC", "Forest", "LogReg", "Tree", "MLP"]
    # raise Exception("Update classifiers before run!")
    plot_colors = ["red", "green", "blue", "gold", "darkorange", "grey", "purple"]
    plot_markers = ["o", "s", "*", "^", "P", "v", "X"]
    ratio_dot_size      = [(x+1)*3 for x in range(len(random_ratios))]
    if not INCOMPLETE_MODE:
        assert len(data_mean_v1) == len(data_mean_v2) == len(data_similar_v1) == len(data_similar_v2) == len(data_multi_v1) == len(data_multi_v2) == (iter_per_ratio * len(random_ratios))
    # prepare data for plotting
    plot_data_mean_v1       = {}
    plot_data_mean_v2       = {}
    plot_data_similar_v1    = {}
    plot_data_similar_v2    = {}
    plot_data_multi_v1      = {}
    plot_data_multi_v2      = {}
    for clf in classifiers:
        plot_data_mean_v1[clf]      = ([], [], [], [], [], []) # [[acc_X], [bias1_Y], [bias2_Y], [f1_X], [newbias_Y], [realacc_X]]
        plot_data_mean_v2[clf]      = ([], [], [], [], [], [])
        plot_data_similar_v1[clf]   = ([], [], [], [], [], [])
        plot_data_similar_v2[clf]   = ([], [], [], [], [], [])
        plot_data_multi_v1[clf]     = ([], [], [], [], [], [])
        plot_data_multi_v2[clf]     = ([], [], [], [], [], [])
    iterations = int(len(data_mean_v1) / len(random_ratios))
    for i in range(0, len(data_mean_v1), iterations):
        d_data_mean_v1      = data_mean_v1[i:(i+iterations)]
        d_data_mean_v2      = data_mean_v2[i:(i+iterations)]
        d_data_similar_v1   = data_similar_v1[i:(i+iterations)]
        d_data_similar_v2   = data_similar_v2[i:(i+iterations)]
        d_data_multi_v1     = data_multi_v1[i:(i+iterations)]
        d_data_multi_v2     = data_multi_v2[i:(i+iterations)]
        for clf in classifiers:
            clf_data_mean_v1    = [x[clf] for x in d_data_mean_v1]
            clf_data_mean_v2    = [x[clf] for x in d_data_mean_v2]
            clf_data_similar_v1 = [x[clf] for x in d_data_similar_v1]
            clf_data_similar_v2 = [x[clf] for x in d_data_similar_v2]
            clf_data_multi_v1   = [x[clf] for x in d_data_multi_v1]
            clf_data_multi_v2   = [x[clf] for x in d_data_multi_v2]
            # process mean_v1 method
            data_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
            for cf_matrices in clf_data_mean_v1:
                tmp_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                        tmp_processed[3].append(np.mean(w))
                        tmp_processed[4].append(k)
                        tmp_processed[5].append(acc_m)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
                data_processed[3].append(np.mean(tmp_processed[3]))
                data_processed[4].append(np.mean(tmp_processed[4]))
                data_processed[5].append(np.mean(tmp_processed[5]))
            plot_data_mean_v1[clf][0].append(np.mean(data_processed[0]))
            plot_data_mean_v1[clf][1].append(np.mean(data_processed[1]))
            plot_data_mean_v1[clf][2].append(np.mean(data_processed[2]))
            plot_data_mean_v1[clf][3].append(np.mean(data_processed[3]))
            plot_data_mean_v1[clf][4].append(np.mean(data_processed[4]))
            plot_data_mean_v1[clf][5].append(np.mean(data_processed[5]))
            # process mean_v2 method
            data_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
            for cf_matrices in clf_data_mean_v2:
                tmp_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                        tmp_processed[3].append(np.mean(w))
                        tmp_processed[4].append(k)
                        tmp_processed[5].append(acc_m)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
                data_processed[3].append(np.mean(tmp_processed[3]))
                data_processed[4].append(np.mean(tmp_processed[4]))
                data_processed[5].append(np.mean(tmp_processed[5]))
            plot_data_mean_v2[clf][0].append(np.mean(data_processed[0]))
            plot_data_mean_v2[clf][1].append(np.mean(data_processed[1]))
            plot_data_mean_v2[clf][2].append(np.mean(data_processed[2]))
            plot_data_mean_v2[clf][3].append(np.mean(data_processed[3]))
            plot_data_mean_v2[clf][4].append(np.mean(data_processed[4]))
            plot_data_mean_v2[clf][5].append(np.mean(data_processed[5]))
            # process similar_v1 method
            data_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
            for cf_matrices in clf_data_similar_v1:
                tmp_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                        tmp_processed[3].append(np.mean(w))
                        tmp_processed[4].append(k)
                        tmp_processed[5].append(acc_m)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
                data_processed[3].append(np.mean(tmp_processed[3]))
                data_processed[4].append(np.mean(tmp_processed[4]))
                data_processed[5].append(np.mean(tmp_processed[5]))
            plot_data_similar_v1[clf][0].append(np.mean(data_processed[0]))
            plot_data_similar_v1[clf][1].append(np.mean(data_processed[1]))
            plot_data_similar_v1[clf][2].append(np.mean(data_processed[2]))
            plot_data_similar_v1[clf][3].append(np.mean(data_processed[3]))
            plot_data_similar_v1[clf][4].append(np.mean(data_processed[4]))
            plot_data_similar_v1[clf][5].append(np.mean(data_processed[5]))
            # process similar_v2 method
            data_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
            for cf_matrices in clf_data_similar_v2:
                tmp_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                        tmp_processed[3].append(np.mean(w))
                        tmp_processed[4].append(k)
                        tmp_processed[5].append(acc_m)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
                data_processed[3].append(np.mean(tmp_processed[3]))
                data_processed[4].append(np.mean(tmp_processed[4]))
                data_processed[5].append(np.mean(tmp_processed[5]))
            plot_data_similar_v2[clf][0].append(np.mean(data_processed[0]))
            plot_data_similar_v2[clf][1].append(np.mean(data_processed[1]))
            plot_data_similar_v2[clf][2].append(np.mean(data_processed[2]))
            plot_data_similar_v2[clf][3].append(np.mean(data_processed[3]))
            plot_data_similar_v2[clf][4].append(np.mean(data_processed[4]))
            plot_data_similar_v2[clf][5].append(np.mean(data_processed[5]))
            # process multi_v1 method
            data_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
            for cf_matrices in clf_data_multi_v1:
                tmp_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                        tmp_processed[3].append(np.mean(w))
                        tmp_processed[4].append(k)
                        tmp_processed[5].append(acc_m)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
                data_processed[3].append(np.mean(tmp_processed[3]))
                data_processed[4].append(np.mean(tmp_processed[4]))
                data_processed[5].append(np.mean(tmp_processed[5]))
            plot_data_multi_v1[clf][0].append(np.mean(data_processed[0]))
            plot_data_multi_v1[clf][1].append(np.mean(data_processed[1]))
            plot_data_multi_v1[clf][2].append(np.mean(data_processed[2]))
            plot_data_multi_v1[clf][3].append(np.mean(data_processed[3]))
            plot_data_multi_v1[clf][4].append(np.mean(data_processed[4]))
            plot_data_multi_v1[clf][5].append(np.mean(data_processed[5]))
            # process multi_v2 method
            data_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
            for cf_matrices in clf_data_multi_v2:
                tmp_processed = [[], [], [], [], [], []] # [[acc], [bias1], [bias2], [f1 score], [new bias], [real acc]]
                for mm in cf_matrices:
                    if len(mm) < 1:
                        continue
                    cf_m, acc_m = mm[0], mm[1]
                    try:
                        x = acc(cf_m)
                        y = bias1(cf_m)
                        z = bias2(cf_m)
                        w = f1score(cf_m)
                        k = newBias(cf_m)
                    except Exception as e:
                        print("Error: {}".format(e))
                        continue
                    if (y > 0) and (z > 0) and len(w) == 2:
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                        tmp_processed[3].append(np.mean(w))
                        tmp_processed[4].append(k)
                        tmp_processed[5].append(acc_m)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
                data_processed[3].append(np.mean(tmp_processed[3]))
                data_processed[4].append(np.mean(tmp_processed[4]))
                data_processed[5].append(np.mean(tmp_processed[5]))
            plot_data_multi_v2[clf][0].append(np.mean(data_processed[0]))
            plot_data_multi_v2[clf][1].append(np.mean(data_processed[1]))
            plot_data_multi_v2[clf][2].append(np.mean(data_processed[2]))
            plot_data_multi_v2[clf][3].append(np.mean(data_processed[3]))
            plot_data_multi_v2[clf][4].append(np.mean(data_processed[4]))
            plot_data_multi_v2[clf][5].append(np.mean(data_processed[5]))
    fig, axes = plt.subplots(3, figsize=(8, 18)) # axes[0] for bias1, axes[1] for bias2, axes[2] for new bias
    if x_axis == "acc":
        axes[0].set_xlabel("Confusion Matrix Accuracy")
        axes[1].set_xlabel("Confusion Matrix Accuracy")
        axes[2].set_xlabel("Confusion Matrix Accuracy")
    elif x_axis == "f1":
        axes[0].set_xlabel("Confusion Matrix F1 Score")
        axes[1].set_xlabel("Confusion Matrix F1 Score")
        axes[2].set_xlabel("Confusion Matrix F1 Score")
    else:
        axes[0].set_xlabel("Real Accuracy")
        axes[1].set_xlabel("Real Accuracy")
        axes[2].set_xlabel("Real Accuracy")
    axes[0].set_ylabel("Bias1 Values")
    axes[1].set_ylabel("Bias2 Values")
    axes[2].set_ylabel("New Bias Values")
    # each classifier has different color
    for clf, clf_c, clf_m in zip(classifiers, plot_colors, plot_markers):
        if switch:
            # plot for mean_v1 method
            axes[0].scatter(plot_data_mean_v1[clf][0] if x_axis == "acc" else plot_data_mean_v1[clf][3] if x_axis == "f1" else plot_data_mean_v1[clf][5], plot_data_mean_v1[clf][1], c=plot_colors[0], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_mean_v1[clf][0] if x_axis == "acc" else plot_data_mean_v1[clf][3] if x_axis == "f1" else plot_data_mean_v1[clf][5], plot_data_mean_v1[clf][2], c=plot_colors[0], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[2].scatter(plot_data_mean_v1[clf][0] if x_axis == "acc" else plot_data_mean_v1[clf][3] if x_axis == "f1" else plot_data_mean_v1[clf][5], plot_data_mean_v1[clf][4], c=plot_colors[0], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for mean_v2 method
            axes[0].scatter(plot_data_mean_v2[clf][0] if x_axis == "acc" else plot_data_mean_v2[clf][3] if x_axis == "f1" else plot_data_mean_v2[clf][5], plot_data_mean_v2[clf][1], c=plot_colors[1], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_mean_v2[clf][0] if x_axis == "acc" else plot_data_mean_v2[clf][3] if x_axis == "f1" else plot_data_mean_v2[clf][5], plot_data_mean_v2[clf][2], c=plot_colors[1], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[2].scatter(plot_data_mean_v2[clf][0] if x_axis == "acc" else plot_data_mean_v2[clf][3] if x_axis == "f1" else plot_data_mean_v2[clf][5], plot_data_mean_v2[clf][4], c=plot_colors[1], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for similar_v1 method
            axes[0].scatter(plot_data_similar_v1[clf][0] if x_axis == "acc" else plot_data_similar_v1[clf][3] if x_axis == "f1" else plot_data_similar_v1[clf][5], plot_data_similar_v1[clf][1], c=plot_colors[2], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_similar_v1[clf][0] if x_axis == "acc" else plot_data_similar_v1[clf][3] if x_axis == "f1" else plot_data_similar_v1[clf][5], plot_data_similar_v1[clf][2], c=plot_colors[2], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[2].scatter(plot_data_similar_v1[clf][0] if x_axis == "acc" else plot_data_similar_v1[clf][3] if x_axis == "f1" else plot_data_similar_v1[clf][5], plot_data_similar_v1[clf][4], c=plot_colors[2], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for similar_v2 method
            axes[0].scatter(plot_data_similar_v2[clf][0] if x_axis == "acc" else plot_data_similar_v2[clf][3] if x_axis == "f1" else plot_data_similar_v2[clf][5], plot_data_similar_v2[clf][1], c=plot_colors[3], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_similar_v2[clf][0] if x_axis == "acc" else plot_data_similar_v2[clf][3] if x_axis == "f1" else plot_data_similar_v2[clf][5], plot_data_similar_v2[clf][2], c=plot_colors[3], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[2].scatter(plot_data_similar_v2[clf][0] if x_axis == "acc" else plot_data_similar_v2[clf][3] if x_axis == "f1" else plot_data_similar_v2[clf][5], plot_data_similar_v2[clf][4], c=plot_colors[3], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for multi_v1 method
            axes[0].scatter(plot_data_multi_v1[clf][0] if x_axis == "acc" else plot_data_multi_v1[clf][3] if x_axis == "f1" else plot_data_multi_v1[clf][5], plot_data_multi_v1[clf][1], c=plot_colors[4], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_multi_v1[clf][0] if x_axis == "acc" else plot_data_multi_v1[clf][3] if x_axis == "f1" else plot_data_multi_v1[clf][5], plot_data_multi_v1[clf][2], c=plot_colors[4], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[2].scatter(plot_data_multi_v1[clf][0] if x_axis == "acc" else plot_data_multi_v1[clf][3] if x_axis == "f1" else plot_data_multi_v1[clf][5], plot_data_multi_v1[clf][4], c=plot_colors[4], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for multi_v2 method
            axes[0].scatter(plot_data_multi_v2[clf][0] if x_axis == "acc" else plot_data_multi_v2[clf][3] if x_axis == "f1" else plot_data_multi_v2[clf][5], plot_data_multi_v2[clf][1], c=plot_colors[5], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_multi_v2[clf][0] if x_axis == "acc" else plot_data_multi_v2[clf][3] if x_axis == "f1" else plot_data_multi_v2[clf][5], plot_data_multi_v2[clf][2], c=plot_colors[5], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[2].scatter(plot_data_multi_v2[clf][0] if x_axis == "acc" else plot_data_multi_v2[clf][3] if x_axis == "f1" else plot_data_multi_v2[clf][5], plot_data_multi_v2[clf][4], c=plot_colors[5], s=ratio_dot_size, marker=clf_m, alpha=0.8)
        else:
            # plot for mean_v1 method
            axes[0].scatter(plot_data_mean_v1[clf][0] if x_axis == "acc" else plot_data_mean_v1[clf][3] if x_axis == "f1" else plot_data_mean_v1[clf][5], plot_data_mean_v1[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[0], alpha=0.8)
            axes[1].scatter(plot_data_mean_v1[clf][0] if x_axis == "acc" else plot_data_mean_v1[clf][3] if x_axis == "f1" else plot_data_mean_v1[clf][5], plot_data_mean_v1[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[0], alpha=0.8)
            axes[2].scatter(plot_data_mean_v1[clf][0] if x_axis == "acc" else plot_data_mean_v1[clf][3] if x_axis == "f1" else plot_data_mean_v1[clf][5], plot_data_mean_v1[clf][4], c=clf_c, s=ratio_dot_size, marker=plot_markers[0], alpha=0.8)
            # plot for mean_v2 method
            axes[0].scatter(plot_data_mean_v2[clf][0] if x_axis == "acc" else plot_data_mean_v2[clf][3] if x_axis == "f1" else plot_data_mean_v2[clf][5], plot_data_mean_v2[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[1], alpha=0.8)
            axes[1].scatter(plot_data_mean_v2[clf][0] if x_axis == "acc" else plot_data_mean_v2[clf][3] if x_axis == "f1" else plot_data_mean_v2[clf][5], plot_data_mean_v2[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[1], alpha=0.8)
            axes[2].scatter(plot_data_mean_v2[clf][0] if x_axis == "acc" else plot_data_mean_v2[clf][3] if x_axis == "f1" else plot_data_mean_v2[clf][5], plot_data_mean_v2[clf][4], c=clf_c, s=ratio_dot_size, marker=plot_markers[1], alpha=0.8)
            # plot for similar_v1 method
            axes[0].scatter(plot_data_similar_v1[clf][0] if x_axis == "acc" else plot_data_similar_v1[clf][3] if x_axis == "f1" else plot_data_similar_v1[clf][5], plot_data_similar_v1[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[2], alpha=0.8)
            axes[1].scatter(plot_data_similar_v1[clf][0] if x_axis == "acc" else plot_data_similar_v1[clf][3] if x_axis == "f1" else plot_data_similar_v1[clf][5], plot_data_similar_v1[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[2], alpha=0.8)
            axes[2].scatter(plot_data_similar_v1[clf][0] if x_axis == "acc" else plot_data_similar_v1[clf][3] if x_axis == "f1" else plot_data_similar_v1[clf][5], plot_data_similar_v1[clf][4], c=clf_c, s=ratio_dot_size, marker=plot_markers[2], alpha=0.8)
            # plot for similar_v2 method
            axes[0].scatter(plot_data_similar_v2[clf][0] if x_axis == "acc" else plot_data_similar_v2[clf][3] if x_axis == "f1" else plot_data_similar_v2[clf][5], plot_data_similar_v2[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[3], alpha=0.8)
            axes[1].scatter(plot_data_similar_v2[clf][0] if x_axis == "acc" else plot_data_similar_v2[clf][3] if x_axis == "f1" else plot_data_similar_v2[clf][5], plot_data_similar_v2[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[3], alpha=0.8)
            axes[2].scatter(plot_data_similar_v2[clf][0] if x_axis == "acc" else plot_data_similar_v2[clf][3] if x_axis == "f1" else plot_data_similar_v2[clf][5], plot_data_similar_v2[clf][4], c=clf_c, s=ratio_dot_size, marker=plot_markers[3], alpha=0.8)
            # plot for multi_v1 method
            axes[0].scatter(plot_data_multi_v1[clf][0] if x_axis == "acc" else plot_data_multi_v1[clf][3] if x_axis == "f1" else plot_data_multi_v1[clf][5], plot_data_multi_v1[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[4], alpha=0.8)
            axes[1].scatter(plot_data_multi_v1[clf][0] if x_axis == "acc" else plot_data_multi_v1[clf][3] if x_axis == "f1" else plot_data_multi_v1[clf][5], plot_data_multi_v1[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[4], alpha=0.8)
            axes[2].scatter(plot_data_multi_v1[clf][0] if x_axis == "acc" else plot_data_multi_v1[clf][3] if x_axis == "f1" else plot_data_multi_v1[clf][5], plot_data_multi_v1[clf][4], c=clf_c, s=ratio_dot_size, marker=plot_markers[4], alpha=0.8)
            # plot for multi_v2 method
            axes[0].scatter(plot_data_multi_v2[clf][0] if x_axis == "acc" else plot_data_multi_v2[clf][3] if x_axis == "f1" else plot_data_multi_v2[clf][5], plot_data_multi_v2[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[5], alpha=0.8)
            axes[1].scatter(plot_data_multi_v2[clf][0] if x_axis == "acc" else plot_data_multi_v2[clf][3] if x_axis == "f1" else plot_data_multi_v2[clf][5], plot_data_multi_v2[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[5], alpha=0.8)
            axes[2].scatter(plot_data_multi_v2[clf][0] if x_axis == "acc" else plot_data_multi_v2[clf][3] if x_axis == "f1" else plot_data_multi_v2[clf][5], plot_data_multi_v2[clf][4], c=clf_c, s=ratio_dot_size, marker=plot_markers[5], alpha=0.8)
    if y_scale:
        axes[0].set_yscale(y_scale)
        axes[1].set_yscale(y_scale)
        axes[2].set_yscale(y_scale)
    if switch:
        custom_legend = [Line2D([0], [0], color='w', markerfacecolor="black", marker=x, label=y, markersize=10) for x,y in zip(plot_markers, classifiers)]
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[0], marker="o", label="mean_v1"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[1], marker="o", label="mean_v2"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[2], marker="o", label="similar_v1"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[3], marker="o", label="similar_v2"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[4], marker="o", label="multi_v1"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[5], marker="o", label="multi_v2"))
    else:
        custom_legend = [Line2D([0], [0], markerfacecolor=x, marker="o", label=y, markersize=10) for x,y in zip(plot_colors, classifiers)]
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[0], label="mean_v1"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[1], label="mean_v2"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[2], label="similar_v1"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[3], label="similar_v2"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[4], label="multi_v1"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[5], label="multi_v2"))
    plt.legend(handles=custom_legend, bbox_to_anchor=(1, 0.6))
    fig.tight_layout()
    fig.suptitle(title)
    plt.subplots_adjust(top=0.94)
    if file_name:
        plt.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def compress_outputs():
    for tt in NAME_TARGETS:
        for dd in NAME_DATASETS:
            if not os.path.exists(os.path.join("condor_outputs", tt, dd)):
                print("Folder not found: {}".format(os.path.join("condor_outputs", tt, dd)))
                continue
            final_results = {}
            if RUN_MEAN_V1: final_results["mean_v1"] = [[] for _ in range(len(random_ratios))]
            if RUN_MEAN_V2: final_results["mean_v2"] = [[] for _ in range(len(random_ratios))]
            if RUN_SIMILAR_V1: final_results["similar_v1"] = [[] for _ in range(len(random_ratios))]
            if RUN_SIMILAR_V2: final_results["similar_v2"] = [[] for _ in range(len(random_ratios))]
            if RUN_MULTI_V1: final_results["multi_v1"] = [[] for _ in range(len(random_ratios))]
            if RUN_MULTI_V2: final_results["multi_v2"] = [[] for _ in range(len(random_ratios))]
            load_complete = True
            need_reload = False
            for key in final_results.keys():
                if not os.path.exists(os.path.join("condor_outputs", tt, dd, "{}.pkl".format(key))):
                    need_reload = True
            if not need_reload:
                continue
            # load data
            for i in range(iter_per_ratio):
                if not os.path.exists(os.path.join("condor_outputs", tt, dd, "output_{:0>4}.pkl".format(i))):
                    if INCOMPLETE_MODE:
                        continue
                    else:
                        print("{} file not found, will skip this folder".format(os.path.join("condor_outputs", tt, dd, "output_{:0>4}.pkl".format(i))))
                        load_complete = False
                        break
                with open(os.path.join("condor_outputs", tt, dd, "output_{:0>4}.pkl".format(i)), "rb") as inFile:
                    output_data = pickle.load(inFile)
                for key, value in output_data.items():
                    assert key in final_results.keys()
                    for j in range(len(random_ratios)):
                        final_results[key][j].append(value[j])
            # dump data
            if load_complete:
                for key in final_results.keys():
                    if INCOMPLETE_MODE and len(final_results[key][0]) < 1:
                        continue
                    dump_data = final_results[key]
                    dump_data = [i for m in dump_data for i in m] # unpack list of list
                    with open(os.path.join("condor_outputs", tt, dd, "{}.pkl".format(key)), "wb") as outFile:
                        pickle.dump(dump_data, outFile)

def plot_debug_func(file_name=None):
    # data = [TN_AA, FP_AA, FN_AA, TP_AA, TN_C, FP_C, FN_C, TP_C]
    def helper_bias1(data):
        # |(FPR_AA/FNR_AA) - (FPR_C/FNR_C)|
        FPR_AA = data[1] / (data[1] + data[0])
        FNR_AA = data[2] / (data[2] + data[3])
        FPR_C  = data[5] / (data[5] + data[4])
        FNR_C  = data[6] / (data[6] + data[7])
        if FNR_AA == 0 or FNR_C == 0: return [-1] # mark error situation
        bias = (FPR_AA / FNR_AA) - (FPR_C / FNR_C)
        return [abs(bias)]
    def helper_bias2(data):
        # |(FPR_AA/FPR_C) - (FNR_AA/FNR_C)|
        FPR_AA = data[1] / (data[1] + data[0])
        FNR_AA = data[2] / (data[2] + data[3])
        FPR_C  = data[5] / (data[5] + data[4])
        FNR_C  = data[6] / (data[6] + data[7])
        if FNR_C == 0 or FPR_C == 0: return [-1] # mark error situation
        bias = (FPR_AA / FPR_C) - (FNR_AA / FNR_C)
        return [abs(bias)]
    def helper_acc(data):
        # (TP + TN) / (TP + TN + FP + FN)
        acc_AA  = (data[3] + data[0]) / (data[0] + data[1] + data[2] + data[3])
        acc_C   = (data[7] + data[4]) / (data[4] + data[5] + data[6] + data[7])
        acc_all = (data[3] + data[0] + data[7] + data[4]) / sum(data)
        return [acc_AA, acc_C, acc_all]
    def helper_precision(data):
        # TP / (TP + FP)
        p_AA = data[3] / (data[3] + data[1]) if (data[3] + data[1]) != 0 else False
        p_C  = data[7] / (data[7] + data[5]) if (data[7] + data[5]) != 0 else False
        return [p_AA, p_C]
    def helper_recall(data):
        # TP / (TP + FN)
        r_AA = data[3] / (data[3] + data[2])
        r_C  = data[7] / (data[7] + data[6])
        return [r_AA, r_C]
    def helper_f1(data):
        # f1 score  = 2 * (precision * recall) / (recall + precision)
        precision_AA = data[3] / (data[3] + data[1]) if (data[3] + data[1]) != 0 else 0
        precision_C  = data[7] / (data[7] + data[5]) if (data[7] + data[5]) != 0 else 0
        recall_AA    = data[3] / (data[3] + data[2])
        recall_C     = data[7] / (data[7] + data[6])
        if (recall_AA + precision_AA) == 0 or (recall_C + precision_C) == 0:
            return [False, False] # mark error situation
        f1_AA        = 2 * (precision_AA * recall_AA) / (recall_AA + precision_AA)
        f1_C         = 2 * (precision_C * recall_C) / (recall_C + precision_C)
        return [f1_AA, f1_C]
    classifiers = ["KNN", "LinearSVC", "Forest", "LogReg", "Tree", "MLP"]
    if not os.path.exists("debug_data.pkl"):
        raise FileNotFoundError("debug_data.pkl not found")
    with open("debug_data.pkl", "rb") as inFile:
        debug_data = pickle.load(inFile)
        assert len(debug_data) == len(random_ratios)
    fig, axes = plt.subplots(12, len(classifiers), figsize=(8*len(classifiers), 4*12))
    plot_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i in range(len(classifiers)):
        clf = classifiers[i]
        clf_data = [x[clf] for x in debug_data]
        plot_data = {
            "bias1": [],
            "bias2": [],
            "acc_AA": [],
            "acc_C": [],
            "acc_all": [],
            "precision_AA": [],
            "recall_AA": [],
            "f1_AA": [],
            "precision_C": [],
            "recall_C": [],
            "f1_C": [],
            "f1_avg": []
        }
        for mm in range(len(random_ratios)):
            bias1, bias2, acc_AA, acc_C, acc_all, p_AA, p_C, r_AA, r_C, f1_AA, f1_C, f1_avg = [],[],[],[],[],[],[],[],[],[],[],[]
            for dd in clf_data[mm]:
                result = [
                    helper_bias1(dd),
                    helper_bias2(dd),
                    helper_acc(dd),
                    helper_precision(dd),
                    helper_recall(dd),
                    helper_f1(dd)
                ]
                result = [ss for tt in result for ss in tt]
                if result[0] > 0 and result[1] > 0 and result[5] and result[6] and result[9] and result[10]:
                    bias1.append(result[0]); bias2.append(result[1])
                    acc_AA.append(result[2]); acc_C.append(result[3]); acc_all.append(result[4])
                    p_AA.append(result[5]); p_C.append(result[6])
                    r_AA.append(result[7]); r_C.append(result[8])
                    f1_AA.append(result[9]); f1_C.append(result[10]); f1_avg.append((result[9]+result[10])/2)
            plot_data["bias1"].append(np.mean(bias1)); plot_data["bias2"].append(np.mean(bias2))
            plot_data["acc_AA"].append(np.mean(acc_AA)); plot_data["acc_C"].append(np.mean(acc_C)); plot_data["acc_all"].append(np.mean(acc_all))
            plot_data["precision_AA"].append(np.mean(p_AA)); plot_data["precision_C"].append(np.mean(p_C))
            plot_data["recall_AA"].append(np.mean(r_AA)); plot_data["recall_C"].append(np.mean(r_C))
            plot_data["f1_AA"].append(np.mean(f1_AA)); plot_data["f1_C"].append(np.mean(f1_C)); plot_data["f1_avg"].append(np.mean(f1_avg))
        for j, name in enumerate(list(plot_data.keys())):
            axes[j, i].plot(random_ratios, plot_data[name], label=clf, color=plot_colors[i])
            axes[j, i].scatter(random_ratios, plot_data[name], s=3)
            axes[j, i].set_ylabel(name)
            axes[j, i].set_xticks(np.arange(0.0, 1.0, 0.05))
            axes[j, i].legend(loc="best")
    fig.tight_layout()
    fig.suptitle("Debug Plot")
    plt.subplots_adjust(top=0.96)
    if file_name:
        plt.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_all(data_folder, plot_folder, name):
    # generate plot for mean_v1.pkl
    if os.path.exists(os.path.join(data_folder, "mean_v1.pkl")) and PLOT_CREATE_MEAN_V1:
        print("Generating plot for mean_v1.pkl ({})".format(name))
        with open(os.path.join(data_folder, "mean_v1.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Mean V1 ({})".format(name), os.path.join(plot_folder, "ratio_mean_v1.png"))
        plot_func(data, "Complete by Mean V1 ({})".format(name), os.path.join(plot_folder, "ratio_mean_v1_scaled.png"), yscale="log")
    # generate plot for mean_v2.pkl
    if os.path.exists(os.path.join(data_folder, "mean_v2.pkl")) and PLOT_CREATE_MEAN_V1:
        print("Generating plot for mean_v2.pkl ({})".format(name))
        with open(os.path.join(data_folder, "mean_v2.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Mean V2 ({})".format(name), os.path.join(plot_folder, "ratio_mean_v2.png"))
        plot_func(data, "Complete by Mean V2 ({})".format(name), os.path.join(plot_folder, "ratio_mean_v2_scaled.png"), yscale="log")
    # generate plot for similar_v1.pkl
    if os.path.exists(os.path.join(data_folder, "similar_v1.pkl")) and PLOT_CREATE_SIMILAR_V1:
        print("Generating plot for similar_v1.pkl ({})".format(name))
        with open(os.path.join(data_folder, "similar_v1.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Similar V1 ({})".format(name), os.path.join(plot_folder, "ratio_similar_v1.png"))
        plot_func(data, "Complete by Similar V1 ({})".format(name), os.path.join(plot_folder, "ratio_similar_v1_scaled.png"), yscale="log")
    # generate plot for similar_v2.pkl
    if os.path.exists(os.path.join(data_folder, "similar_v2.pkl")) and PLOT_CREATE_SIMILAR_V2:
        print("Generating plot for similar_v2.pkl ({})".format(name))
        with open(os.path.join(data_folder, "similar_v2.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Similar V2 ({})".format(name), os.path.join(plot_folder, "ratio_similar_v2.png"))
        plot_func(data, "Complete by Similar V2 ({})".format(name), os.path.join(plot_folder, "ratio_similar_v2_scaled.png"), yscale="log")
    # generate plot for multi_v1.pkl
    if os.path.exists(os.path.join(data_folder, "multi_v1.pkl")) and PLOT_CREATE_MULTI_V1:
        print("Generating plot for multi_v1.pkl ({})".format(name))
        with open(os.path.join(data_folder, "multi_v1.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Multiple Imputation V1 ({})".format(name), os.path.join(plot_folder, "ratio_multi_v1.png"))
        plot_func(data, "Complete by Multiple Imputation V1 ({})".format(name), os.path.join(plot_folder, "ratio_multi_v1_scaled.png"), yscale="log")
    # generate plot for multi_v2.pkl
    if os.path.exists(os.path.join(data_folder, "multi_v2.pkl")) and PLOT_CREATE_MULTI_V1:
        print("Generating plot for multi_v2.pkl ({})".format(name))
        with open(os.path.join(data_folder, "multi_v2.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Multiple Imputation V2 ({})".format(name), os.path.join(plot_folder, "ratio_multi_v2.png"))
        plot_func(data, "Complete by Multiple Imputation V2 ({})".format(name), os.path.join(plot_folder, "ratio_multi_v2_scaled.png"), yscale="log")
    # generate pareto front plots
    if  os.path.exists(os.path.join(data_folder, "mean_v1.pkl")) and \
        os.path.exists(os.path.join(data_folder, "mean_v2.pkl")) and \
        os.path.exists(os.path.join(data_folder, "similar_v1.pkl")) and \
        os.path.exists(os.path.join(data_folder, "similar_v2.pkl")) and \
        os.path.exists(os.path.join(data_folder, "multi_v1.pkl")) and \
        os.path.exists(os.path.join(data_folder, "multi_v2.pkl")) and \
        (PLOT_PARETO_FRONTIER_ACC or PLOT_PARETO_FRONTIER_F1 or PLOT_PARETO_FRONTIER_REALACC):
        data = {}
        with open(os.path.join(data_folder, "mean_v1.pkl"), "rb") as inFile:
            data["mean_v1"] = pickle.load(inFile)
        with open(os.path.join(data_folder, "mean_v2.pkl"), "rb") as inFile:
            data["mean_v2"] = pickle.load(inFile)
        with open(os.path.join(data_folder, "similar_v1.pkl"), "rb") as inFile:
            data["similar_v1"] = pickle.load(inFile)
        with open(os.path.join(data_folder, "similar_v2.pkl"), "rb") as inFile:
            data["similar_v2"] = pickle.load(inFile)
        with open(os.path.join(data_folder, "multi_v1.pkl"), "rb") as inFile:
            data["multi_v1"] = pickle.load(inFile)
        with open(os.path.join(data_folder, "multi_v2.pkl"), "rb") as inFile:
            data["multi_v2"] = pickle.load(inFile)
        if PLOT_PARETO_FRONTIER_ACC:
            print("Generate plots for pareto front acc ({})".format(name))
            plot_func_pareto_front(data, "Pareto Front (Confusion Matrix Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_acc.png"), x_axis="acc")
            plot_func_pareto_front(data, "Pareto Front (Confusion Matrix Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_acc_scaled.png"), y_scale="log", x_axis="acc")
        if PLOT_PARETO_FRONTIER_F1:
            print("Generate plots for pareto front f1 ({})".format(name))
            plot_func_pareto_front(data, "Pareto Front (Confusion Matrix F1 Score) ({})".format(name), os.path.join(plot_folder, "pareto_front_f1.png"), x_axis="f1")
            plot_func_pareto_front(data, "Pareto Front (Confusion Matrix F1 Score) ({})".format(name), os.path.join(plot_folder, "pareto_front_f1_scaled.png"), y_scale="log", x_axis="f1")
        if PLOT_PARETO_FRONTIER_REALACC:
            print("Generate plots for pareto front real acc ({})".format(name))
            plot_func_pareto_front(data, "Pareto Front (Real Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_realacc.png"), x_axis="realacc")
            plot_func_pareto_front(data, "Pareto Front (Real Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_realacc_scaled.png"), y_scale="log", x_axis="realacc")

if __name__=="__main__":
    if not os.path.exists("ratio_analysis_plots"):
        os.makedirs("ratio_analysis_plots")
    for tt in NAME_TARGETS:
        if not os.path.exists(os.path.join("ratio_analysis_plots", tt)):
            os.makedirs(os.path.join("ratio_analysis_plots", tt))
        for dd in NAME_DATASETS:
            if not os.path.exists(os.path.join("ratio_analysis_plots", tt, dd)):
                os.makedirs(os.path.join("ratio_analysis_plots", tt, dd))

    if TRANSFORM_OUTPUTS:
        compress_outputs()

    if PLOT_DEBUG_FUNCTION:
        plot_debug_func(file_name=os.path.join("ratio_analysis_plots", "debug_plot_mean_v1.png"))

    if PLOT_ADULT_ACC:
        plot_all(os.path.join("condor_outputs", "acc", "adult"), os.path.join("ratio_analysis_plots", "acc", "adult"), "adult acc")
    if PLOT_ADULT_F1:
        plot_all(os.path.join("condor_outputs", "f1", "adult"), os.path.join("ratio_analysis_plots", "f1", "adult"), "adult f1")
    if PLOT_COMPAS_ACC:
        plot_all(os.path.join("condor_outputs", "acc", "compas"), os.path.join("ratio_analysis_plots", "acc", "compas"), "compas acc")
    if PLOT_COMPAS_F1:
        plot_all(os.path.join("condor_outputs", "f1", "compas"), os.path.join("ratio_analysis_plots", "f1", "compas"), "compas f1")
    if PLOT_TITANIC_ACC:
        plot_all(os.path.join("condor_outputs", "acc", "titanic"), os.path.join("ratio_analysis_plots", "acc", "titanic"), "titanic acc")
    if PLOT_TITANIC_F1:
        plot_all(os.path.join("condor_outputs", "f1", "titanic"), os.path.join("ratio_analysis_plots", "f1", "titanic"), "titanic f1")
    if PLOT_GERMAN_ACC:
        plot_all(os.path.join("condor_outputs", "acc", "german"), os.path.join("ratio_analysis_plots", "acc", "german"), "german acc")
    if PLOT_GERMAN_F1:
        plot_all(os.path.join("condor_outputs", "f1", "german"), os.path.join("ratio_analysis_plots", "f1", "german"), "german f1")
    if PLOT_COMMUNITIES_ACC:
        plot_all(os.path.join("condor_outputs", "acc", "communities"), os.path.join("ratio_analysis_plots", "acc", "communities"), "communities acc")
    if PLOT_COMMUNITIES_F1:
        plot_all(os.path.join("condor_outputs", "f1", "communities"), os.path.join("ratio_analysis_plots", "f1", "communities"), "communities f1")
    if PLOT_BANK_ACC:
        plot_all(os.path.join("condor_outputs", "acc", "bank"), os.path.join("ratio_analysis_plots", "acc", "bank"), "bank acc")
    if PLOT_BANK_F1:
        plot_all(os.path.join("condor_outputs", "f1", "bank"), os.path.join("ratio_analysis_plots", "f1", "bank"), "bank f1")