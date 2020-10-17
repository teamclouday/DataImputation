# Extended version of script plot
# For experiments on
# Missing At Random
# Missing Not At Random

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from script_single_task import random_ratios
from script_single_task import acc, bias1, bias2, newBias
from script_single_task_ext import RUN_MEAN_V1, RUN_MEAN_V2, RUN_MULTI_V1, RUN_MULTI_V2, RUN_SIMILAR_V1, RUN_SIMILAR_V2

iter_per_ratio = 200

NAME_DATASETS = ["adult", "compas", "titanic", "german", "communities", "bank"]
NAME_TARGETS = ["MAR", "MNAR"]

PLOT_MAR  = True
PLOT_MNAR = True

PLOT_ADULT          = False
PLOT_COMPAS         = False
PLOT_TITANIC        = False
PLOT_GERMAN         = False
PLOT_COMMUNITIES    = False
PLOT_BANK           = False

TRANSFORM_OUTPUTS = True

PLOT_CREATE_MEAN_V1     = True
PLOT_CREATE_MEAN_V2     = True
PLOT_CREATE_SIMILAR_V1  = True
PLOT_CREATE_SIMILAR_V2  = True
PLOT_CREATE_MULTI_V1    = True
PLOT_CREATE_MULTI_V2    = True

PLOT_PARETO_FRONTIER_ACC     = False
PLOT_PARETO_FRONTIER_REALACC = False

def plot_helper(clf_data):
    tmp_data_processed = [[], [], [], []] #  [[acc each fold], [bias1], [bias2], [new bias]], remove -1, [None] cases
    for mm in clf_data:
        if len(mm) < 1:
            continue
        cf_m = mm[0]
        try:
            x = acc(cf_m)
            y = bias1(cf_m)
            z = bias2(cf_m)
            k = newBias(cf_m)
        except Exception as e:
            print("Error: {}".format(e))
            continue
        if (y > 0) and (z > 0):
            tmp_data_processed[0].append(x)
            tmp_data_processed[1].append(y)
            tmp_data_processed[2].append(z)
            tmp_data_processed[3].append(k)
    return tmp_data_processed

def plot_func(data, method_name, file_name=None):
    assert len(data[0]) == len(data[1]) == len(data[2])
    classifiers = ["KNN", "LinearSVC", "Forest", "LogReg", "Tree", "MLP"]
    plot_newbias = {}
    plot_acc = {}
    plot_bias1 = {}
    plot_bias2 = {}
    plot_distribution = []
    for clf in classifiers:
        plot_bias1[clf]   = [[], [], [], []] # [[complete data], [missing data], [complete data error], [missing data error]]
        plot_bias2[clf]   = [[], [], [], []]
        plot_acc[clf]     = [[], [], [], []]
        plot_newbias[clf] = [[], [], [], []]
    for i in range(0, len(data[0])):
        i_data_complete = data[0][i]
        i_data_missing = data[1][i]
        i_data_ratio = data[2][i]
        plot_distribution.append(i_data_ratio)
        for clf in classifiers:
            data_processed_complete = plot_helper(i_data_complete[clf])
            data_processed_missing = plot_helper(i_data_missing[clf])

            if len(data_processed_complete[0]) > 0:
                plot_acc[clf][0].append(np.mean(data_processed_complete[0]))
                plot_acc[clf][2].append(np.std(data_processed_complete[0]))
            if len(data_processed_missing[0]) > 0:
                plot_acc[clf][1].append(np.mean(data_processed_missing[0]))
                plot_acc[clf][3].append(np.std(data_processed_missing[0]))

            if len(data_processed_complete[1]) > 0:
                plot_bias1[clf][0].append(np.mean(data_processed_complete[1]))
                plot_bias1[clf][2].append(np.std(data_processed_complete[1]))
            if len(data_processed_missing[1]) > 0:
                plot_bias1[clf][1].append(np.mean(data_processed_missing[1]))
                plot_bias1[clf][3].append(np.std(data_processed_missing[1]))

            if len(data_processed_complete[2]) > 0:
                plot_bias2[clf][0].append(np.mean(data_processed_complete[2]))
                plot_bias2[clf][2].append(np.std(data_processed_complete[2]))
            if len(data_processed_missing[2]) > 0:
                plot_bias2[clf][1].append(np.mean(data_processed_missing[2]))
                plot_bias2[clf][3].append(np.std(data_processed_missing[2]))

            if len(data_processed_complete[3]) > 0:
                plot_newbias[clf][0].append(np.mean(data_processed_complete[3]))
                plot_newbias[clf][2].append(np.std(data_processed_complete[3]))
            if len(data_processed_missing[3]) > 0:
                plot_newbias[clf][1].append(np.mean(data_processed_missing[3]))
                plot_newbias[clf][3].append(np.std(data_processed_missing[3]))

    plot_distribution = [round(m, 2) for m in plot_distribution]
    plot_distribution_processed = np.unique(plot_distribution, return_counts=True)

    bar_width = 0.5
    bar_dist = 0.2
    bar_pos = [np.array([0.5, 1.0]) + i*(2*bar_width+bar_dist) for i in range(len(classifiers))]
    bar_pos = np.array(bar_pos).ravel()
    bar_names = ["0", "{:.2f}".format(np.mean(plot_distribution))] * len(classifiers)

    fig, axes = plt.subplots(5, figsize=(10, 45))
    # axes[0] shows distribution
    axes[0].set_title("Missing Percentage Distribution")
    axes[0].bar(plot_distribution_processed[0]-0.005, plot_distribution_processed[1], 0.01)
    # axes[1] shows new bias
    axes[1].set_title("New Bias")
    for i in range(len(classifiers)):
        clf = classifiers[i]
        axes[1].bar(bar_pos[i*2], np.mean(plot_newbias[clf][0]), width=bar_width, yerr=np.mean(plot_newbias[clf][2]), color="grey", label="original" if i == 0 else "")
        axes[1].bar(bar_pos[i*2+1], np.mean(plot_newbias[clf][1]), width=bar_width, yerr=np.mean(plot_newbias[clf][3]), label=clf)
    axes[1].set_xticks(bar_pos)
    axes[1].set_xticklabels(bar_names)
    axes[1].legend(loc="best")
    # axes[2] shows accuracy
    axes[2].set_title("accuracy")
    for i in range(len(classifiers)):
        clf = classifiers[i]
        axes[2].bar(bar_pos[i*2], np.mean(plot_acc[clf][0]), width=bar_width, yerr=np.mean(plot_acc[clf][2]), color="grey", label="original" if i == 0 else "")
        axes[2].bar(bar_pos[i*2+1], np.mean(plot_acc[clf][1]), width=bar_width, yerr=np.mean(plot_acc[clf][3]), label=clf)
    axes[2].set_xticks(bar_pos)
    axes[2].set_xticklabels(bar_names)
    axes[2].legend(loc="best")
    # axes[3] shows bias 1
    axes[3].set_title("Bias 1")
    for i in range(len(classifiers)):
        clf = classifiers[i]
        axes[3].bar(bar_pos[i*2], np.mean(plot_bias1[clf][0]), width=bar_width, yerr=np.mean(plot_bias1[clf][2]), color="grey", label="original" if i == 0 else "")
        axes[3].bar(bar_pos[i*2+1], np.mean(plot_bias1[clf][1]), width=bar_width, yerr=np.mean(plot_bias1[clf][3]), label=clf)
    axes[3].set_xticks(bar_pos)
    axes[3].set_xticklabels(bar_names)
    axes[3].legend(loc="best")
    # axes[4] shows bias 2
    axes[4].set_title("Bias 2")
    for i in range(len(classifiers)):
        clf = classifiers[i]
        axes[4].bar(bar_pos[i*2], np.mean(plot_bias2[clf][0]), width=bar_width, yerr=np.mean(plot_bias2[clf][2]), color="grey", label="original" if i == 0 else "")
        axes[4].bar(bar_pos[i*2+1], np.mean(plot_bias2[clf][1]), width=bar_width, yerr=np.mean(plot_bias2[clf][3]), label=clf)
    axes[4].set_xticks(bar_pos)
    axes[4].set_xticklabels(bar_names)
    axes[4].legend(loc="best")
    fig.tight_layout()
    fig.suptitle("Imputation Method: {}".format(method_name))
    plt.subplots_adjust(top=0.96)
    if file_name:
        fig.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def plot_func_pareto_front(data, title, file_name=None, y_scale=False, x_axis="acc"):
    pass

def compress_outputs(target="MAR"):
    for dd in NAME_DATASETS:
        if not os.path.exists(os.path.join("condor_outputs", target, dd)):
            print("Folder not found: {}".format(os.path.join("condor_outputs", target, dd)))
            continue
        final_results = {}
        if RUN_MEAN_V1: final_results["mean_v1"] = [[], [], []] # [[complete data results], [missing data results], [missing ratios]]
        if RUN_MEAN_V2: final_results["mean_v2"] = [[], [], []]
        if RUN_SIMILAR_V1: final_results["similar_v1"] = [[], [], []]
        if RUN_SIMILAR_V2: final_results["similar_v2"] = [[], [], []]
        if RUN_MULTI_V1: final_results["multi_v1"] = [[], [], []]
        if RUN_MULTI_V2: final_results["multi_v2"] = [[], [], []]
        need_reload = False
        for key in final_results.keys():
            if not os.path.exists(os.path.join("condor_outputs", target, dd, "{}.pkl".format(key))):
                need_reload = True
        if not need_reload:
            continue
        # load data
        for i in range(iter_per_ratio):
            if not os.path.exists(os.path.join("condor_outputs", target, dd, "output_{:0>4}.pkl".format(i))):
                continue
            with open(os.path.join("condor_outputs", target, dd, "output_{:0>4}.pkl".format(i)), "rb") as inFile:
                output_data = pickle.load(inFile)
            for key, value in output_data.items():
                assert key in final_results.keys()
                assert len(value) == 3
                final_results[key][0].append(value[0])
                final_results[key][1].append(value[1])
                final_results[key][2].append(value[2])
        # dump data
        for key in final_results.keys():
            if len(final_results[key][0]) < 1:
                continue
            dump_data = final_results[key]
            with open(os.path.join("condor_outputs", target, dd, "{}.pkl".format(key)), "wb") as outFile:
                pickle.dump(dump_data, outFile)

def plot_all(data_folder, plot_folder, name):
    # generate plot for mean_v1.pkl
    if os.path.exists(os.path.join(data_folder, "mean_v1.pkl")) and PLOT_CREATE_MEAN_V1:
        print("Generating plot for mean_v1.pkl ({})".format(name))
        with open(os.path.join(data_folder, "mean_v1.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Mean V1 ({})".format(name), os.path.join(plot_folder, "mean_v1.png"))
    # generate plot for mean_v2.pkl
    if os.path.exists(os.path.join(data_folder, "mean_v2.pkl")) and PLOT_CREATE_MEAN_V1:
        print("Generating plot for mean_v2.pkl ({})".format(name))
        with open(os.path.join(data_folder, "mean_v2.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Mean V2 ({})".format(name), os.path.join(plot_folder, "mean_v2.png"))
    # generate plot for similar_v1.pkl
    if os.path.exists(os.path.join(data_folder, "similar_v1.pkl")) and PLOT_CREATE_SIMILAR_V1:
        print("Generating plot for similar_v1.pkl ({})".format(name))
        with open(os.path.join(data_folder, "similar_v1.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Similar V1 ({})".format(name), os.path.join(plot_folder, "similar_v1.png"))
    # generate plot for similar_v2.pkl
    if os.path.exists(os.path.join(data_folder, "similar_v2.pkl")) and PLOT_CREATE_SIMILAR_V2:
        print("Generating plot for similar_v2.pkl ({})".format(name))
        with open(os.path.join(data_folder, "similar_v2.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Similar V2 ({})".format(name), os.path.join(plot_folder, "similar_v2.png"))
    # generate plot for multi_v1.pkl
    if os.path.exists(os.path.join(data_folder, "multi_v1.pkl")) and PLOT_CREATE_MULTI_V1:
        print("Generating plot for multi_v1.pkl ({})".format(name))
        with open(os.path.join(data_folder, "multi_v1.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Multiple Imputation V1 ({})".format(name), os.path.join(plot_folder, "multi_v1.png"))
    # generate plot for multi_v2.pkl
    if os.path.exists(os.path.join(data_folder, "multi_v2.pkl")) and PLOT_CREATE_MULTI_V1:
        print("Generating plot for multi_v2.pkl ({})".format(name))
        with open(os.path.join(data_folder, "multi_v2.pkl"), "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Multiple Imputation V2 ({})".format(name), os.path.join(plot_folder, "multi_v2.png"))
    # generate pareto front plots
    # if  os.path.exists(os.path.join(data_folder, "mean_v1.pkl")) and \
    #     os.path.exists(os.path.join(data_folder, "mean_v2.pkl")) and \
    #     os.path.exists(os.path.join(data_folder, "similar_v1.pkl")) and \
    #     os.path.exists(os.path.join(data_folder, "similar_v2.pkl")) and \
    #     os.path.exists(os.path.join(data_folder, "multi_v1.pkl")) and \
    #     os.path.exists(os.path.join(data_folder, "multi_v2.pkl")) and \
    #     (PLOT_PARETO_FRONTIER_ACC or PLOT_PARETO_FRONTIER_REALACC):
    #     data = {}
    #     with open(os.path.join(data_folder, "mean_v1.pkl"), "rb") as inFile:
    #         data["mean_v1"] = pickle.load(inFile)
    #     with open(os.path.join(data_folder, "mean_v2.pkl"), "rb") as inFile:
    #         data["mean_v2"] = pickle.load(inFile)
    #     with open(os.path.join(data_folder, "similar_v1.pkl"), "rb") as inFile:
    #         data["similar_v1"] = pickle.load(inFile)
    #     with open(os.path.join(data_folder, "similar_v2.pkl"), "rb") as inFile:
    #         data["similar_v2"] = pickle.load(inFile)
    #     with open(os.path.join(data_folder, "multi_v1.pkl"), "rb") as inFile:
    #         data["multi_v1"] = pickle.load(inFile)
    #     with open(os.path.join(data_folder, "multi_v2.pkl"), "rb") as inFile:
    #         data["multi_v2"] = pickle.load(inFile)
    #     if PLOT_PARETO_FRONTIER_ACC:
    #         print("Generate plots for pareto front acc ({})".format(name))
    #         plot_func_pareto_front(data, "Pareto Front (Confusion Matrix Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_acc.png"), x_axis="acc")
    #         plot_func_pareto_front(data, "Pareto Front (Confusion Matrix Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_acc_scaled.png"), y_scale=True, x_axis="acc")
    #     if PLOT_PARETO_FRONTIER_REALACC:
    #         print("Generate plots for pareto front real acc ({})".format(name))
    #         plot_func_pareto_front(data, "Pareto Front (Real Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_realacc.png"), x_axis="realacc")
    #         plot_func_pareto_front(data, "Pareto Front (Real Accuracy) ({})".format(name), os.path.join(plot_folder, "pareto_front_realacc_scaled.png"), y_scale=True, x_axis="realacc")

if __name__=="__main__":
    if not os.path.exists("other_analysis_plots"):
        os.makedirs("other_analysis_plots")
    for tt in NAME_TARGETS:
        if not os.path.exists(os.path.join("other_analysis_plots", tt)):
            os.makedirs(os.path.join("other_analysis_plots", tt))
        for dd in NAME_DATASETS:
            if not os.path.exists(os.path.join("other_analysis_plots", tt, dd)):
                os.makedirs(os.path.join("other_analysis_plots", tt, dd))

    if TRANSFORM_OUTPUTS:
        if PLOT_MAR:
            compress_outputs(target="MAR")
        if PLOT_MNAR:
            compress_outputs(target="MNAR")
    
    if PLOT_MAR:
        if PLOT_ADULT:
            plot_all(os.path.join("condor_outputs", "MAR", "adult"), os.path.join("other_analysis_plots", "MAR", "adult"), "adult MAR")
        if PLOT_COMPAS:
            plot_all(os.path.join("condor_outputs", "MAR", "compas"), os.path.join("other_analysis_plots", "MAR", "compas"), "compas MAR")
        if PLOT_TITANIC:
            plot_all(os.path.join("condor_outputs", "MAR", "titanic"), os.path.join("other_analysis_plots", "MAR", "titanic"), "titanic MAR")
        if PLOT_GERMAN:
            plot_all(os.path.join("condor_outputs", "MAR", "german"), os.path.join("other_analysis_plots", "MAR", "german"), "german MAR")
        if PLOT_COMMUNITIES:
            plot_all(os.path.join("condor_outputs", "MAR", "communities"), os.path.join("other_analysis_plots", "MAR", "communities"), "communities MAR")
        if PLOT_BANK:
            plot_all(os.path.join("condor_outputs", "MAR", "bank"), os.path.join("other_analysis_plots", "MAR", "bank"), "bank MAR")

    if PLOT_MNAR:
        if PLOT_ADULT:
            plot_all(os.path.join("condor_outputs", "MNAR", "adult"), os.path.join("other_analysis_plots", "MNAR", "adult"), "adult MNAR")
        if PLOT_COMPAS:
            plot_all(os.path.join("condor_outputs", "MNAR", "compas"), os.path.join("other_analysis_plots", "MNAR", "compas"), "compas MNAR")
        if PLOT_TITANIC:
            plot_all(os.path.join("condor_outputs", "MNAR", "titanic"), os.path.join("other_analysis_plots", "MNAR", "titanic"), "titanic MNAR")
        if PLOT_GERMAN:
            plot_all(os.path.join("condor_outputs", "MNAR", "german"), os.path.join("other_analysis_plots", "MNAR", "german"), "german MNAR")
        if PLOT_COMMUNITIES:
            plot_all(os.path.join("condor_outputs", "MNAR", "communities"), os.path.join("other_analysis_plots", "MNAR", "communities"), "communities MNAR")
        if PLOT_BANK:
            plot_all(os.path.join("condor_outputs", "MNAR", "bank"), os.path.join("other_analysis_plots", "MNAR", "bank"), "bank MNAR")