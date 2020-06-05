# script for generating plots from local data

# this script is currently for computing random ratios on compas analysis

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from script_parallel import iter_per_ratio, random_ratios

PLOT_CREATE_MEAN        = False
PLOT_CREATE_SIMILAR_V1  = False
PLOT_CREATE_SIMILAR_V2  = True
PLOT_CREATE_MULTI       = False

def plot_func(data, method_name, file_name=None, bias1_ylim=None, bias2_ylim=None, plot_error=True):
    assert len(data) == (iter_per_ratio * len(random_ratios))
    classifiers = ["KNN", "LinearSVC", "SVC", "Forest", "LogReg", "Tree", "MLP"]
    d_acc   = [x[0] for x in data]
    d_bias1 = [x[1] for x in data]
    d_bias2 = [x[2] for x in data]
    assert len(d_acc) == len(d_bias1) == len(d_bias2)
    plot_bias1 = {}
    plot_bias2 = {}
    plot_acc = {}
    for clf in classifiers:
        plot_bias1[clf] = [[], []] # [[actual averaged data], [data for error bar]]
        plot_bias2[clf] = [[], []]
        plot_acc[clf]   = [[], []]
    for i in range(0, len(d_acc), iter_per_ratio):
        i_acc = d_acc[i:(i+iter_per_ratio)]
        i_bias1 = d_bias1[i:(i+iter_per_ratio)]
        i_bias2 = d_bias2[i:(i+iter_per_ratio)]
        for clf in classifiers:
            clf_acc = [x[clf] for x in i_acc]
            clf_bias1 = [x[clf] for x in i_bias1]
            clf_bias2 = [x[clf] for x in i_bias2]
            data_processed = [[], [], []] # [[acc avg], [bias1], [bias2]], remove -1 cases
            for xx,yy,zz in zip(clf_acc, clf_bias1, clf_bias2):
                tmp_data_processed = [[], [], []] #  [[acc each fold], [bias1], [bias2]], remove -1 cases
                for x,y,z in zip(xx, yy, zz):
                    if (y > 0) and (z > 0):
                        tmp_data_processed[0].append(x)
                        tmp_data_processed[1].append(y)
                        tmp_data_processed[2].append(z)
                data_processed[0].append(sum(tmp_data_processed[0]) / len(tmp_data_processed[0]))
                data_processed[1].append(sum(tmp_data_processed[1]) / len(tmp_data_processed[1]))
                data_processed[2].append(sum(tmp_data_processed[2]) / len(tmp_data_processed[2]))
            plot_acc[clf][0].append(sum(data_processed[0]) / len(data_processed[0]))
            plot_acc[clf][1].append((max(data_processed[0]) - min(data_processed[0])) / 2)
            plot_bias1[clf][0].append(sum(data_processed[1]) / len(data_processed[1]))
            plot_bias1[clf][1].append((max(data_processed[1]) - min(data_processed[1])) / 2)
            plot_bias2[clf][0].append(sum(data_processed[2]) / len(data_processed[2]))
            plot_bias2[clf][1].append((max(data_processed[2]) - min(data_processed[2])) / 2)
    plot_gap = 0.002
    fig, axes = plt.subplots(3, figsize=(10, 15))
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
    if bias1_ylim:
        axes[0].set_ylim(bias1_ylim)
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
    if bias2_ylim:
        axes[1].set_ylim(bias2_ylim)
    # axes[2] shows accuracy
    axes[2].set_title("Accuracy")
    for i, clf in enumerate(classifiers):
        if plot_error:
            axes[2].errorbar(random_ratios+(i-3)*plot_gap, plot_acc[clf][0], yerr=plot_acc[clf][1], label=clf)
        else:
            axes[2].plot(random_ratios+(i-3)*plot_gap, plot_acc[clf][0], label=clf)
        axes[2].scatter(random_ratios+(i-3)*plot_gap, plot_acc[clf][0], s=2)
    axes[2].legend(loc="best")
    axes[2].set_xticks(np.arange(0.0, 1.0, 0.05))
    fig.tight_layout()
    fig.suptitle("Imputation Method: {}".format(method_name))
    plt.subplots_adjust(top=0.94)
    if file_name:
        fig.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__=="__main__":
    if not os.path.exists("ratio_analysis_plots"):
        os.makedirs("ratio_analysis_plots")
    # generate plot for mean.pkl
    if os.path.exists("mean.pkl") and PLOT_CREATE_MEAN:
        print("Generating plot for mean.pkl")
        with open("mean.pkl", "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Mean", "ratio_analysis_plots/ratio_mean.png")
        plot_func(data, "Complete by Mean", "ratio_analysis_plots/ratio_mean_scaled.png", bias1_ylim=[-0.2, 1.8])
        plot_func(data, "Complete by Mean", "ratio_analysis_plots/ratio_mean_scaled_clean.png", bias1_ylim=[-0.2, 1.8], plot_error=False)
    # generate plot for similar_v1.pkl
    if os.path.exists("similar_v1.pkl") and PLOT_CREATE_SIMILAR_V1:
        print("Generating plot for similar_v1.pkl")
        with open("similar_v1.pkl", "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Similar V1", "ratio_analysis_plots/ratio_similar_v1.png")
        plot_func(data, "Complete by Similar V1", "ratio_analysis_plots/ratio_similar_v1_scaled.png", bias1_ylim=[-0.2, 1.5])
        plot_func(data, "Complete by Similar V1", "ratio_analysis_plots/ratio_similar_v1_scaled_clean.png", bias1_ylim=[-0.2, 1.5], plot_error=False)
    # generate plot for similar_v2.pkl
    if os.path.exists("similar_v2.pkl") and PLOT_CREATE_SIMILAR_V2:
        print("Generating plot for similar_v2.pkl")
        with open("similar_v2.pkl", "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Similar V2", "ratio_analysis_plots/ratio_similar_v2.png")
        plot_func(data, "Complete by Similar V2", "ratio_analysis_plots/ratio_similar_v2_scaled.png", bias1_ylim=[0.0, 1.6], bias2_ylim=[0.0, 1.5])
        plot_func(data, "Complete by Similar V2", "ratio_analysis_plots/ratio_similar_v2_scaled_clean.png", bias1_ylim=[0.0, 1.6], bias2_ylim=[0.0, 1.5], plot_error=False)
    # generate plot for multi.pkl
    if os.path.exists("multi.pkl") and PLOT_CREATE_MULTI:
        print("Generating plot for multi.pkl")
        with open("multi.pkl", "rb") as inFile:
            data = pickle.load(inFile)
        plot_func(data, "Complete by Multiple Imputation", "ratio_analysis_plots/ratio_multi.png")
        plot_func(data, "Complete by Multiple Imputation", "ratio_analysis_plots/ratio_multi_scaled.png", bias1_ylim=[-0.1, 1.4])
        plot_func(data, "Complete by Multiple Imputation", "ratio_analysis_plots/ratio_multi_scaled_clean.png", bias1_ylim=[-0.1, 1.4], plot_error=False)