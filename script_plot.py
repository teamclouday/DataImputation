# script for generating plots from local data

# this script is currently for computing random ratios on compas analysis

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from script_parallel import iter_per_ratio, random_ratios

PLOT_CREATE_MEAN        = False
PLOT_CREATE_SIMILAR_V1  = False
PLOT_CREATE_SIMILAR_V2  = False
PLOT_CREATE_MULTI       = False

PLOT_PARETO_FRONTIER    = True

def plot_func(data, method_name, file_name=None, bias1_ylim=None, bias2_ylim=None, plot_error=True):
    assert len(data) == (iter_per_ratio * len(random_ratios))
    classifiers = ["KNN", "LinearSVC", "SVC", "Forest", "LogReg", "Tree", "MLP"]
    raise Exception("Update classifiers before run!")
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
                data_processed[0].append(np.mean(tmp_data_processed[0]))
                data_processed[1].append(np.mean(tmp_data_processed[1]))
                data_processed[2].append(np.mean(tmp_data_processed[2]))
            plot_acc[clf][0].append(np.mean(data_processed[0]))
            plot_acc[clf][1].append(np.std(data_processed[0]))
            plot_bias1[clf][0].append(np.mean(data_processed[1]))
            plot_bias1[clf][1].append(np.std(data_processed[1]))
            plot_bias2[clf][0].append(np.mean(data_processed[2]))
            plot_bias2[clf][1].append(np.std(data_processed[2]))
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

def plot_func_pareto_front(data, file_name=None, y_scale=None, switch=False):
    # extract data
    data_mean       = data["mean"]
    data_similar_v1 = data["similar_v1"]
    data_similar_v2 = data["similar_v2"]
    data_multi      = data["multi"]
    classifiers = ["KNN", "LinearSVC", "SVC", "Forest", "LogReg", "Tree", "MLP"]
    raise Exception("Update classifiers before run!")
    plot_colors = ["red", "green", "blue", "gold", "darkorange", "grey", "purple"]
    plot_markers = ["o", "s", "*", "^", "P", "v", "X"]
    ratio_dot_size      = [(x+1)*3 for x in range(len(random_ratios))]
    assert len(data_mean) == len(data_similar_v1) == len(data_similar_v2) == len(data_multi) == (iter_per_ratio * len(random_ratios))
    # prepare data for plotting
    plot_data_mean          = {}
    plot_data_similar_v1    = {}
    plot_data_similar_v2    = {}
    plot_data_multi         = {}
    for clf in classifiers:
        plot_data_mean[clf]         = ([], [], []) # [[acc_X], [bias1_Y], [bias2_Y]]
        plot_data_similar_v1[clf]   = ([], [], [])
        plot_data_similar_v2[clf]   = ([], [], [])
        plot_data_multi[clf]        = ([], [], [])
    for i in range(0, len(data_mean), iter_per_ratio):
        d_data_mean         = data_mean[i:(i+iter_per_ratio)]
        d_data_similar_v1   = data_similar_v1[i:(i+iter_per_ratio)]
        d_data_similar_v2   = data_similar_v2[i:(i+iter_per_ratio)]
        d_data_multi        = data_multi[i:(i+iter_per_ratio)]
        for clf in classifiers:
            clf_data_mean       = [(x[0][clf], x[1][clf], x[2][clf]) for x in d_data_mean]
            clf_data_similar_v1 = [(x[0][clf], x[1][clf], x[2][clf]) for x in d_data_similar_v1]
            clf_data_similar_v2 = [(x[0][clf], x[1][clf], x[2][clf]) for x in d_data_similar_v2]
            clf_data_multi      = [(x[0][clf], x[1][clf], x[2][clf]) for x in d_data_multi]
            # process mean method
            data_processed = [[], [], []] # [[acc], [bias1], [bias2]]
            for xx, yy, zz in clf_data_mean:
                tmp_processed = [[], [], []] # [[acc], [bias1], [bias2]]
                for x, y, z in zip(xx, yy, zz):
                    if (y > 0) and (z > 0):
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
            plot_data_mean[clf][0].append(np.mean(data_processed[0]))
            plot_data_mean[clf][1].append(np.mean(data_processed[1]))
            plot_data_mean[clf][2].append(np.mean(data_processed[2]))
            # process similar_v1 method
            data_processed = [[], [], []] # [[acc], [bias1], [bias2]]
            for xx, yy, zz in clf_data_similar_v1:
                tmp_processed = [[], [], []] # [[acc], [bias1], [bias2]]
                for x, y, z in zip(xx, yy, zz):
                    if (y > 0) and (z > 0):
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
            plot_data_similar_v1[clf][0].append(np.mean(data_processed[0]))
            plot_data_similar_v1[clf][1].append(np.mean(data_processed[1]))
            plot_data_similar_v1[clf][2].append(np.mean(data_processed[2]))
            # process similar_v2 method
            data_processed = [[], [], []] # [[acc], [bias1], [bias2]]
            for xx, yy, zz in clf_data_similar_v2:
                tmp_processed = [[], [], []] # [[acc], [bias1], [bias2]]
                for x, y, z in zip(xx, yy, zz):
                    if (y > 0) and (z > 0):
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
            plot_data_similar_v2[clf][0].append(np.mean(data_processed[0]))
            plot_data_similar_v2[clf][1].append(np.mean(data_processed[1]))
            plot_data_similar_v2[clf][2].append(np.mean(data_processed[2]))
            # process multi method
            data_processed = [[], [], []] # [[acc], [bias1], [bias2]]
            for xx, yy, zz in clf_data_multi:
                tmp_processed = [[], [], []] # [[acc], [bias1], [bias2]]
                for x, y, z in zip(xx, yy, zz):
                    if (y > 0) and (z > 0):
                        tmp_processed[0].append(x)
                        tmp_processed[1].append(y)
                        tmp_processed[2].append(z)
                data_processed[0].append(np.mean(tmp_processed[0]))
                data_processed[1].append(np.mean(tmp_processed[1]))
                data_processed[2].append(np.mean(tmp_processed[2]))
            plot_data_multi[clf][0].append(np.mean(data_processed[0]))
            plot_data_multi[clf][1].append(np.mean(data_processed[1]))
            plot_data_multi[clf][2].append(np.mean(data_processed[2]))
    fig, axes = plt.subplots(2, figsize=(8, 12)) # axes[0] for bias1, axes[1] for bias2
    axes[0].set_xlim([0.38, 0.7])
    axes[1].set_xlim([0.38, 0.7])
    axes[0].set_xlabel("Accuracy")
    axes[1].set_xlabel("Accuracy")
    axes[0].set_ylabel("Bias1 Values")
    axes[1].set_ylabel("Bias2 Values")
    # each classifier has different color
    for clf, clf_c, clf_m in zip(classifiers, plot_colors, plot_markers):
        if switch:
            # plot for mean method
            axes[0].scatter(plot_data_mean[clf][0], plot_data_mean[clf][1], c=plot_colors[0], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_mean[clf][0], plot_data_mean[clf][2], c=plot_colors[0], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for similar_v1 method
            axes[0].scatter(plot_data_similar_v1[clf][0], plot_data_similar_v1[clf][1], c=plot_colors[1], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_similar_v1[clf][0], plot_data_similar_v1[clf][2], c=plot_colors[1], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for similar_v2 method
            axes[0].scatter(plot_data_similar_v2[clf][0], plot_data_similar_v2[clf][1], c=plot_colors[2], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_similar_v2[clf][0], plot_data_similar_v2[clf][2], c=plot_colors[2], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            # plot for multi method
            axes[0].scatter(plot_data_multi[clf][0], plot_data_multi[clf][1], c=plot_colors[3], s=ratio_dot_size, marker=clf_m, alpha=0.8)
            axes[1].scatter(plot_data_multi[clf][0], plot_data_multi[clf][2], c=plot_colors[3], s=ratio_dot_size, marker=clf_m, alpha=0.8)
        else:
            # plot for mean method
            axes[0].scatter(plot_data_mean[clf][0], plot_data_mean[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[0], alpha=0.8)
            axes[1].scatter(plot_data_mean[clf][0], plot_data_mean[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[0], alpha=0.8)
            # plot for similar_v1 method
            axes[0].scatter(plot_data_similar_v1[clf][0], plot_data_similar_v1[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[1], alpha=0.8)
            axes[1].scatter(plot_data_similar_v1[clf][0], plot_data_similar_v1[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[1], alpha=0.8)
            # plot for similar_v2 method
            axes[0].scatter(plot_data_similar_v2[clf][0], plot_data_similar_v2[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[2], alpha=0.8)
            axes[1].scatter(plot_data_similar_v2[clf][0], plot_data_similar_v2[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[2], alpha=0.8)
            # plot for multi method
            axes[0].scatter(plot_data_multi[clf][0], plot_data_multi[clf][1], c=clf_c, s=ratio_dot_size, marker=plot_markers[3], alpha=0.8)
            axes[1].scatter(plot_data_multi[clf][0], plot_data_multi[clf][2], c=clf_c, s=ratio_dot_size, marker=plot_markers[3], alpha=0.8)
    if y_scale:
        axes[0].set_yscale(y_scale)
        axes[1].set_yscale(y_scale)
    if switch:
        custom_legend = [Line2D([0], [0], color='w', markerfacecolor="black", marker=x, label=y, markersize=10) for x,y in zip(plot_markers, classifiers)]
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[0], marker="o", label="mean"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[1], marker="o", label="similar_v1"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[2], marker="o", label="similar_v2"))
        custom_legend.append(Line2D([0], [0], markersize=10, markerfacecolor=plot_colors[3], marker="o", label="multi"))
    else:
        custom_legend = [Line2D([0], [0], markerfacecolor=x, marker="o", label=y, markersize=10) for x,y in zip(plot_colors, classifiers)]
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[0], label="mean"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[1], label="similar_v1"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[2], label="similar_v2"))
        custom_legend.append(Line2D([0], [0], color='w', markersize=10, markerfacecolor="black", marker=plot_markers[3], label="multi"))
    plt.legend(handles=custom_legend, bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.suptitle("Pareto Front Plots")
    plt.subplots_adjust(top=0.94)
    if file_name:
        plt.savefig(file_name, transparent=False, bbox_inches='tight', pad_inches=0.1)
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
    # generate pareto front plots
    if os.path.exists("mean.pkl") and os.path.exists("similar_v1.pkl") and os.path.exists("similar_v2.pkl") and os.path.exists("multi.pkl") and PLOT_PARETO_FRONTIER:
        data = {}
        with open("mean.pkl", "rb") as inFile:
            data["mean"] = pickle.load(inFile)
        with open("similar_v1.pkl", "rb") as inFile:
            data["similar_v1"] = pickle.load(inFile)
        with open("similar_v2.pkl", "rb") as inFile:
            data["similar_v2"] = pickle.load(inFile)
        with open("multi.pkl", "rb") as inFile:
            data["multi"] = pickle.load(inFile)
        plot_func_pareto_front(data, "ratio_analysis_plots/pareto_front.png")
        plot_func_pareto_front(data, "ratio_analysis_plots/pareto_front_scaled.png", y_scale="log")
        plot_func_pareto_front(data, "ratio_analysis_plots/pareto_front_v2.png", switch=True)
        plot_func_pareto_front(data, "ratio_analysis_plots/pareto_front_scaled_v2.png", y_scale="log", switch=True)