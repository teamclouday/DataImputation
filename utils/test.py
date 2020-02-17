# codes for the test platform

from utils.data import *
from utils.generator import *
from utils.completer import *
from utils.predictor import *
from utils.model_header import *
import matplotlib.pyplot as plt
import seaborn as sns

class TestMachine:
    def __init__(self, data_func, random_func,
                 predictor_cv=5, random_gen=1,
                 complete_func=[], model_func=[],
                 record_time=False):
        self.data_func_name = data_func.__name__
        self.random_func_name = random_func.__name__
        self.record_time = record_time
        self.original_data = data_func(print_time=self.record_time)
        assert random_gen >= 1 # make sure at least one
        if random_gen > 1:
            print("You're using random generation > 1: {}".format(random_gen))
            self.missing_data = [random_func(self.original_data, print_time=self.record_time) for _ in range(random_gen)]
        else:
            self.missing_data = random_func(self.original_data, print_time=self.record_time)
        self.completers = [
                           complete_by_value,
                           complete_by_mean_col,
                           complete_by_nearby_row,
                           complete_by_similar_row,
                           complete_by_most_freq
                           ] if complete_func == [] else complete_func
        self.models = [
                        KNN,
                        SGD,
                        DecisionTree,
                        SVM,
                        Forest
                        ] if model_func == [] else model_func
        self.predictor_cv = predictor_cv
    
    def run(self):
        scores = []
        for model in self.models:
            scores.append(["original", model.__name__, model(self.original_data, self.predictor_cv, self.record_time)])
        for completer in self.completers:
            if type(self.missing_data) is list:
                completed_data = [completer(x, print_time=self.record_time) for x in self.missing_data]
                for model in self.models:
                    model_score = [model(cc, self.predictor_cv, self.record_time) for cc in completed_data]
                    scores.append([completer.__name__, model.__name__, sum(model_score)/len(model_score)])
            else:
                completed_data = completer(self.missing_data, print_time=self.record_time)
                for model in self.models:
                    scores.append([completer.__name__, model.__name__, model(completed_data, self.predictor_cv, self.record_time)])
        print("All tests complete")
        self.scores = pd.DataFrame(data=scores, columns=["Completer Functions", "Models", "Scores"], index=None)
        print(self.scores.to_string())

    def plot_compare_models(self, size=(12, 6), save_file_name=None):
        plt.figure(figsize=size)
        plt.title("Model comparison on [{0}] with [{1}]".format(self.data_func_name, self.random_func_name))
        ax = sns.barplot(x=self.scores["Completer Functions"], y=self.scores["Scores"], hue="Models", data=self.scores, palette="magma")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="center", xytext=(0, 10), textcoords="offset points")
        plt.ylim([0.0, 1.5])
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(save_file_name)
        plt.show()
    
    def plot_compare_completers(self, size=(12, 6), save_file_name=None):
        plt.figure(figsize=size)
        plt.title("Completer comparison on [{0}] with [{1}]".format(self.data_func_name, self.random_func_name))
        ax = sns.barplot(x=self.scores["Models"], y=self.scores["Scores"], hue="Completer Functions", data=self.scores, palette="magma")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="center", xytext=(0, 10), textcoords="offset points")
        plt.ylim([0.0, 1.5])
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(save_file_name)
        plt.show()