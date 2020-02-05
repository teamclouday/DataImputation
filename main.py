# this file contains codes that test various research targets

import shutil
from utils.data import *
from utils.generator import *
from utils.completer import *
from utils.predictor import *
from utils.model_header import *
import matplotlib.pyplot as plt
import seaborn as sns

class TestMachine:
    def __init__(self, data_func, random_func, complete_func=[], model_func=[]):
        self.data_func_name = data_func.__name__
        self.random_func_name = random_func.__name__
        self.original_data = data_func()
        self.missing_data = random_func(self.original_data)
        self.completers = [complete_by_value,
                           complete_by_mean_col,
                           complete_by_nearby_row] if complete_func == [] else complete_func
        self.models = [KNN, SGD, DecisionTree] if model_func == [] else model_func
    
    def run(self):
        scores = []
        for model in self.models:
            scores.append(["Original", model.__name__, model(self.original_data)])
        for completer in self.completers:
            completed_data = completer(self.missing_data)
            for model in self.models:
                scores.append([completer.__name__, model.__name__, model(completed_data)])
        print("All tests complete")
        self.scores = pd.DataFrame(data=scores, columns=["Completers Functions", "Models", "Scores"], index=None)
        print(self.scores.to_string())

    def plot(self, size=(12, 8), save_file_name=None):
        plt.figure(figsize=size)
        plt.title("Comparison on [{0}] with [{1}]".format(self.data_func_name, self.random_func_name))
        ax = sns.barplot(x=self.scores["Completers Functions"], y=self.scores["Scores"], hue="Models", data=self.scores, palette="magma")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ".2f"), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="center", xytext=(0, 10), textcoords="offset points")
        plt.ylim([0.0, 1.5])
        if save_file_name is not None:
            plt.savefig(save_file_name)
        plt.show()

def printBar():
    w, _ = shutil.get_terminal_size()
    w -= 1
    print("-"*w)

if __name__ == "__main__":
    dataset_prepare()
    machine = TestMachine(create_iris_dataset, gen_complete_random)
    machine.run()
    machine.plot()
