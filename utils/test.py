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
                 record_time=False, n_jobs=1,
                 grid_search=False):
        self.data_func_name = data_func.__name__
        self.random_func_name = random_func.__name__
        self.record_time = record_time
        self.original_data = data_func(print_time=self.record_time)
        self._log_message("Dataset loaded")
        assert random_gen >= 1 # make sure at least one
        if random_gen > 1:
            print("You're using random generation > 1: {}".format(random_gen))
            self.missing_data = [random_func(self.original_data, print_time=self.record_time) for _ in range(random_gen)]
        else:
            self.missing_data = random_func(self.original_data, print_time=self.record_time)
        self._log_message("Missing values generated")
        self.completers = [
                           complete_by_value,
                           complete_by_mean_col,
                           complete_by_nearby_row,
                           complete_by_similar_row,
                           complete_by_most_freq,
                           complete_by_multi
                           ] if complete_func == [] else complete_func
        self.models = [
                        KNN,
                        SGD,
                        DecisionTree,
                        SVM,
                        Forest
                        ] if model_func == [] else model_func
        self.predictor_cv = predictor_cv
        self.n_jobs = n_jobs
        self.grid_search = grid_search
        if not os.path.exists(os.path.join("img", "TestMachine")):
            os.makedirs(os.path.join("img", "TestMachine"))
    
    def run(self):
        scores = []
        self._log_message("Run on original dataset")
        for model in self.models:
            scores.append(["original", model.__name__, model(self.original_data, self.predictor_cv, self.record_time, self.grid_search, self.n_jobs)])
        self._log_message("Original dataset scores collected")
        for completer in self.completers:
            self._log_message("Run on missing dataset")
            if type(self.missing_data) is list:
                completed_data = [completer(x, print_time=self.record_time) for x in self.missing_data]
                for model in self.models:
                    model_score = [model(cc, self.predictor_cv, self.record_time, self.grid_search, self.n_jobs) for cc in completed_data] if type(completed_data[0]) is not list \
                            else [np.mean([model(m, self.predictor_cv, self.record_time, self.grid_search, self.n_jobs) for m in cc]) for cc in completed_data]
                    scores.append([completer.__name__, model.__name__, sum(model_score)/len(model_score)])
            else:
                completed_data = completer(self.missing_data, print_time=self.record_time)
                for model in self.models:
                    model_score = model(completed_data, self.predictor_cv, self.record_time, self.grid_search, self.n_jobs) if type(completed_data) is not list \
                        else np.mean([model(cc, self.predictor_cv, self.record_time, self.grid_search, self.n_jobs) for cc in completed_data])
                    scores.append([completer.__name__, model.__name__, model_score])
            self._log_message("Scores collected")
        print("All tests complete")
        self.scores = pd.DataFrame(data=scores, columns=["Completer Functions", "Models", "Scores"], index=None)
        self._log_message("Score dataframe generated")
        print(self.scores.to_string())

    def plot_compare_models(self, size=None, save_file_name=None):
        if size is None:
            plt.figure(figsize=self._calc_figsize())
        else:
            plt.figure(figsize=size)
        plt.title("Model comparison on [{0}] with [{1}]".format(self.data_func_name, self.random_func_name))
        ax = sns.barplot(x=self.scores["Completer Functions"], y=self.scores["Scores"], hue="Models", data=self.scores, palette="magma")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="center", xytext=(0, 10), textcoords="offset points")
        plt.ylim([0.0, 1.5])
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(os.path.join("img", "TestMachine", save_file_name))
        plt.show()
    
    def plot_compare_completers(self, size=None, save_file_name=None):
        if size is None:
            plt.figure(figsize=self._calc_figsize())
        else:
            plt.figure(figsize=size)
        plt.title("Completer comparison on [{0}] with [{1}]".format(self.data_func_name, self.random_func_name))
        ax = sns.barplot(x=self.scores["Models"], y=self.scores["Scores"], hue="Completer Functions", data=self.scores, palette="magma")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="center", xytext=(0, 10), textcoords="offset points")
        plt.ylim([0.0, 1.5])
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(os.path.join("img", "TestMachine", save_file_name))
        plt.show()

    def _calc_figsize(self):
        Y = 5 + (max(len(self.completers), len(self.models)) + 1) / 2
        X = (len(self.completers) * len(self.models) + 1.8*(max(len(self.completers), len(self.models)) - 1)) / 2
        return (X, Y)

    def _log_message(self, message):
        print("TestMachine: {}".format(message))

class BiasDatasetTest:
    def __init__(self, dataset_func=[], models=[],
                 predictor_cv=5, record_time=False,
                 search_best_model=True, n_jobs=1):
        self.predictor_cv = predictor_cv
        self.record_time = record_time
        self.grid_search = search_best_model
        self.n_jobs = n_jobs
        self.dataset_func = [
            # create_heart_dataset,
            create_adult_dataset,
            create_bank_dataset,
            create_drug_dataset
        ] if dataset_func == [] else dataset_func
        self._gen_data()
        self.models = [
            KNN,
            SGD,
            DecisionTree,
            # SVM,
            Forest
        ] if models == [] else models
        if not os.path.exists(os.path.join("img", "BiasDatasetTest")):
            os.makedirs(os.path.join("img", "BiasDatasetTest"))

    def _gen_data(self):
        self._log_message("Start loading datasets")
        data = []
        for func in self.dataset_func:
            data.append(func(print_time=self.record_time))
            assert data[-1].protected is not None
        self.data = data
        self._log_message("All datasets loaded")

    def plot_confusion_mat(self, savefig=False):
        size = self._calc_confusion_size()
        for dd in self.data:
            self._log_message("Now working on {} data".format(dd.name))
            self._log_message("Now training {} models".format("best" if self.grid_search else "normal"))
            f, ax = plt.subplots(size[3], size[2], figsize=(size[0], size[1]))
            for i in range(size[3]):
                for j in range(size[2]):
                    if (j + i*size[2]) >= len(self.models):
                        ax[i, j].axis("off")
                        continue
                    _, estimator = self.models[j + i*size[2]](dd, self.predictor_cv, print_time=self.record_time, grid_search=self.grid_search, n_jobs=self.n_jobs, return_model=True)
                    ax[i, j].set_title(self.models[j + i*size[2]].__name__)
                    y_true = dd.y
                    y_pred = estimator.predict(dd.X)
                    conf_mat = confusion_matrix(y_true, y_pred)
                    class_names = dd.encoder.inverse_transform(np.arange(len(np.unique(y_true))))
                    df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
                    heatmap = sns.heatmap(df_cm, annot=True, ax=ax[i, j], fmt="d", cmap=plt.cm.Blues)
                    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
                    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
                    ax[i, j].set_xlabel("Predicted Label")
                    ax[i, j].set_ylabel("True Label")
            f.suptitle("Confusion matrixes for {} data".format(dd.name))
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            self._log_message("Confusion graph generated for {} data".format(dd.name))
            if savefig:
                plt.savefig(os.path.join("img", "BiasDatasetTest", dd.name + "_cm.png"))
            plt.show()

    def plot_confusion_mat_protected(self, savefig=False):
        for dd in self.data:
            assert dd.protected is not None
            self._log_message("Now working on {} data, protected features: {}".format(dd.name, dd.protected))
            self._log_message("Now training {} models".format("best" if self.grid_search else "normal"))
            for feature in dd.protected:
                unique_features = dd.X[feature].unique()
                unique_features_names = dd.encoders[feature].inverse_transform(unique_features)
                self._log_message("Now working on {} with {} unique values: {}".format(feature, len(unique_features), unique_features_names))
                if len(unique_features) > 20:
                    print("ERROR: [plot_confusion_mat_protected] feature {} have > 20 unique values ({})\nFailed to plot".format(feature, len(unique_features)))
                    return
                size = self._calc_confusion_size_protected(num_features=len(unique_features))
                f, ax = plt.subplots(size[3], size[2], figsize=(size[0], size[1]))
                y_true = dd.y
                for j in range(size[2]):
                    _, estimator = self.models[j](dd, self.predictor_cv, print_time=self.record_time, grid_search=self.grid_search, n_jobs=self.n_jobs, return_model=True)
                    y_pred = estimator.predict(dd.X)
                    for i in range(size[3]):
                        ax[i, j].set_title(self.models[j].__name__)
                        index_selected = dd.X.index[dd.X[feature] == unique_features[i]].tolist()
                        y_true_selected = np.array(y_true)[index_selected]
                        y_pred_selected = np.array(y_pred)[index_selected]
                        conf_mat = confusion_matrix(y_true_selected, y_pred_selected, labels=np.unique(y_true_selected))
                        class_names = dd.encoder.inverse_transform(np.unique(y_true_selected))
                        df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
                        heatmap = sns.heatmap(df_cm, annot=True, ax=ax[i, j], fmt="d", cmap=plt.cm.Blues)
                        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
                        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
                        ax[i, j].set_xlabel("Predicted Label")
                        if j == 0:
                            ax[i, j].set_ylabel("True Label ({})".format(unique_features_names[i]))
                        else:
                            ax[i, j].set_ylabel("True Label")
                f.suptitle("Confusion matrixes for {} data ({})".format(dd.name, feature), size=20)
                plt.tight_layout()
                plt.subplots_adjust(top=(1-3/size[1]))
                self._log_message("Confusion graph generated for {} data, protected feature: {}".format(dd.name, feature))
                if savefig:
                    plt.savefig(os.path.join("img", "BiasDatasetTest", dd.name + "_" + feature + "_cm.png"))
                plt.show()

    def _calc_confusion_size(self):
        width_unit = math.ceil(len(self.models) ** 0.5)
        height_unit = math.ceil(len(self.models) / width_unit)
        graph_width = 5
        graph_height = 5
        return (width_unit * graph_width, height_unit * graph_height, width_unit, height_unit)

    def _calc_confusion_size_protected(self, num_features=None):
        if num_features is None:
            return self._calc_confusion_size()
        assert num_features >= 1
        width_unit = len(self.models)
        height_unit = num_features
        graph_width = 5
        graph_height = 5
        return (width_unit * graph_width, height_unit * graph_height, width_unit, height_unit)

    def _log_message(self, message):
        print("BiasDatasetTest: {}".format(message))