# this file contains functions that prepares dataset

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd

# download iris dataset
def _dataset_download_iris(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
    FILE_NAMES = ["iris.data", "iris.names"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH+name, os.path.join(folder, name))
    print("Iris dataset is downloaded")

# download bank dataset
def _dataset_download_bank(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/"
    FILE_NAME = "bank.zip"
    if not os.path.exists(folder):
        os.makedirs(folder)
    urllib.request.urlretrieve(URL_PATH + FILE_NAME, os.path.join(folder, FILE_NAME))
    with zipfile.ZipFile(os.path.join(folder, FILE_NAME), "r") as zipF:
        zipF.extractall(folder)
    print("Bank dataset is downloaded")

# download adult dataset
def _dataset_download_adult(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    FILE_NAMES = ["adult.data", "adult.names", "adult.test"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    print("Adult dataset is downloaded")

# function that checks for existence of datasets
def dataset_prepare():
    dataset_folders = [
        os.path.join("dataset", "iris"),
        os.path.join("dataset", "bank"),
        os.path.join("dataset", "adult")
    ]
    load_functions = [
        _dataset_download_iris,
        _dataset_download_bank,
        _dataset_download_adult
    ]
    for folder, func in zip(dataset_folders, load_functions):
        if not os.path.exists(folder):
            func(folder)

# create a dataset for testing
def create_test_dataset(size=(30, 8)):
    if len(size) != 2 or size[0] < 1 or size[1] < 1:
        print("Error: create_test_dataset, wrong values in size")
        sys.exit(1)
    names = np.arange(size[1])
    data = np.random.permutation(size[0]*size[1]).reshape(size[0], size[1]).astype(np.float32)
    data = pd.DataFrame(data, columns=names)
    labels = np.arange(size[0], dtype=np.int32)
    return Dataset("test", data, labels)

# create a class object
def create_adult_dataset():
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ]
    data = pd.read_csv(os.path.join("dataset", "adult", "adult.data"), header=None)
    X = data.iloc[:, :-1].copy()
    X.columns = names
    y = data.iloc[:, -1].copy()
    return Dataset("adult", X, y)

def create_bank_dataset():
    data = pd.read_csv(os.path.join("dataset", "bank", "bank-full.csv"), sep=";")
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    return Dataset("bank", X, y)

def create_iris_dataset():
    names = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class"
    ]
    data = pd.read_csv(os.path.join("dataset", "iris", "iris.data"), names=names)
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    return Dataset("iris", X, y)

class Dataset:
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y

    def copy(self):
        return Dataset(self.name, self.X.copy(), self.y.copy())
