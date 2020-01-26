# this file contains functions that prepares dataset

import os
import urllib
import zipfile

# download iris dataset
def dataset_download_iris(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
    FILE_NAMES = ["iris.data", "iris.names"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH+name, os.path.join(folder, name))
    print("Iris dataset is downloaded")

# download bank dataset
def dataset_download_bank(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/"
    FILE_NAME = "bank.zip"
    if not os.path.exists(folder):
        os.makedirs(folder)
    urllib.request.urlretrieve(URL_PATH + FILE_NAME, os.path.join(folder, FILE_NAME))
    with zipfile.ZipFile(os.path.join(folder, FILE_NAME), "r") as zipF:
        zipF.extractall(folder)
    print("Bank dataset is downloaded")

# download adult dataset
def dataset_download_adult(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    FILE_NAMES = ["adult.data", "adult.names", "adult.test"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    print("Adult dataset is downloaded")