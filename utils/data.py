# this file contains functions that prepares dataset

import os
import sys
import time
import math
import kaggle
import sqlite3
import inspect
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from functools import partial
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

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

# download compas dataset
def _dataset_download_compas(folder):
    URL_PATH = "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    FILE_NAMES = ["compas-scores-two-years.csv", "compas-scores-two-years-violent.csv", "compas-scores.csv", "cox-parsed.csv", "cox-violent-parsed.csv", "compas.db"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    conn = sqlite3.connect(os.path.join("dataset", "compas", "compas.db"))
    table_names = ["casearrest", "compas", "people", "summary", "charge", "jailhistory", "prisonhistory"]
    for name in table_names:
        df = pd.read_sql_query("SELECT * FROM " + name, conn)
        df.to_csv(os.path.join("dataset", "compas", "compas.db." + name + ".csv"), index=False)
    conn.close()
    print("Compas dataset is downloaded")

# download heart dataset
def _dataset_download_heart(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    FILE_NAMES = ["heart-disease.names", "new.data"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    # convert new.data
    outFile = open(os.path.join(folder, "converted.data"), "w")
    with open(os.path.join(folder, FILE_NAMES[1]), "r") as inFile:
        new_line = []
        for line in inFile.readlines():
            line = line.rstrip().split()
            if len(line) == 3:
                line = ",".join(line)
                new_line.append(line)
                new_line = ",".join(new_line)
                print(new_line, file=outFile)
                new_line = []
            else:
                line = ",".join(line)
                new_line.append(line)
    outFile.close()
    print("Heart Disease dataset is downloaded")

# download drug dataset
def _dataset_download_drug(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/"
    FILE_NAMES = ["drug_consumption.data"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    print("Drug Consumption dataset is downloaded")

# download titanic dataset
def _dataset_download_titanic(folder):
    try:
        kaggle.api.authenticate()
    except Exception as e:
        print("Error: Kaggle Authentication Failed\n{}".format(e))
    if not os.path.exists(folder):
        os.makedirs(folder)
    kaggle.api.competition_download_files("titanic", folder)
    with zipfile.ZipFile(os.path.join(folder, "titanic.zip"), "r") as zipF:
        zipF.extractall(folder)
    print("Titanic dataset is downloaded")

# download german credit dataset
def _dataset_download_german(folder):
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
    FILE_NAMES = ["german.data", "german.data-numeric", "german.doc"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    print("German credit dataset is downloaded")

# download communities and crime dataset
def _dataset_download_communities(folder):
    URL_PATH = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/"
    FILE_NAMES = ["communities.data", "communities.names"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    print("Communities and crime dataset is downloaded")

# download Recidivism in juvenile justice dataset
def _dataset_download_juvenile(folder):
    URL_PATH = "http://cejfe.gencat.cat/web/.content/home/recerca/opendata/jjuvenil/reincidenciaJusticiaMenors/"
    FILE_NAMES = ["reincidenciaJusticiaMenors.xlsx", "recidivismJuvenileJustice_variables_EN.pdf"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in FILE_NAMES:
        urllib.request.urlretrieve(URL_PATH + name, os.path.join(folder, name))
    print("Recidivism in juvenile justice dataset is downloaded")

# function that checks for existence of datasets
def dataset_prepare(force_download=False):
    dataset_folders = [
        #os.path.join("dataset", "iris"),
        #os.path.join("dataset", "bank"),
        os.path.join("dataset", "adult"),
        os.path.join("dataset", "compas"),
        #os.path.join("dataset", "heart"),
        #os.path.join("dataset", "drug"),
        os.path.join("dataset", "titanic"),
        os.path.join("dataset", "german"),
        os.path.join("dataset", "communities"),
        os.path.join("dataset", "juvenile"),
    ]
    load_functions = [
        #_dataset_download_iris,
        #_dataset_download_bank,
        _dataset_download_adult,
        _dataset_download_compas,
        #_dataset_download_heart,
        #_dataset_download_drug,
        _dataset_download_titanic,
        _dataset_download_german,
        _dataset_download_communities,
        _dataset_download_juvenile,
    ]
    for folder, func in zip(dataset_folders, load_functions):
        if not force_download:
            if not os.path.exists(folder):
                func(folder)
        else:
            func(folder)

# create a dataset for testing
def create_test_dataset(size=(30, 8), print_time=False):
    if print_time:
        tt = time.process_time()
    if len(size) != 2 or size[0] < 1 or size[1] < 1:
        print("Error: create_test_dataset, wrong values in size")
        sys.exit(1)
    names = np.arange(size[1])
    data = np.random.permutation(size[0]*size[1]).reshape(size[0], size[1]).astype(np.float32)
    data = pd.DataFrame(data, columns=names)
    labels = np.arange(size[0], dtype=np.int32)
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("test", data, labels)

# create a class object
def create_adult_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
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
    protected_features = ["education", "marital-status", "race", "sex"]
    data = pd.read_csv(os.path.join("dataset", "adult", "adult.data"), header=None)
    X = data.iloc[:, :-1].copy()
    X.columns = names
    y = data.iloc[:, -1].copy()
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("adult", X, y, protected_features=protected_features)

def create_bank_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    protected_features = ["marital", "education"]
    data = pd.read_csv(os.path.join("dataset", "bank", "bank-full.csv"), sep=";")
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("bank", X, y, protected_features=protected_features)

def create_iris_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
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
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("iris", X, y)

def create_heart_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    names = {
        3:"age",
        4:"sex",
        9:"cp",
        10:"trestbps",
        12:"chol",
        16:"fbs",
        19:"restecg",
        32:"thalach",
        38:"exang",
        40:"oldpeak",
        41:"slope",
        44:"ca",
        51:"thal",
        58:"num"
    }
    data = pd.read_csv(os.path.join("dataset", "heart", "converted.data"), names=list(names.values()), usecols=[(x-1) for x in list(names.keys())])
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    # return Dataset("heart", X, y)
    raise Exception("Heart dataset cannot be used")
    return None

def create_drug_dataset(print_time=False, target_drug="Heroin"):
    if print_time:
        tt = time.process_time()
    names = [
        "ID", "Age Range", "Gender", "Education", "Country", "Ethnicity",
        "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS",
        "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke",
        "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth",
        "Mushrooms", "Nicotine", "Semer", "VSA"
    ]
    labels = [
        "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke",
        "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth",
        "Mushrooms", "Nicotine", "Semer", "VSA"
    ]
    assert target_drug in labels
    labels = {x:y for (y,x) in enumerate(labels)}
    data = pd.read_csv(os.path.join("dataset", "drug", "drug_consumption.data"), names=names)
    X = data.iloc[:, :-len(labels)].copy()
    y = data.iloc[:, -(labels[target_drug])].copy()
    y.replace(["CL0", "CL1"], "NonUser", inplace=True)
    y.replace(["CL2", "CL3", "CL4", "CL5", "CL6"], "User", inplace=True)
    protected_features = ["Gender", "Education", "Ethnicity"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("drug_"+target_drug, X, y, convert_all=True, protected_features=protected_features)

def create_compas_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    data = pd.read_csv(os.path.join("dataset", "compas", "compas-scores.csv"))
    data = data[(data['days_b_screening_arrest'].isnull()) | (data['days_b_screening_arrest'] <= 30)]
    data = data[(data['days_b_screening_arrest'].isnull()) | (data['days_b_screening_arrest'] >= -30)] # select within 30 days
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] != 'O']
    data.reset_index(drop=True, inplace=True)
    X = data[["age",
              "age_cat",
              "c_charge_degree",
              "priors_count",
              "juv_misd_count",
              "juv_fel_count",
              "juv_other_count",
              "c_charge_desc",
              "days_b_screening_arrest",
              "sex",
              "race"]].copy()
    X["length_of_stay"] = pd.to_datetime(data["c_jail_out"]) - pd.to_datetime(data["c_jail_in"])
    X["length_of_stay"] = X["length_of_stay"] / pd.Timedelta(hours=1)
    y = data[["is_recid"]].copy().to_numpy().ravel()
    protected_features = ["race", "sex"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("compas", X, y, protected_features=protected_features)

def create_titanic_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    data = pd.read_csv(os.path.join("dataset", "titanic", "train.csv"))
    X = data.drop(["Survived", "Cabin", "PassengerId", "Ticket"], axis=1)
    X["Name"] = X["Name"].apply(lambda x: x.split(",")[1].split()[0])
    y = data[["Survived"]].copy().to_numpy().ravel()
    protected_features = ["Sex"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("titanic", X, y, protected_features=protected_features)

def create_german_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    names = [
        "Status_account", "Duration_month", "Credit_history", "Purpose", "Credit_amount",
        "Savings_account", "Employment_since", "Installment_rate", "Personal_status_sex",
        "Debtors_guarantors", "Residence_since", "Property", "Age", "Installment_plans",
        "Housing", "Number_credits", "Job", "Num_liable_people", "Telephone", "Foreign",
        "Target"
    ]
    data = pd.read_csv(os.path.join("dataset", "german", "german.data"), names=names, sep=" ")
    X = data.drop(["Target"], axis=1).copy()
    y = data[["Target"]].copy().to_numpy().ravel()
    protected_features = ["Personal_status_sex"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("german", X, y, protected_features=protected_features)

def create_communities_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    names = [
        'state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize',
        'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29',
        'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf',
        'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc',
        'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap',
        'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
        'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
        'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par',
        'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
        'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8',
        'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10',
        'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
        'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR',
        'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos',
        'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
        'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
        'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
        'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic',
        'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp',
        'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked',
        'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
        'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'ViolentCrimesPerPop'
    ]
    data = pd.read_csv(os.path.join("dataset", "communities", "communities.data"), names=names)

    print("Communities dataset needs some pre-processing!")

    X = data.drop(["ViolentCrimesPerPop"], axis=1).copy()
    y = data[["ViolentCrimesPerPop"]].copy().to_numpy().ravel()
    protected_features = []

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("communities", X, y, protected_features=protected_features)

def create_juvenile_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    data = pd.ExcelFile(os.path.join("dataset", "juvenile", "reincidenciaJusticiaMenors.xlsx"))
    data = data.parse(data.sheet_names)

    print("Juvenile dataset needs some pre-processing!")

    X = data.drop(["V132_REINCIDENCIA_2013"], axis=1).copy()
    y = data[["V132_REINCIDENCIA_2013"]].copy().to_numpy().ravel()
    protected_features = []

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("juvenile", X, y, protected_features=protected_features)

class Dataset:
    def __init__(self, name, X, y, auto_convert=True, types=None, convert_all=False, protected_features=[], encoders=None):
        # auto_convert: whether to label-encode categorical values
        # types: original data types for X
        # convert_all:  whether to label-encode all attributes (both numerical and categorical)
        # protected_features: pre-defined protected features
        # encoders: list => [X_encoders, y_encoder]
        #           store label encoders for X and y features
        self.name = name
        self.X = X
        if self.X.drop(protected_features, axis=1).isnull().sum().sum() > 0:
            self.has_nan = True
        else:
            self.has_nan = False
        self.y = y
        self.convert_all = convert_all
        if len(protected_features) > 0:
            assert len(protected_features) == len([x for x in protected_features if x in self.X.columns.tolist()])
        self.protected = protected_features
        self.encoder = None
        self.encoders = None
        if auto_convert:
            self.encoder = self._convert_categories()
        if encoders is not None:
            self.encoder = encoders[1]
            self.encoders = encoders[0]
        self.types = X.dtypes
        if types is not None:
            self.types = types

    def _convert_categories(self):
        columns = self.X.columns
        columns_for_convert = []
        if not self.convert_all:
            for col in columns:
                if col in self.protected:
                    continue
                if not is_numeric_dtype(self.X[col]):
                    columns_for_convert.append(col)
        else:
            columns_for_convert = columns
        self.encoders = {}
        # credit: https://stackoverflow.com/questions/54444260/labelencoder-that-keeps-missing-values-as-nan
        for col in columns_for_convert:
            encoder = LabelEncoder()
            series = self.X[col]
            self.X[col] = pd.Series(encoder.fit_transform(series[series.notnull()]), index=series[series.notnull()].index) # keep nan, convert convert others
            self.encoders[col] = encoder
        # convert for y values also
        encoder = LabelEncoder()
        encoder.fit(self.y)
        self.y = encoder.transform(self.y)
        return encoder
        
    def copy(self):
        return Dataset(self.name, self.X.copy(), self.y.copy(), auto_convert=False, types=self.types, protected_features=self.protected, encoders=[self.encoders, self.encoder])