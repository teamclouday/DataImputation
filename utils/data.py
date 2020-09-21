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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
        os.path.join("dataset", "bank"),
        os.path.join("dataset", "adult"),
        os.path.join("dataset", "compas"),
        #os.path.join("dataset", "heart"),
        #os.path.join("dataset", "drug"),
        os.path.join("dataset", "titanic"),
        os.path.join("dataset", "german"),
        os.path.join("dataset", "communities"),
        #os.path.join("dataset", "juvenile"),
    ]
    load_functions = [
        #_dataset_download_iris,
        _dataset_download_bank,
        _dataset_download_adult,
        _dataset_download_compas,
        #_dataset_download_heart,
        #_dataset_download_drug,
        _dataset_download_titanic,
        _dataset_download_german,
        _dataset_download_communities,
        #_dataset_download_juvenile,
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
    protected_features = [
        #"marital-status",
        #"race",
        "sex"
    ]
    observed_features = ["age", "workclass"]
    data = pd.read_csv(os.path.join("dataset", "adult", "adult.data"), header=None)
    X = data.iloc[:, :-1].copy()
    X.columns = names
    # remove unpredictive columns
    X = X.drop(["fnlwgt", "capital-gain", "capital-loss", "native-country"], axis=1).copy()
    y = data.iloc[:, -1].copy()
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("adult", X, y, protected_features=protected_features, observed_features=observed_features)

def create_bank_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    protected_features = ["age"]
    observed_features = ["job", "marital"]
    data = pd.read_csv(os.path.join("dataset", "bank", "bank-full.csv"), sep=";")
    data["age"] = data["age"].apply(lambda x: "elder" if x >= 35 else "young")
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy().to_numpy().ravel()
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("bank", X, y, protected_features=protected_features, observed_features=observed_features)

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
    return Dataset("drug_"+target_drug, X, y, protected_features=protected_features)

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
              # "c_charge_desc",
              "days_b_screening_arrest",
              "sex",
              "race"]].copy()
    X["length_of_stay"] = pd.to_datetime(data["c_jail_out"]) - pd.to_datetime(data["c_jail_in"])
    X["length_of_stay"] = X["length_of_stay"] / pd.Timedelta(hours=1)
    y = data[["is_recid"]].copy().to_numpy().ravel()
    protected_features = [
        "race",
        #"sex"
    ]
    observed_features = ["age", "age_cat"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("compas", X, y, protected_features=protected_features, observed_features=observed_features)

def create_titanic_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    data = pd.read_csv(os.path.join("dataset", "titanic", "train.csv"))
    X = data.drop(["Survived", "Cabin", "PassengerId", "Ticket", "Name"], axis=1)
    # X["Name"] = X["Name"].apply(lambda x: x.split(",")[1].split()[0])
    y = data[["Survived"]].copy().to_numpy().ravel()
    protected_features = ["Sex"]
    observed_features = ["Age", "Pclass"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("titanic", X, y, protected_features=protected_features, observed_features=observed_features)

def create_german_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    names = [
        "Status_account", "Duration_month", "Credit_history", "Purpose", "Credit_amount",
        "Savings_account", "Employment_since", "Installment_rate", "Personal_status",
        "Debtors_guarantors", "Residence_since", "Property", "Age", "Installment_plans",
        "Housing", "Number_credits", "Job", "Num_liable_people", "Telephone", "Foreign",
        "Target"
    ]
    data = pd.read_csv(os.path.join("dataset", "german", "german.data"), names=names, sep=" ")
    # combine sex data
    # data["Personal_status_sex"] = data["Personal_status_sex"].apply(lambda x: "male" if x in ["A91", "A93", "A94"] else "female")
    
    # categorize Age feature
    data["Age"] = data["Age"].apply(lambda x: "elder" if x >= 26 else "young")

    X = data.drop(["Target"], axis=1).copy()
    y = data[["Target"]].copy().to_numpy().ravel()
    protected_features = ["Age"]
    observed_features = ["Job", "Purpose"]
    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("german", X, y, protected_features=protected_features, observed_features=observed_features)

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
    data = pd.read_csv(os.path.join("dataset", "communities", "communities.data"), names=names, na_values=["?"])

    # remove unpredictive attributes
    data = data.drop(["state", "county", "community", "communityname", "fold"], axis=1)
    # remove attributes with too many missing values
    data = data.drop(['LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop',
        'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol',
        'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor',
        'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg',
        'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'PolicBudgPerPop'], axis=1)
    
    def helper_max(array, names):
        return [names[np.argmax(dd)] for dd in array]
    # combine race percentage attributes
    data["race_c"] = helper_max(data[["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"]].to_numpy(), ["black", "white", "asian", "hispanic"])
    data = data.drop(["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"], axis=1)

    X = data.drop(["ViolentCrimesPerPop"], axis=1).copy()
    y_mean = data["ViolentCrimesPerPop"].mean()
    y = data["ViolentCrimesPerPop"].apply(lambda m: "Violent" if m >= y_mean else "Harmless").to_numpy().ravel()
    protected_features = ["race_c"]
    observed_features = ["population", "PopDens"]

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("communities", X, y, protected_features=protected_features, observed_features=observed_features)

def create_juvenile_dataset(print_time=False):
    if print_time:
        tt = time.process_time()
    data = pd.read_excel(os.path.join("dataset", "juvenile", "reincidenciaJusticiaMenors.xlsx"), index_col=None)

    # remove attributes with too many missing values
    # data = data[['id','V1_sexe','V2_estranger','V3_nacionalitat','V5_edat_fet_agrupat','V6_provincia',
    #     'V7_comarca','V8_edat_fet','V9_edat_final_programa','V10_data_naixement','V11_antecedents',
    #     'V13_nombre_fets_agrupat','V14_fet','V15_fet_agrupat','V16_fet_violencia','V17_fet_tipus',
    #     'V19_fets_desagrupats','V21_fet_nombre','V22_data_fet','V23_territori','V24_programa',
    #     'V25_programa_mesura','V27_durada_programa_agrupat','V28_temps_inici','V29_durada_programa',
    #     'V30_data_inici_programa','V31_data_fi_programa','V115_reincidencia_2015','V122_rein_fet_2013',
    #     'V132_REINCIDENCIA_2013']]
    data = data[[a for a,b in data.isnull().sum().to_dict().items() if b <= len(data)/2]]
    # remove unpredictive attributes
    data = data.drop(['id', 'V7_comarca', 'V9_edat_final_programa', 'V10_data_naixement',
        'V22_data_fet', 'V30_data_inici_programa', 'V31_data_fi_programa'], axis=1)
    # remove recidivism variables
    data = data.drop(['V115_reincidencia_2015','V122_rein_fet_2013'], axis=1)

    X = data.drop(["V132_REINCIDENCIA_2013"], axis=1).copy()
    y = data[["V132_REINCIDENCIA_2013"]].copy().to_numpy().ravel()
    protected_features = ['V1_sexe']

    if print_time:
        print("Performance Monitor: ({:.4f}s) ".format(time.process_time() - tt) + inspect.stack()[0][3])
    return Dataset("juvenile", X, y, protected_features=protected_features)

class Dataset:
    def __init__(self, name, X, y, types=None, protected_features=[], observed_features=[], categorical_features=None, encoders=None):
        """
        `X`: a pandas dataframe storing original dataset\\
        `y`: a numpy flat array of targets\\
        `types`: set original data types for X (used in imputation methods)\\
        `protected_features`: pre-defined protected features\\
        `observed_features`: pre-defined observed features for MAR\\
        `categorical_features`: a python list. if None, automatically detect categorical features, else store the categorical_features\\
        `encoders`: set X and y label encoders, set to None to trigger encoding
        """
        self.name = name
        self.X = X
        self.y = y
        self.X_encoded = None
        if len(protected_features) > 0:
            assert len(protected_features) == len([x for x in protected_features if x in self.X.columns.tolist()])
        if len(observed_features) > 0:
            assert len(observed_features) == len([x for x in observed_features if ((x in self.X.columns.tolist()) and (x not in protected_features))])
        self.protected_features = protected_features
        self.observed_features = observed_features
        self.categorical_features = categorical_features
        if encoders is None:
            self.X_encoders, self.y_encoder = self._convert_categories()
        else:
            self.X_encoders = encoders[0]
            self.y_encoder = encoders[1]
        if types is not None:
            self.types = types
        else:
            self.types = X.dtypes
            for category in self.categorical_features:
                self.types[category] = np.int32 # set desired dtype to be int32

    def _convert_categories(self):
        columns = self.X.columns
        columns_for_convert = self.categorical_features
        if columns_for_convert is None:
            columns_for_convert = []
            for col in columns:
                # skip protected features
                if col in self.protected_features:
                    continue
                if not is_numeric_dtype(self.X[col]):
                    columns_for_convert.append(col)
        self.categorical_features = columns_for_convert
        X_encoders = {}
        # credit: https://stackoverflow.com/questions/54444260/labelencoder-that-keeps-missing-values-as-nan
        for col in columns_for_convert:
            encoder = LabelEncoder()
            series = self.X[col]
            self.X[col] = pd.Series(encoder.fit_transform(series[series.notnull()]), index=series[series.notnull()].index) # keep nan, convert convert others
            X_encoders[col] = encoder
        # convert for y values also
        y_encoder = LabelEncoder()
        y_encoder.fit(self.y)
        self.y = y_encoder.transform(self.y)
        return (X_encoders, y_encoder)
        
    def copy(self):
        """
        Duplicate the dataset object
        """
        return Dataset(self.name, self.X.copy(), self.y.copy(), 
            types=self.types, protected_features=self.protected_features,
            categorical_features=self.categorical_features,
            observed_features=self.observed_features,
            encoders=[self.X_encoders, self.y_encoder])

    def preprocess(self):
        """
        Apply one-hot-encoding on categorical features before feeding into classifiers
        """
        assert self.categorical_features is not None
        self.X_encoded = self.X.copy()
        for category in self.categorical_features:
            expected_categories = list(range(len(self.X_encoders[category].classes_)))
            # clip the values in this category
            self.X_encoded[category].clip(lower=min(expected_categories), upper=max(expected_categories), inplace=True)
            # setup one-hot-encoder
            ohe = OneHotEncoder(categories=[expected_categories], sparse=False)
            encoded = pd.DataFrame(ohe.fit_transform(self.X_encoded[category].to_numpy().reshape(-1, 1)), columns=ohe.get_feature_names([category]))
            # drop first column in order for better classifier performance
            encoded = encoded.drop(columns=[ohe.get_feature_names([category])[0]])
            # combine the new column
            self.X_encoded = pd.concat([self.X_encoded, encoded], axis=1).drop([category], axis=1)
        # self.X_encoded = pd.get_dummies(self.X, columns=self.categorical_features, prefix_sep="=")

    def get_missing_ratio(self):
        """
        Compute the missing percentage
        """
        return self.X.isnull().sum().sum() / self.X.size