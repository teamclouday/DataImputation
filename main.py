# this file contains codes that test various research targets

import shutil
from utils.data import *
from utils.generator import *
from utils.completer import *

def printBar():
    w, _ = shutil.get_terminal_size()
    w -= 1
    print("-"*w)

def test():
    data = create_test_dataset()
    print(data.X.to_string())
    printBar()
    data_c_r = gen_complete_random(data)
    print(data_c_r.X.to_string())
    printBar()
    data_complete = complete_by_nearby_row(data_c_r)
    print(data_complete.X.to_string())

def test_iris():
    data = create_iris_dataset()
    print(data.X.head(50).to_string())
    printBar()
    data_c_r = gen_complete_random(data)
    print(data_c_r.X.head(50).to_string())
    data_complete = complete_by_mean_col(data)
    print(data_complete.X.head(50).to_string())

def test_bank():
    data = create_bank_dataset()
    print(data.X.head(50).to_string())
    printBar()
    data_c_r = gen_complete_random(data)
    print(data_c_r.X.head(50).to_string())
    printBar()
    data_complete = complete_by_mean_col(data_c_r)
    print(data_complete.X.head(50).to_string())

def test_adult():
    data = create_adult_dataset()
    print(data.X.head(50).to_string())

if __name__ == "__main__":
    dataset_prepare()
    test_bank()
