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
    print(data.X.to_string())
    data_c_r = gen_complete_random(data)
    print(data_c_r.X.to_string())

if __name__ == "__main__":
    dataset_prepare()
    test()
