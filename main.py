# this file contains codes that test various research targets

from utils.data import *
from utils.generator import *

def test():
    data = create_test_dataset()
    print("Original Dataset:")
    print(data.X.to_string())
    data_c_r = gen_complete_random(data)
    print("Complete Random:")
    print(data_c_r.X.to_string())
    data_r = gen_random(data)
    print("Random:")
    print(data_r.X.to_string())

def test_iris():
    data = create_iris_dataset()
    print(data.X.to_string())
    data_c_r = gen_complete_random(data)
    print(data_c_r.X.to_string())

if __name__ == "__main__":
    dataset_prepare()
    test_iris()
