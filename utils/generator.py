# this file contains codes for generating missing values in a dataset

import random
import pandas as pd
from utils.data import *

# generate by complete random
def gen_complete_random(data, random_ratio=0.3):
    if random_ratio > 0.5:
        print("Warning: gen_complete_random, random missing ratio > 0.5")
    