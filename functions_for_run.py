import subprocess
from datetime import datetime
import os
import subprocess
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import random
from tqdm import tqdm
import glob
from mpl_toolkits.mplot3d import Axes3D

def generate_randomseed(number_of_seed, range_floor = 1, range_upper = 1000000):
    random_numbers = []
    
    for _ in range(number_of_seed):
        random_number = random.randint(range_floor, range_upper)
        random_numbers.append(random_number)
    
    return random_numbers

def mkdir_1(path):
    try:
        os.makedirs(path)
        # print(f"Folder '{path}' created successfully") ## used for tests 
    except FileExistsError:
        exist_warning = f"Folder '{path}' already exists"
        # print(exist_warning) ## used for tests 
    return path
