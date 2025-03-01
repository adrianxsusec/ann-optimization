import pandas as pd
import numpy as np
from line_profiler import profile

def pandas_read(filename):
    return pd.read_csv(filename).values

def original_read(filename):
    return np.genfromtxt(filename, delimiter=",")

def np_read(filename):
    return np.loadtxt(filename, delimiter=",")

@profile
def main():
    filename_train = '../ann-optimization/train.csv'
    filename_test = '../ann-optimization/test.csv'

    pandas_read(filename_train)
    pandas_read(filename_test)
    np_read(filename_train)
    np_read(filename_test)
    original_read(filename_train)
    original_read(filename_test)

if __name__ == '__main__':
    main()