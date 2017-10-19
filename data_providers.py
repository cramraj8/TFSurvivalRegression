# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


DATA_FILE = ['./Brain_Integ_X.csv', './Brain_Integ_Y.csv']

def data_providers(data_file=['./Brain_Integ_X.csv', './Brain_Integ_Y.csv']):
    """The function for reading datasets.

    This function reads the features, lable datasets separately and filters out
    only the dead patients' respective observations for further processings.

  Args:
    data_file: list of strings representing the paths for input files.
    Here the input features and input ground truth values are provided separately.

  Returns:
    `Numpy array`, extracted feature matrix and label column.

  Example:
    >>> read_dataset()
    ( [[2.3, 2.4, 6.5],[2.3, 5.4,3.3]], [12, 82] )

    """

    data_feed = pd.read_csv(data_file[0], skiprows=[0], header=None)
    labels_feed = pd.read_csv(data_file[1], skiprows=[1], header=0)
    survival = labels_feed['Survival']
    censored = labels_feed['Censored']

    survival = survival.values
    censored = censored.values
    data = data_feed.values
    data = np.float32(data)

    #Filtering only the dead patients for survival analysis
    censored_survival = survival[censored == 1]
    censored_data = data[censored == 1]

    y = np.asarray(censored_survival)
    x = np.asarray(censored_data)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    return (x, y)


if __name__ == '__main__':

    data_x, data_y = data_providers(DATA_FILE)
