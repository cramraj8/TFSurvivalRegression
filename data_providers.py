# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd


def data_providers(x_data_file='./data/Brain_Integ_X.csv',
                   y_data_file='./data/Brain_Integ_Y.csv'):
    """
    This function reads the data file and extracts the features and labelled
    values.
    Then according to that patient is dead, only those observations would be
    taken care
    for deep learning trainig.

    Args:
        data_file: list of strings representing the paths of input files. Here the input features and ground truth values are separated in 2 files.
    Returns:
        `Numpy array`, extracted feature columns and label column.
    Example:
        >>> read_dataset()
        ( [[2.3, 2.4, 6.5],[2.3, 5.4,3.3]], [12, 82] )
    """

    data_feed = pd.read_csv(x_data_file, skiprows=[0], header=None)
    labels_feed = pd.read_csv(y_data_file, skiprows=[1], header=0)
    survival = labels_feed['Survival']
    censored = labels_feed['Censored']

    survival = survival.values
    censored = censored.values
    data = data_feed.values
    data = np.float32(data)

    censored_survival = survival[censored == 1]
    censored_features = data[censored == 1]
    censored_data = censored[censored == 1]

    y = np.asarray(censored_survival, dtype=np.int32)
    x = np.asarray(censored_features)
    c = np.asarray(censored_data, dtype=np.int32)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    print('Shape of C : ', c.shape)
    return (x, y, c)


if __name__ == '__main__':

    X_DATA_FILE = './data/Brain_Integ_X.csv'
    Y_DATA_FILE = './data/Brain_Integ_Y.csv'
    data_x, data_y, c = data_providers(X_DATA_FILE, Y_DATA_FILE)
