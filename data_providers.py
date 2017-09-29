import numpy as np
import pandas as pd


def data_providers(data_file=['./Brain_Integ_X.csv', './Brain_Integ_Y.csv']):
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

    data_feed = pd.read_csv(data_file[0], skiprows=[0], header=None)
    labels_feed = pd.read_csv(data_file[1], skiprows=[1], header=0)
    survival = labels_feed['Survival']
    censored = labels_feed['Censored']

    survival = survival.values
    censored = censored.values
    data = data_feed.values
    data = np.float32(data)

    censored_survival = survival[censored == 1]
    censored_data = data[censored == 1]

    y = np.asarray(censored_survival)
    x = np.asarray(censored_data)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    return (x, y)

if __name__ == '__main__':

    data_file = ['./Brain_Integ_X.csv', './Brain_Integ_Y.csv']
    data_x, data_y = data_providers(data_file)
