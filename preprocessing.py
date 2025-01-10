# figure out what you want to keep from the pyMAISE _handler.py file
# need to write a class to load datsets, split into training and testing, scale
# the datasets, convert them into Torch tensors, and then create a dataset
# directory

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

# putting this on the backburner for now
'''class data:
    def __init__(self, filepath, column_names, output_names, filepath2=None, train_split=0.7):
        self.filepath = filepath
        self.column_names = column_names
        self.output_names = output_names
        self.split = train_split
        if filepath2:
            self.filepath2 = filepath2

    def load_data(self.filepath, self.filepath2=None):
        if filepath2==None:
            df = pd.read_csv(filepath)'''


def get_chf(synthetic=False):
    if synthetic==False:
        train_df = pd.read_csv('datasets/chf_train.csv')
        test_df = pd.read_csv('datasets/chf_valid.csv')
    else:
        train_df = pd.read_csv('datasets/chf_train_synth.csv')
        test_df = pd.read_csv('datasets/chf_test_synth.csv')
    x_train = train_df.iloc[:, [0, 1, 2, 3, 4, 5]].values  # Input columns (1-6) D, L, P, G, T, Xe
    y_train = train_df.iloc[:, [6]].values  # CHF
    x_test = test_df.iloc[:, [0, 1, 2, 3, 4, 5]].values  
    y_test = test_df.iloc[:, [6]].values

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double)
    train_label = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_label = torch.tensor(Y_test, dtype=torch.double).unsqueeze(1)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    return dataset



dataset = get_chf()
print( dataset['train_input'] )