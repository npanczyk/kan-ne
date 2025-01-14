# class to train the KAN
# methods to get all necessary metrics (MAE, MAPE, MSE, RMSE, RSMPE, R2)
# method to print symbolic function
# method to generate plot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from kan import *
from kan.utils import ex_round
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class NKAN:
    def __init__(self, dataset, seed, device, hidden_nodes=None, k=3, grid=5):
        self.dataset = dataset
        self.seed = seed
        self.device = device
        if hidden_nodes:
            self.hidden_nodes = hidden_nodes
        else:
            self.hidden_nodes = dataset['train_input'].shape[1]

    def get_model():
        return

    def get_metrics():
        return

    def get_schematic():
        return 

    def get_equation():
        return

    def get_importances():
        return

    