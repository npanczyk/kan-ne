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
from preprocessing import get_chf
from accessories import rmspe, mape


class NKAN:
    def __init__(self, dataset, seed, device, hidden_nodes=None, k=3, grid=5):
        """Class that creates and trains a KAN model based on an input dataset.

        Args:
            dataset (dict): a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output) and feature/output labels.
            seed (int): value to set random state for reproducible results
            device (str): cpu or gpu
            hidden_nodes (list, optional): an ordered list containing the number of nodes in each hidden layer. Defaults to None.
            k (int, optional): the spline order. Defaults to 3.
            grid (int, optional): the number of grid intervals. Defaults to 5.
        """
        self.dataset = dataset
        self.seed = seed
        self.device = device
        if hidden_nodes:
            print("Hidden nodes given manually.")
            self.hidden_nodes = hidden_nodes
        else:
            print("Hidden nodes implied.")
            self.hidden_nodes = [self.dataset['train_input'].shape[1]] # should this be self.dataset?
            print(f'Hidden nodes: {self.hidden_nodes}')
        self.k = k
        self.grid = grid

    def get_model(self):
        """Uses input dataset to train and return a KAN model (without tuning).

        Returns:
            pykan KAN model object: model trained on dataset provided to class
        """
        width = [self.dataset['train_input'].shape[1]] + self.hidden_nodes + [self.dataset['train_output'].shape[1]]
        print(width)
        model = KAN(width=width, grid=self.grid, k=self.k, seed=self.seed, device=self.device)
        data = {
            'train_input':self.dataset['train_input'],
            'train_label':self.dataset['train_output'],
            'test_input':self.dataset['test_input'],
            'test_label':self.dataset['test_output']
        }
        model.fit(data, opt='LBFGS', steps=100, lamb=0.001, lamb_entropy=2)
        print("Model trained.")
        return model

    def get_metrics(self, model, save_as):
        # get predictions and unscale data
        scaler = self.dataset['y_scaler']
        X_test = self.dataset['test_input'] # still scaled
        Y_test = self.dataset['test_output'] # still scaled
        Y_pred = model(X_test)
        print( Y_pred.shape, Y_test.shape)
        y_test = scaler.inverse_transform(Y_test.detach().numpy())
        y_pred = scaler.inverse_transform(Y_pred.detach().numpy())
        metrics = {
            'OUTPUT':self.dataset['output_labels'],
            'MAE':[],
            'MAPE':[],
            'MSE':[],
            'RMSE':[],
            'RMSPE':[],
            'R2':[]
        }
        for i in range(len(self.dataset['output_labels'])):
            # get metrics for each output
            yi_test = y_test[:,i]
            yi_pred = y_pred[:,i]
            metrics['MAE'].append(mean_absolute_error(yi_test, yi_pred))
            metrics['MAPE'].append(mape(yi_test, yi_pred))
            metrics['MSE'].append(mean_squared_error(yi_test, yi_pred))
            metrics['RMSE'].append(np.sqrt(mean_squared_error(yi_test, yi_pred)))
            metrics['RMSPE'].append(rmspe(yi_test, yi_pred))
            metrics['R2'].append(r2_score(yi_test, yi_pred))
        metrics_df = pd.DataFrame.from_dict(metrics)
        # check to see fi there 
        if not os.path.exists('results'):
            os.makedirs('results')
        metrics_df.to_csv(f'results/{save_as}.csv', index=False)
        return metrics_df

    def get_schematic():
        return 

    def get_equation(model, output_label):
        return

    def get_importances():
        return

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset  = get_chf()
    test_kan = NKAN(dataset=dataset, seed=42, device=device)
    model = test_kan.get_model()
    metrics = test_kan.get_metrics(model, 'chf_test')
    print( metrics )