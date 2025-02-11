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
from sympy import symbols, sympify
from preprocessing import *
from accessories import *
from plotting import plot_feature_importances


class NKAN:
    def __init__(self, dataset, seed, device, params):
        """Class that creates and trains a KAN model based on an input dataset.

        Args:
            dataset (dict): a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output) and feature/output labels.
            seed (int): value to set random state for reproducible results
            device (str): cpu or gpu
            params (dict): a dictionary containing necessary parameters from hyperparameter tuning (depth, grid, k, steps, lamb, lamb_entropy, lr_1, and lr_2).
        """
        self.dataset = dataset
        self.seed = seed
        self.device = device
        self.depth = int(params["depth"])
        self.grid = int(params["grid"])
        self.k = int(params["k"])
        self.steps = int(params["steps"])
        self.lamb = params["lamb"]
        self.lamb_entropy = params["lamb_entropy"]
        self.lr_1 = params["lr_1"]
        self.lr_2 = params["lr_2"]
        self.hidden_nodes_per_layer = self.dataset["train_input"].shape[1]
        # depth is the number of layers, we have to create a list for pykan to
        # generate the kan with these two dimensions
        self.hidden_nodes = [self.hidden_nodes_per_layer for i in range(self.depth)]


    def get_model(self):
        """Uses input dataset to train and return a KAN model.

        Returns:
            pykan KAN model object: model trained on dataset provided to class
        """
        width = [self.dataset['train_input'].shape[1]] + self.hidden_nodes + [self.dataset['train_output'].shape[1]]
        model = KAN(width=width, grid=self.grid, k=self.k, seed=self.seed, device=self.device)
        data = {
            'train_input':self.dataset['train_input'],
            'train_label':self.dataset['train_output'],
            'test_input':self.dataset['test_input'],
            'test_label':self.dataset['test_output']
        }
        model.fit(data, opt='LBFGS', steps=self.steps, lamb=self.lamb, lamb_entropy=self.lamb_entropy, lr=self.lr_1)
        print("Model trained.")
        model = model.prune()
        model.fit(data, opt='LBFGS', steps=self.steps, lamb=self.lamb, lamb_entropy=self.lamb_entropy, lr=self.lr_2, update_grid=False)
        print("Model pruned and re-trained.")
        return model

    def get_metrics(self, model, save_as, p=4):
        """Gets a variety of metrics for each output of the given KAN model evaluated against the test set in dataset.

        Args:
            model (pykan model object): KAN model generated for dataset
            save_as (str): name of csv file to save metrics to
            p (float): the precision, number of decimal places to round metrics 

        Returns:
            pandas DataFrame: dataframe containing metrics for each output
        """
        # get predictions and unscale data
        scaler = self.dataset['y_scaler']
        X_test = self.dataset['test_input'] # still scaled
        Y_test = self.dataset['test_output'] # still scaled
        Y_pred = model(X_test)
        y_test = scaler.inverse_transform(Y_test.cpu().detach().numpy())
        y_pred = scaler.inverse_transform(Y_pred.cpu().detach().numpy())
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
            metrics['MAE'].append(round(mean_absolute_error(yi_test, yi_pred), p))
            metrics['MAPE'].append(round(mape(yi_test, yi_pred), p))
            metrics['MSE'].append(round(mean_squared_error(yi_test, yi_pred), p))
            metrics['RMSE'].append(round(np.sqrt(mean_squared_error(yi_test, yi_pred)), p))
            metrics['RMSPE'].append(round(rmspe(yi_test, yi_pred), p))
            metrics['R2'].append(round(r2_score(yi_test, yi_pred),p))
        metrics_df = pd.DataFrame.from_dict(metrics)
        # check to see fi there 
        if not os.path.exists('results'):
            os.makedirs('results')
        metrics_df.to_csv(f'results/{save_as}.csv', index=False)
        return metrics_df

    def get_schematic():
        return 

    def get_equation(self, model, save_as, lib=None, metrics=False):
        # remove lib after testing (here now to make things faster)
        lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','tan','abs']
        model.auto_symbolic(lib=lib)
        if not os.path.exists('equations'):
            os.makedirs('equations')
        f = open(f"equations/{save_as}_equation.txt", "w")
        for i, output in enumerate(self.dataset['output_labels']):
            formula = model.symbolic_formula()[0][i]
            clean_formula = ex_round(formula, 4)
            f.write(f"{output} = {clean_formula}")
            f.write("\n")
        f.close()
        # generate and save the metrics here!
        if metrics:
            p = 4
            scaler = self.dataset['y_scaler']
            X_test = self.dataset['test_input'] # still scaled
            Y_test = self.dataset['test_output'] # still scaled
            num_vars = len(self.dataset['feature_labels'])
            y_test = scaler.inverse_transform(Y_test.detach().numpy())
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
                expression = ex_round(model.symbolic_formula()[0][i], 4)
                yi_pred = y_pred_sym(expression, num_vars, X_test, scaler)
                metrics['MAE'].append(round(mean_absolute_error(yi_test, yi_pred), p))
                metrics['MAPE'].append(round(mape(yi_test, yi_pred), p))
                metrics['MSE'].append(round(mean_squared_error(yi_test, yi_pred), p))
                metrics['RMSE'].append(round(np.sqrt(mean_squared_error(yi_test, yi_pred)), p))
                metrics['RMSPE'].append(round(rmspe(yi_test, yi_pred), p))
                metrics['R2'].append(round(r2_score(yi_test, yi_pred),p))
            metrics_df = pd.DataFrame.from_dict(metrics)
            # check to see fi there 
            if not os.path.exists('results'):
                os.makedirs('results')
            metrics_df.to_csv(f'results/{save_as}_symetrics.csv', index=False)
        return 

    def get_importances(self, model, save_as):
        """Uses pykan built-in feature importance functionality to rank features from a given model and plot their importances. 

        Args:
            model (pykan model object): KAN model created from dataset
            save_as (str): string to save feature importance plot as

        Returns:
            pytorch tensor: feature importances in order of original dataset
        """
        importances = model.feature_score
        importances = importances.detach().numpy()
        fig = plot_feature_importances(importances, self.dataset['feature_labels'])
        if not os.path.exists('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures/{save_as}.png', dpi=300)
        return importances

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_name = 'MITR_tpe_250211'
    dataset  = get_mitr(cuda=True)
    params = {'depth': 2, 'grid': 4, 'k': 4, 'lamb': 0.00013821835586671683, 'lamb_entropy': 4.21645832233589, 'lr_1': 1.75, 'lr_2': 2, 'steps': 125}
    test_kan = NKAN(dataset=dataset, seed=42, device=device, params=params)
    model = test_kan.get_model()
    #equation = test_kan.get_equation(model, test_name, metrics=True)
    metrics = test_kan.get_metrics(model, test_name)
    #importances = test_kan.get_importances(model, test_name)