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
from preprocessing import *
from accessories import rmspe, mape
from plotting import plot_feature_importances


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

    def get_model(self, test=False, tuning=False):
        """Uses input dataset to train and return a KAN model (without tuning).

        Returns:
            pykan KAN model object: model trained on dataset provided to class
        """
        width = [self.dataset['train_input'].shape[1]] + self.hidden_nodes + [self.dataset['train_output'].shape[1]]
        print(width)
        model = KAN(width=width, grid=self.grid, k=self.k, seed=self.seed, device=self.device)
        # DELETE CONDITIONAL AFTER TESTING
        if test:
            data = {
            'train_input':self.dataset['train_input'][0:50],
            'train_label':self.dataset['train_output'][0:50],
            'test_input':self.dataset['test_input'][0:50],
            'test_label':self.dataset['test_output'][0:50]
        }
        else:
            data = {
                'train_input':self.dataset['train_input'],
                'train_label':self.dataset['train_output'],
                'test_input':self.dataset['test_input'],
                'test_label':self.dataset['test_output']
            }
        model.fit(data, opt='LBFGS', steps=100, lamb=0.001, lamb_entropy=2)
        print("Model trained.")
        model = model.prune()
        model.fit(data, opt='LBFGS', steps=100, lamb=0.001, lamb_entropy=2)
        print("Model pruned and re-trained.")
        # get the average r2 score of all of the outputs for hyperparameter tuning
        if tuning:
            scaler = self.dataset['y_scaler']
            X_test = self.dataset['test_input'] # still scaled
            Y_test = self.dataset['test_output'] # still scaled
            Y_pred = model(X_test)
            y_test = scaler.inverse_transform(Y_test.detach().numpy())
            y_pred = scaler.inverse_transform(Y_pred.detach().numpy())
            r2s = []
            for i in range(len(self.dataset['output_labels'])):
                yi_test = y_test[:,i]
                yi_pred = y_pred[:,i]
                r2s.append(r2_score(yi_test, yi_pred))
            print(r2s)
            avg_r2 = np.mean(r2s)
            return avg_r2
        else:
            return model

    def get_metrics(self, model, save_as):
        """Gets a variety of metrics for each output of the given KAN model evaluated against the test set in dataset.

        Args:
            model (pykan model object): KAN model generated for dataset
            save_as (str): name of csv file to save metrics to

        Returns:
            pandas DataFrame: dataframe containing metrics for each output
        """
        # get predictions and unscale data
        scaler = self.dataset['y_scaler']
        X_test = self.dataset['test_input'] # still scaled
        Y_test = self.dataset['test_output'] # still scaled
        Y_pred = model(X_test)
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

    def get_equation(self, model, save_as, lib=None):
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
    test_name = 'chf_122'
    dataset  = get_chf()
    test_kan = NKAN(dataset=dataset, seed=42, device=device)
    model = test_kan.get_model(test=False)
    #r2 = test_kan.get_model(test=False, tuning=True)
    #equation = test_kan.get_equation(model, test_name)
    metrics = test_kan.get_metrics(model, test_name)
    #importances = test_kan.get_importances(model, test_name)