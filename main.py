import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from kan import *
from kan.utils import ex_round
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
torch.set_default_dtype(torch.float64)
from sympy import symbols, sympify, latex
from preprocessing import *
from accessories import *
from plotting import plot_feature_importances, plot_overfitting
import shutil
from datetime import datetime
import time


class NKAN():
    def __init__(self, dataset, seed, device, params):
        """Class that creates and trains a KAN model based on an input dataset.

        Args:
            dataset (dict): a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output) and feature/output labels.
            seed (int): value to set random state for reproducible results
            device (str): cpu or gpu
            params (dict): a dictionary containing necessary parameters from hyperparameter tuning (depth, grid, k, steps, lamb, lamb_entropy, lr_1, and lr_2).
        """
        self.dataset = dataset
        self.hidden_nodes_per_layer = self.dataset["train_input"].shape[1]
        # depth is the number of layers, we have to create a list for pykan to
        # generate the kan with these two dimensions
        self.depth = int(params["depth"])
        self.hidden_nodes = [self.hidden_nodes_per_layer for i in range(self.depth)]
        self.width = [self.dataset['train_input'].shape[1]] + self.hidden_nodes + [self.dataset['train_output'].shape[1]]
        self.k = int(params["k"])
        self.grid = int(params["grid"])
        # inherit initialization from KAN class
        # super().__init__(width, grid, k, seed, device)
        self.steps = int(params["steps"])
        self.lamb = params["lamb"]
        self.lamb_entropy = params["lamb_entropy"]
        self.lr_1 = params["lr_1"]
        self.lr_2 = params["lr_2"]
        self.reg_metric = params["reg_metric"]
        self.device = device
        self.seed = seed
        self.data = {
            'train_input':self.dataset['train_input'],
            'train_label':self.dataset['train_output'],
            'test_input':self.dataset['test_input'],
            'test_label':self.dataset['test_output']
        }
        self.model = KAN(width=self.width, grid=self.grid, k=self.k, seed=self.seed, device=self.device)
        
    


    def get_model(self, save=False, save_as=None):
        """Uses input dataset to train and return a KAN model.

        Args:
            save (bool, optional): determines whether or not to save the model object to be reloaded later. Defaults to False.
            save_as (str, optional): name of the saved model object in /models. Defaults to None.

        Returns:
            pykan KAN model object: model trained on dataset provided to class
        """
        model = self.model
        model.fit(self.data, opt='LBFGS', steps=self.steps, lamb=self.lamb, lamb_entropy=self.lamb_entropy, lr=self.lr_1, reg_metric=self.reg_metric)
        print("Model trained.")
        model = model.prune()
        model.fit(self.data, opt='LBFGS', steps=self.steps, lamb=self.lamb, lamb_entropy=self.lamb_entropy, lr=self.lr_2, reg_metric=self.reg_metric, update_grid=False)
        print("Model pruned and re-trained.")
        if not os.path.exists('models'):
            os.makedirs('models')
        if save:
            model.saveckpt(f'models/{save_as}')
        return model
    
    def refine(self, model, grids, save_as=f'refine_{str(datetime.now())}'):
        """Makes a plot to check overfitting while training a model from scratch. 

        Args:
            grids (list): List of integers that represent grid refinements to implement.
            plot (bool, optional): _description_. Defaults to True.
            save_as (_type_, optional): _description_. Defaults to str(datetime.now()).

        Returns:
            _type_: _description_
        """
        width = [self.dataset['train_input'].shape[1]] + self.hidden_nodes + [self.dataset['train_output'].shape[1]]
        layer_params = [width[i]*width[i+1] for i in range(len(width) - 1)]
        n_params = np.array(grids) * np.sum(layer_params)
        train_rmse = []
        test_rmse = []
        cont_train_rmse = []
        cont_test_rmse = []

        for i in range(len(grids)):
            model = model.refine(grids[i])
            results = model.fit(self.data, opt='LBFGS', steps=self.steps, 
                                lamb=self.lamb, lamb_entropy=self.lamb_entropy, lr=self.lr_1, stop_grid_update_step=20)
            train_rmse.append(results['train_loss'][-1].item())
            test_rmse.append(results['test_loss'][-1].item())
            cont_train_rmse += results['train_loss']
            cont_test_rmse += results['test_loss']

        fig = plot_overfitting(n_params, train_rmse, test_rmse, cont_train_rmse, cont_test_rmse, save_as=save_as)

        return fig

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
        if str(self.device) == "cuda":
            y_test = scaler.inverse_transform(Y_test.cpu().detach().numpy())  # unscaled
            y_pred = scaler.inverse_transform(Y_pred.cpu().detach().numpy())  # unscaled
        else:
            y_test = scaler.inverse_transform(Y_test.detach().numpy())  # unscaled
            y_pred = scaler.inverse_transform(Y_pred.detach().numpy())  # unscaled
        metrics_df = metrics(self.dataset['output_labels'], y_test, y_pred, p)
        # check to see if results folder exists 
        if not os.path.exists('results'):
            os.makedirs('results')
        metrics_df.to_csv(f'results/{save_as}.csv', index=False)
        return metrics_df

    def get_schematic():
        return 

    def get_equation(self, model, save_as, simple=0.8, lib=None):
        """Converts splines into symbolic functions for the model object. Saves symbolic functions for each output to a text file under /equations. If metrics, calculates and saves symbolic metrics to a file under /results.

        Args:
            model (kan model object): kan model either directly from get_model() or from NKAN.loadpkct() (inherited from KAN). 
            save_as (str): _description_
            simple (float between 0 and 1, optional): 1 weights simplicity of symbolic expression completely over accuracy (R2), 0 does the opposite . Defaults to 0.8.
            lib (list, optional): Library of symbolic functions written as strings. Defaults to None.
            metrics (bool, optional): If true, calculates and saves symbolic metrics under /results. Defaults to False.
        """
        n_outputs = len(self.dataset['output_labels'])
        start = time.time()
        # this permanently converts activation functions
        model.auto_symbolic(lib=lib, weight_simple=simple)
        # get the conversion time
        end = time.time()
        # this whole chunk is just writing the equations to a file
        if not os.path.exists('equations'):
            os.makedirs('equations')
        sym_file = open(f"equations/{save_as}.txt", "w")
        tex_file = open(f"equations/{save_as}_latex.txt", "w")
        variable_map = get_variable_map(self.dataset['feature_labels'])
        print(variable_map)
        for i, output in enumerate(self.dataset['output_labels']):
            formula = model.symbolic_formula()[0][i]
            # round all the coefficients
            latex_formula = latex(ex_round(formula, 4))
            sympy_formula = str(ex_round(formula, 4))
            for char, replacement in variable_map.items():
                latex_formula = latex_formula.replace(char,replacement)
            sym_file.write(sympy_formula)
            sym_file.write("\n")
            output_as_latex = '\\text{'+str(output)+'}'
            tex_file.write(output +' = '+ latex_formula)
            tex_file.write("\n")
        sym_file.close()
        tex_file.close()
        # generate and save the metrics here!
        scaler = self.dataset['y_scaler']
        X_test = self.dataset['test_input'] # still scaled
        Y_test = self.dataset['test_output'] # still scaled
        num_vars = len(self.dataset['feature_labels'])
        n_outputs = len(self.dataset['output_labels'])
        if str(self.device) == "cuda":
            y_test = scaler.inverse_transform(Y_test.cpu().detach().numpy())  # unscaled
        else:
            y_test = scaler.inverse_transform(Y_test.detach().numpy())  # unscaled
        expressions = [ex_round(model.symbolic_formula()[0][i], 4) for i in range(n_outputs)]
        y_pred = y_pred_sym(expressions, num_vars, X_test, scaler, str(self.device))
        metrics_df = metrics(self.dataset['output_labels'], y_test, y_pred, p=4)
        metrics_df['CONVERSION_TIME'] = end - start
        # save symbolic metrics to results 
        if not os.path.exists('results'):
            os.makedirs('results')
        metrics_df.to_csv(f'results/{save_as}_symetrics.csv', index=False)
        return expressions

    def get_importances(self, model, save_as):
        """Uses pykan built-in feature importance functionality to rank features from a given model and plot their importances. 

        Args:
            model (pykan model object): KAN model created from dataset
            save_as (str): string to save feature importance plot as

        Returns:
            pytorch tensor: feature importances in order of original dataset
        """
        importances = model.feature_score
        if str(self.device) == "cuda":
            importances = importances.cpu().detach().numpy()
        else:
            importances = importances.detach().numpy()
        fig = plot_feature_importances(importances, self.dataset['feature_labels'])
        if not os.path.exists('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures/{save_as}.png', dpi=300)
        return importances
