import numpy as np
import pandas as pd
from sympy import symbols, sympify, lambdify
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
torch.set_default_dtype(torch.float64)
import os

def rmspe(ytest, ypred):
    """Generates root mean square percentage error.

    Args:
        ytest (np array): true values
        ypred (np array): predicted values

    Returns:
        np array: array of rmspe scores
    """
    inside = np.square((ytest - ypred)/ytest)
    return np.sqrt(np.mean(inside))*100

def mape(ytest, ypred):
    """Generate mean absolute percentage error.

    Args:
        ytest (np array): true values
        ypred (np array): predicted values

    Returns:
        np array: array of MAPE scores
    """
    return np.mean(np.abs((ytest - ypred)/ytest))*100


def get_variable_map(feature_labels):
    # variable_map = {
    #     'sqrt':'\\sqrt', 
    #     'exp':'\\exp', 
    #     'log':'\\log', 
    #     'abs': '\\abs',
    #     'asin':'\\arcsin', 
    #     'acos':'\\arccos', 
    #     'atan':'\\arctan', 
    #     'atanh':'\\arctanh', 
    #     'sin':'\\sin',
    #     'cos':'\\cos', 
    #     'tan':'\\tan', 
    #     'tanh':'\\tanh', 
    #     '**':'^',
    # }
    variable_map = {}
    n_features = len(feature_labels)
    for j in range(n_features):
        # go in descending order here to make sure x_11 gets subbed before x_1
        variable_map[f'x_{{{n_features - j}}}'] = f'\\mathit{{{feature_labels[n_features-j-1]}}}'
    return variable_map



def y_pred_sym(expressions, num_vars, X_test, scaler, device):
    """Gets predictions based on X_test and a symbolic expression.

    Args:
        expressions (list): List of symbolic expressions generated by autosymbolic().
        num_vars (int): The number of features in dataset.
        X_test (numpy array or tensor): Test dataset.
        scaler (sklearn object): Y scaler for transformed data.
        device (str): Device to check for (e.g., "cpu" or "cuda").

    Returns:
        numpy array: An unscaled prediction for each X test value.
    """
    Y_preds = np.zeros((X_test.shape[0], len(expressions)))
    for i, expression in enumerate(expressions):
        # set up symbolic variables to match autosymbolic()
        variables = symbols(f'x_1:{num_vars + 1}')
        
        # convert expression using lambdify so vector ops can be used
        numerical_function = lambdify(variables, expression, 'numpy')
        
        # make sure X_test is pushed to cpu and converted to numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.detach().cpu().numpy() 
        Y_pred_sym = numerical_function(*X_test.T)
        Y_preds[:, i] = Y_pred_sym.flatten()

    y_pred_sym = scaler.inverse_transform(Y_preds)
    return y_pred_sym

def metrics(output_labels, y_test, y_pred, p):
    """Generates a metrics dataframe when given a test set and a predicted set.

    Args:
        output_labels (list): The model's output names
        y_test (NumPy Array): test set, an array of unscaled values for each output
        y_pred (NumPy Arra): an array of unscaled values, as predicted by the model, for each output
        p (int): precision (number of decimal places to round metrics to)

    Returns:
        pandas DataFrame: a DataFrame containing MAE, MAPE, MSE, RMSE, RMSPE, and R2 values for the model's predictions of the test set
    """
    metrics = {
            'OUTPUT':output_labels,
            'MAE':[],
            'MAPE':[],
            'MSE':[],
            'RMSE':[],
            'RMSPE':[],
            'R2':[]
        }
    for i in range(len(output_labels)):
        # get metrics for each output
        yi_test = y_test[:,i]
        yi_pred = y_pred[:,i]
        metrics['MAE'].append(round(mean_absolute_error(yi_test, yi_pred), p))
        metrics['MAPE'].append(round(mape(yi_test, yi_pred), p))
        metrics['MSE'].append(round(mean_squared_error(yi_test, yi_pred), p))
        metrics['RMSE'].append(round(np.sqrt(mean_squared_error(yi_test, yi_pred)), p))
        metrics['RMSPE'].append(round(rmspe(yi_test, yi_pred), p))
        metrics['R2'].append(round(r2_score(yi_test, yi_pred),p))
    return pd.DataFrame.from_dict(metrics)

def print_shap(path, save_as, type):
    df = pd.read_pickle(path)
    if not os.path.exists('shap-values/'):
        os.makedirs('shap-values')
    df.to_csv(f'shap-values/{save_as}_{type}.csv')
    to_latex = df.to_latex(float_format="%.5f", index=True, longtable=True, caption=f'Absolute value of mean SHAP values for {type.upper()} model of {save_as.upper()} dataset.')
    print(to_latex)
    return 
