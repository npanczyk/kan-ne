import numpy as np
from sympy import symbols

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

def y_pred_sym(expression, num_vars, X_test, scaler):
    """Gets predictions based on X_test and a symbolic expression

    Args:
        symbolic_expr (str): A symbolic expression for the output of interest. Variables must be of the form: x_1, x_2,..., x_n for n features. 
        num_vars (int): The number of features in the dataset of interest
        X_test (pytorch tensor): _description_
        scaler (sklearn object): Y scaler for transformed data

    Returns:
        list: An unscaled prediction for each X test value
    """
    symbolic_expr = sympify(expression)
    variables = symbols('x_1:%d' % (num_vars + 1))
    Y_pred_sym = [] # scaled predictions based on symbolic expression
    X_test = X_test.detach().numpy()
    for row in X_test:
        inputs = {variable, value for variable, value in zip(variables, row)}
        Y_pred_sym.append(float(symbolic_expr.subs(inputs).evalf()))
    Y_pred_sym = np.array(Y_pred_sym).reshape(-1, 1)
    y_pred_sym = scaler.inverse_transform(Y_pred_sym)
    return y_pred_sym

if __name__=="__main__":
    symbolic_metrics('x_1 + x_2', 2)