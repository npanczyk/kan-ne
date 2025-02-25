import numpy as np
from sympy import symbols, sympify, lambdify

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

def y_pred_sym(expressions, num_vars, X_test, scaler, device):
    """Gets predictions based on X_test and a symbolic expression

    Args:
        symbolic_expr (str): A symbolic expression for the output of interest. Variables must be of the form: x_1, x_2,..., x_n for n features. 
        num_vars (int): The number of features in the dataset of interest
        X_test (pytorch tensor): _description_
        scaler (sklearn object): Y scaler for transformed data

    Returns:
        list: An unscaled prediction for each X test value
    """
    Y_preds = np.zeros((X_test.shape[0], len(expressions)))
    for i, expression in enumerate(expressions):
        symbolic_expr = sympify(expression)
        variables = symbols('x_1:%d' % (num_vars + 1))
        Y_pred_sym = [] # scaled predictions based on symbolic expression
        # NOT SURE WHY THIS WORKED BUT X_test TRANSFERS ITSELF to CPU?
        # if device == "cuda":
        #     print(type(X_test))
        #     print(X_test)
        #     X_test = X_test.cpu().detach().numpy()
        # else:
        #     X_test = X_test.detach().numpy()
        for row in X_test:
            inputs = {variable: value for variable, value in zip(variables, row)}
            Y_pred_sym.append(float(symbolic_expr.subs(inputs).evalf()))
        Y_pred_sym = np.array(Y_pred_sym).reshape(-1, 1)
        Y_preds[:, i] = Y_pred_sym.flatten()
    y_pred_sym = scaler.inverse_transform(Y_preds)
    return y_pred_sym

def get_variable_map():
    map = {
        'sqrt':'\\sqrt', 
        'exp':'\\exp', 
        'log':'\\log', 
        'abs': '\\abs', 
        'sin':'\\sin',
        'cos':'\\cos', 
        'tan':'\\tan', 
        'tanh':'\\tanh', 
        'arcsin':'\\arcsin', 
        'arccos':'\\arccos', 
        'arctan':'\\arctan', 
        'arctanh':'\\arctanh',
        '**':'^'
    }
    return map