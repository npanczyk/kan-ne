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
        inputs = {variable: value for variable, value in zip(variables, row)}
        Y_pred_sym.append(float(symbolic_expr.subs(inputs).evalf()))
    Y_pred_sym = np.array(Y_pred_sym).reshape(-1, 1)
    y_pred_sym = scaler.inverse_transform(Y_pred_sym)
    return y_pred_sym

def symbolic_FI(expr, X_test, Y_test, input_names, output_names, save_as, range=300):
    num_vars = len(input_names)
    s_expr = sympify(expr)
    # equation as function with arg "inputs"
    variables = symbols('x_1:%d' % (num_vars + 1))
    print(variables)
    compute_Y = lambdify([variables], s_expr)
    # wrapper function, called model
    model = lambda inputs: np.array([compute_Y(variables) for variables in inputs])
    explainer = shap.KernelExplainer(model, X_test[0:range])
    shap_values = explainer.shap_values(X_test[0:])
    shap_mean = pd.DataFrame(np.abs(shap_values).mean(axis=0),columns=output_names,index=input_names)
    fig, ax = plt.subplots(figsize=(8,8))
    df_mean_sorted = shap_mean.iloc[:, 0].sort_values(ascending=False)
    print(df_mean_sorted)
    ax.bar(df_mean_sorted.index, df_mean_sorted.iloc[:], capsize=4, width=0.3, color="lightseagreen")
    ax.set_ylabel("Mean of |SHAP Values|")
    plt.xticks(rotation=90, ha="right") ## FIX THIS!
    plt.tight_layout()
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/{save_as}_symbolic_FI.png', dpi=300)
    return fig, ax

if __name__=="__main__":
    from preprocessing import get_chf
    import shap
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    dataset = get_chf()
    X_test = dataset['test_input'].detach().numpy()
    Y_test = dataset['test_output'].detach().numpy()
    input_names = dataset['feature_labels']
    output_names = dataset['output_labels']
    save_as = 'TEST'
    expr = '0.117*x_1 + 0.0639*x_2 + 0.1276*x_4 - 0.0387*x_6 - 0.1296*exp(0.1521*x_1 - 0.2177*x_2) - 0.0318*exp(-0.2068*x_2 - 0.0277*sin(3.0794*x_3 - 5.3524)) - 0.0454*exp(-0.1174*x_1 + 0.0918*x_2 - 0.0082*x_3 - 0.0084*x_5) + 0.0031*sin(3.157*x_3 + 4.0237) + 0.0195*sin(3.2175*x_3 - 8.5904) + 0.0059*sin(6.5958*x_5 - 6.9819) + 0.3798'
    symbolic_FI(expr, X_test, Y_test, input_names, output_names, save_as, range=10)