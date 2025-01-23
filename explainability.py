from preprocessing import *
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sympy import symbols, sympify, lambdify

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
    # ACTIVATE SHAP-ENV BEFORE RUNNING
    dataset = get_htgr()
    X_test = dataset['test_input'].detach().numpy()
    Y_test = dataset['test_output'].detach().numpy()
    input_names = dataset['feature_labels']
    output_names = dataset['output_labels']
    equation_file = 'equations/htgr_test_equation.txt'
    with open(equation_file) as file:
        for output in output_names:
            save_as = f'TEST_output'
            expr = file.readline().split('=')[1].lstrip()
            print(expr)
    #symbolic_FI(expr, X_test, Y_test, input_names, output_names, save_as, range=10)