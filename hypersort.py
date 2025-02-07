import pandas as pd
import numpy as np

def sort_params(params_file, r2_file):
    with open(params_file) as f1:
        params = []
        for line in f1:
            params.append(eval(line))
    df = pd.DataFrame.from_dict(params)
    with open(r2_file) as f2:
        r2_list = []
        for line in f2:
            r2_list.append(np.round(float(line.removeprefix("AVG R2 SCORE:").lstrip().strip()), 5))
    df['AVG R2'] = r2_list
    df.sort_values(by='AVG R2', ascending=False, inplace=True)
    print(df.to_latex(float_format="%.5f", formatters={"lamb": lambda x: f"{x:.3e}"}, longtable=True))
    return df 

def print_space(space):
    df = pd.DataFrame.from_dict(space)
    print(df.to_latex(float_format="%.5e"))
    return df


if __name__=="__main__":
    params_file = "hyperparameters/CHF_250206/CHF_250206_params.txt"
    r2_file = "hyperparameters/CHF_250206/CHF_250206_R2.txt"
    space = {
        "depth": ["hp.choice", [1, 2, 3, 4]],
        "grid": ["hp.choice", [4, 5, 6, 7, 8, 9, 10]],
        "k": ["hp.choice", [2, 3, 4, 5, 6, 7, 8]],
        "steps": ["hp.choice", [50, 75, 100, 125, 150, 200, 225, 250]],
        "lamb": ["hp.uniform", [0, 0.001]],
        "lamb_entropy": ["hp.uniform", [0, 5]],
        "lr_1": ["hp.choice", [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]], 
        "lr_2": ["hp.choice", [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]],
    }
    params_df = sort_params(params_file, r2_file)
    #space_df = print_space(space)

