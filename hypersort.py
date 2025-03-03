import pandas as pd
import numpy as np

def sort_params(params_file, r2_file, short=True):
    with open(params_file) as f1:
        params = []
        for line in f1:
            params.append(eval(line))
    df = pd.DataFrame.from_dict(params)
    with open(r2_file) as f2:
        r2_list_spline = []
        r2_list_sym = []
        for line in f2:
            r2s = line.split(',')
            r2_list_spline.append(np.round(float(r2s[0].removeprefix("AVG R2 SCORE:").lstrip().strip()), 5))
            r2_list_sym.append(np.round(float(r2s[1].lstrip().removeprefix("SYMBOLIC:").strip()), 5))
    df['SPLINE AVG R2'] = r2_list_spline
    df['SYMBOLIC AVG R2'] = r2_list_sym
    df.sort_values(by='SYMBOLIC AVG R2', ascending=False, inplace=True)
    reg_dict = {
        "edge_forward_spline_n":"EFSN",
        "edge_forward_sum":"EFS",
        "edge_forward_spline_u":"EFSU",
        "edge_backward":"EB",
        "node_backward":"NB"
    } 
    for label in reg_dict.keys():
        df.replace(label, reg_dict[label], inplace=True)
    df['lr_1'] = df['lr_1'].apply(lambda x: f'{x:.2f}'.rstrip('0').rstrip('.'))
    df['lr_2'] = df['lr_2'].apply(lambda x: f'{x:.2f}'.rstrip('0').rstrip('.'))
    if short:
        df = df[0:10]
        to_latex = df.to_latex(float_format="%.5f", formatters={"lamb": lambda x: f"{x:.3e}"}, longtable=True)
        print(to_latex)
    else:
        print(df.to_latex(float_format="%.5f", formatters={"lamb": lambda x: f"{x:.3e}"}, longtable=True))
    return df

def print_space(space):
    df = pd.DataFrame.from_dict(space)
    print(df.to_latex(float_format="%.5e"))
    return df


if __name__=="__main__":
    params_file = "hyperparameters/XS_2025-03-02/XS_2025-03-02_params.txt"
    r2_file = "hyperparameters/XS_2025-03-02/XS_2025-03-02_R2.txt"
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

