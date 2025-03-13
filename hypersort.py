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
    return df

def print_space(space):
    df = pd.DataFrame.from_dict(space)
    print(df.to_latex(float_format="%.5e"))
    return df

def get_top(datasets_dict):
    top = []
    for dataset, path in datasets_dict.items():
        print(path)
        params_file = path+"/"+str(path.split('/')[-1])+'_params.txt'
        r2_file = path+"/"+str(path.split('/')[-1])+'_R2.txt'
        df = sort_params(params_file, r2_file, short=False)
        df['Dataset']= [dataset.upper() for i in range(df.shape[0])]
        df.dropna(inplace=True)
        print(df)
        top.append(df.iloc[0])
    top_df = pd.DataFrame(top)
    top_df.set_index('Dataset', inplace=True)
    to_latex = top_df.to_latex(float_format="%.5f", formatters={"lamb": lambda x: f"{x:.3e}"}, longtable=True)
    print(to_latex)
    return top_df


if __name__=="__main__":
    datasets_dict = {
        'fp': 'best_hyperparams/symbolic/FP_250301',
        'bwr': 'best_hyperparams/symbolic/BWR_2025-03-03',
        'heat': 'best_hyperparams/symbolic/HEAT_2025-03-02',
        'htgr': 'best_hyperparams/symbolic/HTGR_2025-03-02',
        'mitr_a': 'best_hyperparams/symbolic/MITR_A_2025-03-03',
        'mitr_b': 'best_hyperparams/symbolic/MITR_B_2025-03-02',
        'mitr_c': 'best_hyperparams/symbolic/MITR_C_2025-03-02',
        'mitr': 'best_hyperparams/symbolic/MITR_FULL_2025-03-02',
        'chf': 'best_hyperparams/symbolic/CHF_250301',
        'rea': 'best_hyperparams/symbolic/REA_2025-03-02',
        'xs': 'best_hyperparams/symbolic/XS_2025-03-02'
    }
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
    get_top(datasets_dict)
    #params_df = sort_params(params_file, r2_file)
    #space_df = print_space(space)

