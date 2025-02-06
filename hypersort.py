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
    return df 

if __name__=="__main__":
    params_file = "hyperparameters/CHF_discrete_250206/CHF_discrete_250206_params.txt"
    r2_file = "hyperparameters/CHF_discrete_250206/CHF_discrete_250206_R2.txt"
    df = sort_params(params_file, r2_file)
    print(df.to_latex(float_format="%.5f"))

