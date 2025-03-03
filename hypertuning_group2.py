import numpy as np
import torch
from hyperopt import tpe, hp, fmin, Trials, rand
from preprocessing import *
from functools import partial
from sklearn.metrics import r2_score
from kan import *
import shutil
import os
from hypertuning import *
torch.set_default_dtype(torch.float64)
import datetime as dt

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    regions = ['A', 'FULL']
    datasets_dict = {
        'heat': get_bwr,
    }
    for region in regions:
        datasets_dict[f'mitr_{region}'] = partial(get_mitr, region=region)
    for model, dataset in datasets_dict.items():
        print(f'MODEL: {model}')    
        tuner = Tuner(
                        dataset = dataset(cuda=True), 
                        run_name = f"{model.upper()}_{str(dt.date.today())}", 
                        space = set_space(), 
                        max_evals = 200, 
                        seed = 42, 
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        symbolic = True)
        try:
            tune_case(tuner)
        except Exception as e:
            print(f"{model.upper()} TUNING INTERRUPTED! Error: {e}")     

   
