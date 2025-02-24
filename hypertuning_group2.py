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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    mitr_C_tuner = Tuner(
                    dataset = get_mitr(cuda=True, region='C'), 
                    run_name = "MITR_C_250224", 
                    space = set_space(), 
                    max_evals = 150, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mitr_tuner = Tuner(
                    dataset = get_mitr(cuda=True), 
                    run_name = "MITR_full_250224", 
                    space = set_space(), 
                    max_evals = 150, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        tune_case(mitr_C_tuner)
    except Exception as e:
        print(f"TUNING INTERRUPTED! MITR C stopped prematurely. Error: {e}")     
    try:
        tune_case(mitr_tuner)
    except Exception as e:
        print(f"TUNING INTERRUPTED! MITR stopped prematurely. Error: {e}")
   
