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
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    # chf_tuner = Tuner(
    #                 dataset = get_chf(cuda=True), 
    #                 run_name = "CHF_tpe_250210", 
    #                 space = set_space(), 
    #                 max_evals = 150, 
    #                 seed = 42, 
    #                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # rea_tuner = Tuner(
    #                 dataset = get_rea(cuda=True), 
    #                 run_name = "REA_tpe_250210", 
    #                 space = set_space(), 
    #                 max_evals = 150, 
    #                 seed = 42, 
    #                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))      
    # bwr_tuner = Tuner(
    #                 dataset = get_bwr(cuda=True), 
    #                 run_name = "BWR_tpe_250209", 
    #                 space = set_space(), 
    #                 max_evals = 150, 
    #                 seed = 42, 
    #                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    htgr_tuner = Tuner(
                    dataset = get_htgr(cuda=True), 
                    run_name = "HTGR_tpe_250212", 
                    space = set_space(), 
                    max_evals = 150, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # try:
    #     tune_case(chf_tuner)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! CHF stopped prematurely. Error: {e}")
    # try:
    #     tune_case(rea_tuner)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! REA stopped prematurely. Error: {e}")
    # try:
    #     tune_case(bwr_tuner)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! BWR stopped prematurely. Error: {e}")
    try:
        tune_case(htgr_tuner)
    except Exception as e:
        print(f"TUNING INTERRUPTED! HTGR stopped prematurely. Error: {e}")        
