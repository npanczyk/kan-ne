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
    htgr_tuner_1 = Tuner(
                    dataset = get_htgr(cuda=True, quadrant=1), 
                    run_name = "HTGR_Q1_250212", 
                    space = set_space(), 
                    max_evals = 150, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    htgr_tuner_2 = Tuner(
                dataset = get_htgr(cuda=True, quadrant=2), 
                run_name = "HTGR_Q2_250212", 
                space = set_space(), 
                max_evals = 150, 
                seed = 42, 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    htgr_tuner_3 = Tuner(
                    dataset = get_htgr(cuda=True, quadrant=3), 
                    run_name = "HTGR_Q3_250212", 
                    space = set_space(), 
                    max_evals = 150, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    htgr_tuner_4 = Tuner(
                    dataset = get_htgr(cuda=True, quadrant=4), 
                    run_name = "HTGR_Q4_250212", 
                    space = set_space(), 
                    max_evals = 150, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    tune_case(htgr_tuner_1)

    # try:
    #     tune_case(htgr_tuner_1)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! HTGR 1 stopped prematurely. Error: {e}")     
    # try:
    #     tune_case(htgr_tuner_2)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! HTGR 2 stopped prematurely. Error: {e}")
    # try:
    #     tune_case(htgr_tuner_3)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! HTGR 3 stopped prematurely. Error: {e}")
    # try:
    #     tune_case(htgr_tuner_4)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! HTGR 4 stopped prematurely. Error: {e}")   
