# MITR TUNING
import numpy as np
import torch
from hyperopt import tpe, hp, fmin, Trials
from preprocessing import *
from functools import partial
from sklearn.metrics import r2_score
from kan import *
import shutil
import os
from hypertuning.py import objective, tune, save_setup

torch.set_default_dtype(torch.float64)


def obj(params):
    dataset = get_mitr()  # UPDATE THIS FOR EACH DATASET
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"hyperparameters/{run_name}/{run_name}_params.txt", "a") as results:
        results.write(f"{params}\n")
    depth = int(params["depth"])
    grid = int(params["grid"])
    k = int(params["k"])
    steps = int(params["steps"])
    lamb = params["lamb"]
    lamb_entropy = params["lamb_entropy"]
    lr_1 = params["lr_1"]
    lr_2 = params["lr_2"]
    return objective(
        depth,
        grid,
        k,
        steps,
        lamb,
        lamb_entropy,
        lr_1,
        lr_2,
        dataset=dataset,
        seed=seed,
        device=device,
    )


if __name__ == "__main__":
    run_name = "MITR_250203"
    save_setup()
    space = {
        "depth": hp.quniform("depth", 1, 4, 1),
        "grid": hp.quniform("grid", 1, 10, 1),
        "k": hp.choice("k", [1, 2, 3, 4, 5]),
        "steps": hp.quniform("steps", 10, 20, 1),
        "lamb": hp.uniform("lamb", 0, 1),
        "lamb_entropy": hp.uniform("lamb_entropy", 0, 10),
        "lr_1": hp.choice("lr_1", [0.0001, 0.001, 0.01, 0.1, 1]),
        "lr_2": hp.choice("lr_2", [0.0001, 0.001, 0.01, 0.1, 1]),
    }
    best, trials = tune(obj, space=space, max_evals=100)
    with open(f"hyperparameters/{run_name}/{run_name}_results.txt", "w") as results:
        results.write(str(best))
