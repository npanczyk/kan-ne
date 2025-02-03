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
from hypertuning.py import objective, tune

torch.set_default_dtype(torch.float64)


def obj(params):
    dataset = get_mitr()  # UPDATE THIS FOR EACH DATASET
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"hyperparameters/{run_name}_params.txt", "a") as results:
        results.write(f"{params}\n")
    depth = int(params["depth"])
    grid = int(params["grid"])
    k = int(params["k"])
    steps = int(params["steps"])
    lamb = params["lamb"]
    lamb_entropy = params["lamb_entropy"]
    return objective(
        depth,
        grid,
        k,
        steps,
        lamb,
        lamb_entropy,
        dataset=dataset,
        seed=seed,
        device=device,
    )


if __name__ == "__main__":
    run_name = "MITR_250203"
    if os.path.exists(f"hyperparameters/{run_name}_params.txt"):
        os.remove(f"hyperparameters/{run_name}_params.txt")
        os.remove(f"hyperparameters/{run_name}_R2.txt")
        os.remove(f"hyperparameters/{run_name}_pruned.txt")
    space = {
        "depth": hp.quniform("depth", 1, 4, 1),
        "grid": hp.quniform("grid", 1, 10, 1),
        "k": hp.choice("k", [1, 2, 3, 4, 5]),
        "steps": hp.quniform("steps", 10, 200, 1),
        "lamb": hp.uniform("lamb", 0, 1),
        "lamb_entropy": hp.uniform("lamb_entropy", 0, 10),
    }
    best, trials = tune(obj, space=space, max_evals=100)
    with open(f"hyperparameters/{run_name}_results.txt", "w") as results:
        results.write(str(best))
