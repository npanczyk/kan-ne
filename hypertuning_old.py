# this is where we will do all the hyperparameter tuning
# see: https://kindxiaoming.github.io/pykan/API_demo/API_6_training_hyperparameter.html for tuning advice

# PSEUDO-CODE
# 1. set nodes per layer to # of features
# 2. start with the 2 layers (1 hidden)
# 3. train kan and get metric
# 4. repeat with up to 4 layers (3 hidden)
# 5. choose best # of layers
# 6. double nodes per layer
# 7. train and get metric
# 8. repeat until you have quadrupled the nodes per layer
# 9. choose optimum # of nodes per layer
# 10. return hyperparams as list of [input nodes, layer1 nodes, layer2 nodes, ..., output nodes]
import numpy as np
import torch
from hyperopt import tpe, hp, fmin, Trials
from preprocessing import *
from functools import partial
from sklearn.metrics import r2_score
from kan import *
import shutil
import os

torch.set_default_dtype(torch.float64)


def objective(depth, grid, k, steps, lamb, lamb_entropy, lr_1, lr_2, dataset, seed, device):
    """This function is used as an objective function for hyperopt's fmin().

    Args:
        depth (int): The number of hidden layers that contain equal number of nodes to the size of the feature set.
        grid (int): The number of grid intervals used by pykan to generate the network. Should range from 1-10.
        k (int): The spline order used by pykan. Should range from 1-5.
        steps (int): The number of training steps taken by pykan. Should range (10-200, by 10s)
        lamb (float): Overall penalty strength. Should range 0 to 1.
        lamb_entropy (float): Entropy penalty strength. Should range 0 to 10.
        lr (float): Learning rate.
        dataset (dict): a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output) and feature/output labels.
        seed (int): an integer to set the seed for the run.
        device (): a pytorch device to use cpu/gpu

    Returns:
        float: A negative, averaged R2 score for all trials of all outputs in a dataset as predicted by a KAN network with the hyperparameter combination tested by hyperopt.
    """
    # default the hidden nodes per layer as the same as the number of input features
    # this ensures that we start with a "block" of nodes and let
    # pykan trim them into a "tree"
    hidden_nodes_per_layer = dataset["train_input"].shape[1]
    # depth is the number of layers, we have to create a list for pykan to
    # generate the kan with these two dimensions
    hidden_nodes = [hidden_nodes_per_layer for i in range(depth)]
    width = (
        [dataset["train_input"].shape[1]]
        + hidden_nodes
        + [dataset["train_output"].shape[1]]
    )
    # here we initialize the KAN
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device)
    data = {
        "train_input": dataset["train_input"],
        "train_label": dataset["train_output"],
        "test_input": dataset["test_input"],
        "test_label": dataset["test_output"],
    }
    # now, we fit the KAN using the dataset and some hyperparams
    # we do not change the optimizer as part of this search
    model.fit(data, opt="LBFGS", steps=steps, lamb=lamb, lamb_entropy=lamb_entropy, lr=lr_1)
    try:
        # let pykan prune some extraneous connections
        model = model.prune()
        # fit again, now we have a "tree"
        model.fit(
            data,
            opt="LBFGS",
            steps=steps,
            lamb=lamb,
            lamb_entropy=lamb_entropy,
            lr=lr_2,
            update_grid=False,
        )  # set update grid to False to fix pruning NAN loss error
        with open(f"hyperparameters/{run_name}/{run_name}_pruned.txt", "a") as results:
            results.write("Model pruned and refit.\n")
    except RuntimeError:
        with open(f"hyperparameters/{run_name}/{run_name}_pruned.txt", "a") as results:
            results.write("PRUNING SKIPPED!!!\n")
    finally:
        # get the average r2 score of all of the outputs for hyperparameter tuning
        scaler = dataset["y_scaler"]
        X_test = dataset["test_input"]  # still scaled
        Y_test = dataset["test_output"]  # still scaled
        Y_pred = model(X_test)
        y_test = scaler.inverse_transform(Y_test.detach().numpy())  # unscaled
        y_pred = scaler.inverse_transform(Y_pred.detach().numpy())  # unscaled
        r2s = []
        for i in range(len(dataset["output_labels"])):
            yi_test = y_test[:, i]
            yi_pred = y_pred[:, i]
            r2s.append(r2_score(yi_test, yi_pred))
        # SHOULD WE DO SOMETHING HERE TO HANDLE NEGATIVE SCORES?
        r2s = np.array(r2s)
        avg_r2 = np.mean(r2s)
        with open(f"hyperparameters/{run_name}/{run_name}_R2.txt", "a") as results:
            results.write(f"AVG R2 SCORE: {avg_r2}\n")
        # delete model folder at the end of the run
        shutil.rmtree("model")
        return -1 * avg_r2  # make negative because fmin is a minimizer


def obj(params):
    dataset = get_chf(synthetic=False)  # UPDATE THIS FOR EACH DATASET
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


def tune(obj, space, max_evals, algorithm=None):
    trials = Trials()
    best = fmin(obj, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best, trials

def save_setup(run_name):
    if not os.path.exists(f"hyperparameters/{run_name}"):
        os.makedirs(f"hyperparameters/{run_name}")
    if os.path.exists(f"hyperparameters/{run_name}/{run_name}_params.txt"):
        os.remove(f"hyperparameters/{run_name}/{run_name}_params.txt")
        os.remove(f"hyperparameters/{run_name}/{run_name}_R2.txt")
        os.remove(f"hyperparameters/{run_name}/{run_name}_pruned.txt")
    return 


if __name__ == "__main__":
    run_name = "chf"
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
    best, trials = tune(obj, space=space, max_evals=2)
    with open(f"hyperparameters/{run_name}/{run_name}_results.txt", "w") as results:
        results.write(str(best))
