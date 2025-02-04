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

########################  TUNER CLASS ###########################
class Tuner():
    def __init__(self, dataset, run_name, space, max_evals, seed, device):
        self.dataset = dataset
        self.run_name = run_name
        self.space = space
        self.max_evals = max_evals
        self.seed = seed
        self.device = device
        # make sure our file structure is ready to go
        try:
            os.makedirs(f"hyperparameters/{self.run_name}")
        except FileExistsError:
            files = os.listdir(f"hyperparameters/{self.run_name}")
            for file in files:
                os.remove(f"hyperparameters/{self.run_name}/{file}")

    def objective(self, params):
        """This function is used as an starter objective function for hyperopt's fmin() (not the final objective function).

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
        # write the params for this run to an output file
        with open(f"hyperparameters/{self.run_name}/{self.run_name}_params.txt", "a") as results:
            results.write(f"{params}\n")
        # break up the params dictionary
        depth = int(params["depth"])
        grid = int(params["grid"])
        k = int(params["k"])
        steps = int(params["steps"])
        lamb = params["lamb"]
        lamb_entropy = params["lamb_entropy"]
        lr_1 = params["lr_1"]
        lr_2 = params["lr_2"]
        # default the hidden nodes per layer as the same as the number of input features
        # this ensures that we start with a "block" of nodes and let
        # pykan trim them into a "tree"
        hidden_nodes_per_layer = self.dataset["train_input"].shape[1]
        # depth is the number of layers, we have to create a list for pykan to
        # generate the kan with these two dimensions
        hidden_nodes = [hidden_nodes_per_layer for i in range(depth)]
        width = (
            [self.dataset["train_input"].shape[1]]
            + hidden_nodes
            + [self.dataset["train_output"].shape[1]]
        )
        # here we initialize the KAN
        model = KAN(width=width, grid=grid, k=k, seed=self.seed, device=self.device)
        data = {
            "train_input": self.dataset["train_input"],
            "train_label": self.dataset["train_output"],
            "test_input": self.dataset["test_input"],
            "test_label": self.dataset["test_output"],
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
            ) # change update_grid to False to fix NaN error
            # keep a history of which runs successfully pruned
            with open(f"hyperparameters/{self.run_name}/{self.run_name}_pruned.txt", "a") as results:
                results.write("Model pruned and refit.\n")
        except RuntimeError:
            # and which ones didn't
            with open(f"hyperparameters/{self.run_name}/{self.run_name}_pruned.txt", "a") as results:
                results.write("PRUNING SKIPPED!!!\n")
        finally:
            # get the average r2 score of all of the outputs for hyperparameter tuning
            scaler = self.dataset["y_scaler"]
            X_test = self.dataset["test_input"]  # still scaled
            Y_test = self.dataset["test_output"]  # still scaled
            Y_pred = model(X_test)
            y_test = scaler.inverse_transform(Y_test.detach().numpy())  # unscaled
            y_pred = scaler.inverse_transform(Y_pred.detach().numpy())  # unscaled
            r2s = []
            for i in range(len(self.dataset["output_labels"])):
                yi_test = y_test[:, i]
                yi_pred = y_pred[:, i]
                r2s.append(r2_score(yi_test, yi_pred))
            # SHOULD WE DO SOMETHING HERE TO HANDLE NEGATIVE SCORES?
            r2s = np.array(r2s)
            avg_r2 = np.mean(r2s)
            # keeping track of our avg R2 scores for each run
            with open(f"hyperparameters/{self.run_name}/{self.run_name}_R2.txt", "a") as results:
                results.write(f"AVG R2 SCORE: {avg_r2}\n")
            # delete model folder at the end of the run
            shutil.rmtree("model")
            return -1 * avg_r2  # make negative because fmin is a minimizer

######################## OTHER FUNCTIONS ###########################
def standard_space():
    """Sets a standard space for hyperopt fmin function based on tuner params.

    Returns:
        dict: Dictionary with necessary hyperparameter spaces for hyperopt
    """
    space = {
        "depth": hp.quniform("depth", 1, 4, 1),
        "grid": hp.quniform("grid", 1, 10, 1),
        "k": hp.choice("k", [1, 2, 3, 4, 5]),
        "steps": hp.quniform("steps", 100, 200, 1),
        "lamb": hp.uniform("lamb", 0, 1),
        "lamb_entropy": hp.uniform("lamb_entropy", 0, 10),
        "lr_1": hp.choice("lr_1", [0.0001, 0.001, 0.01, 0.1, 1]),
        "lr_2": hp.choice("lr_2", [0.0001, 0.001, 0.01, 0.1, 1]),
    }
    return space


def tune(obj, space, max_evals, algorithm=None):
    trials = Trials()
    best = fmin(obj, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best, trials




if __name__ == "__main__":
    chf_tuner = Tuner(
                    dataset= get_chf(), 
                    run_name = "CHF_test", 
                    space = standard_space(), 
                    max_evals = 50, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    best, trials = tune(
                    obj=chf_tuner.objective, 
                    space=chf_tuner.space, 
                    max_evals=chf_tuner.max_evals)
    # write the best results to our folder                
    with open(f"hyperparameters/{chf_tuner.run_name}/{chf_tuner.run_name}_results.txt", "w") as results:
        results.write(str(best))
