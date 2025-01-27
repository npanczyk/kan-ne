# this is where we will do all the hyperparameter tuning 
# see: https://kindxiaoming.github.io/pykan/API_demo/API_6_training_hyperparameter.html for tuning advice

#PSEUDO-CODE 
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
from kan import KAN

def objective(depth, 
              grid, 
              k, 
              steps, 
              lamb, 
              lamb_entropy, 
              dataset=None, 
              seed=None, 
              device=None
              ):
    """_summary_

    Args:
        dataset (dict): a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output) and feature/output labels.
        depth (int): The number of hidden layers that contain equal number of nodes to the size of the feature set.
        grid (int): The number of grid intervals used by pykan to generate the network. Should range from 1-10. 
        k (int): The spline order used by pykan. Should range from 1-5.
        steps (int): The number of training steps taken by pykan. Should range (10-200, by 10s)
        lamb (float): Overall penalty strength. Should range 0 to 1.
        lamb_entropy (float): Entropy penalty strength. Should range 0 to 10. 

    Returns:
        _type_: _description_
    """
    hidden_nodes_per_layer = dataset['train_input'].shape[1]
    hidden_nodes = [hidden_nodes_per_layer for i in range(depth)]
    print(hidden_nodes)
    width = [dataset['train_input'].shape[1]] + hidden_nodes + [dataset['train_output'].shape[1]]
    print(width)
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device)
    data = {
        'train_input': dataset['train_input'],
        'train_label': dataset['train_output'],
        'test_input': dataset['test_input'],
        'test_label': dataset['test_output']
    }
    model.fit(data, opt='LBFGS', steps=steps, lamb=lamb, lamb_entropy=lamb_entropy)
    print("Model trained.")
    model = model.prune()
    model.fit(data, opt='LBFGS', steps=steps, lamb=lamb, lamb_entropy=lamb_entropy)
    print("Model pruned and re-trained.")
    # get the average r2 score of all of the outputs for hyperparameter tuning
    scaler = dataset['y_scaler']
    X_test = dataset['test_input'] # still scaled
    Y_test = dataset['test_output'] # still scaled
    Y_pred = model(X_test)
    y_test = scaler.inverse_transform(Y_test.detach().numpy())
    y_pred = scaler.inverse_transform(Y_pred.detach().numpy())
    r2s = []
    for i in range(len(self.dataset['output_labels'])):
        yi_test = y_test[:,i]
        yi_pred = y_pred[:,i]
        r2s.append(r2_score(yi_test, yi_pred))
    avg_r2 = np.mean(r2s)
    print(av_r2)
    return {'loss': -1*avg_r2, 'status': STATUS_OK }

def set_space(dataset):
    space = {
        "depth": hp.quniform('depth', 1, 4, 1), 
        "grid": hp.quniform('grid', 1, 10, 1), 
        "k": hp.choice('k', [1,2,3,4,5]), 
        "steps":hp.quniform('steps', 10, 200, 1), 
        "lamb": hp.uniform('lamb', 0, 1), 
        "lamb_entropy": hp.uniform('lamb_entropy', 0, 10)
    }
    return space


def tune(test_set, seed, device, space, max_evals, algorithm=None):
    trials = Trials()
    best = fmin(partial(objective, 
                dataset=test_set, 
                seed=seed, 
                device=device
                ),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
                )
    return best, trials


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chf_dataset  = get_chf()
    seed = 42
    space = set_space(chf_dataset)
    # print(dataset)
    # best, trials = tune(test_set=chf_dataset, 
    #                     seed=seed, 
    #                     device=device, 
    #                     space=space, 
    #                     max_evals=10
    #                     )

    obj = partial(objective, 
                dataset=chf_dataset, 
                seed=seed, 
                device=device
                )

    test_points = {
        "depth": 1, 
        "grid": 10, 
        "k":3, 
        "steps": 200, 
        "lamb": 1, 
        "lamb_entropy": 0
    }
    loss = obj(**test_points)
    print(loss)