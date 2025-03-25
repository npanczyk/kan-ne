import numpy as np
import torch
from hyperopt import tpe, hp, fmin, Trials, rand
from preprocessing import *
from functools import partial
from sklearn.metrics import r2_score
import shutil
import os
#Sklearn tools
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#Keras specials
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

torch.set_default_dtype(torch.float64)

'''
ULTIMATELY DID NOT USE THIS SCRIPT, USED PYMAISE BEST HYPERPARAMS INSTEAD
'''

########################  FNN TUNER CLASS #########################

class FNN_Tuner():
    def __init__(self, dataset, run_name, space, max_evals, seed, device):
        self.dataset = dataset
        self.run_name = run_name
        self.space = space
        self.max_evals = max_evals
        self.seed = seed
        self.device = device
        try:
            os.makedirs(f"FNN_hyperparameters/{self.run_name}")
        except FileExistsError:
            files = os.listdir(f"FNN_hyperparameters/{self.run_name}")
            for file in files:
                os.remove(f"FNN_hyperparameters/{self.run_name}/{file}")


    def objective(self, params):
        # HYPERPARAMS TO TEST
        n_dens_layers = int(params['n_dens_layers'])
        nodes_per_layer = int(params['nodes_per_layer'])
        lr = params['lr']
        data = {
            "train_input": self.dataset["train_input"],
            "train_label": self.dataset["train_output"],
            "test_input": self.dataset["test_input"],
            "test_label": self.dataset["test_output"],
        }
        Xtrain = data["train_input"]
        Ytrain = data["train_label"]
        Xtest = data["test_input"]
        Ytest = data["test_label"]
        # set the number of nodes in each dense layer
        n_nodes = [nodes_per_layer for layer in range(n_dens_layers)]
        model = Sequential()
        # add input layer
        model.add(Input(shape=(Xtrain.shape[1],))) # make the input layer shape the number of training features
        # add first dense layer with 50 nodes
        model.add(Dense(n_nodes[0], kernel_initializer='normal', activation='relu'))
        # add dropout layer with a 50% dropout rate
        model.add(Dropout(0.5))
        # add the rest of the dense layers (except for the first) in a loop
        for i in range(1, n_dens_layers):
            model.add(Dense(n_nodes[i], kernel_initializer='normal', activation='relu'))
        # add output layer based on ytrain columns and linear activation
        model.add(Dense(Ytrain.shape[1], kernel_initializer='normal', activation='linear'))
        model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=lr), metrics=['mean_absolute_error'])
        reduce_lr = ReduceLROnPlateau('mean_absolute_error', factor=0.9, patience=5, verbose=True)
        checkpoint_best = ModelCheckpoint('best_model.keras', verbose=True, save_best_only=True, monitor='mean_absolute_error', mode='min')
        checkpoint_last = ModelCheckpoint('last_model.keras', verbose=True )
        cb_list = [reduce_lr, checkpoint_best, checkpoint_last]
        # load the best and last models and define new variables for these models
        best = load_model('best_model.keras')
        last = load_model('last_model.keras')
        # predict using both models
        Ybest = best.predict(Xtest)
        Ylast = last.predict(Xtest)
        model_metrics = {
            'R2':[]
        }
        Yguesses = [Ybest, Ylast]
        for y in Yguesses:
            model_metrics['R2'].append(r2_score(Ytest, y))
        avg_r2 = np.mean(model_metrics['R2'])
        with open(f"FNN_hyperparameters/{self.run_name}/{self.run_name}_R2.txt", "a") as results:
                results.write(f"AVG R2 SCORE: {avg_r2}\n")
        return -1*avg_r2

######################## OTHER FUNCTIONS ###########################
def standard_space():
    """Sets a standard space for hyperopt fmin function based on tuner params.

    Returns:
        dict: Dictionary with necessary hyperparameter spaces for hyperopt
    """
    space = {
        "n_dens_layers": hp.quniform("depth", 1, 10, 1),
        "nodes_per_layer": hp.quniform("grid", 50, 200, 10),
        "lr": hp.choice("lr_1", [0.0001, 0.0006, 0.01, 0.1, 0.0009]),
    }
    return space


def tune(obj, space, max_evals, algorithm=None):
    trials = Trials()
    best = fmin(obj, space=space, algo=rand.suggest, max_evals=max_evals, trials=trials)
    return best, trials

##################### TUNING INDIVIDUAL DATASETS #####################

def tune_case(tuner):
    best, trials = tune(
                    obj=tuner.objective, 
                    space=tuner.space, 
                    max_evals=tuner.max_evals)
    # write the best results to our folder                
    with open(f"hyperparameters/{tuner.run_name}/{tuner.run_name}_results.txt", "w") as results:
        results.write(str(best))
    return 


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    chf_tuner = FNN_Tuner(
                    dataset = get_chf(cuda=True), 
                    run_name = "CHF_250205", 
                    space = standard_space(), 
                    max_evals = 3, 
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))              
    tune_case(chf_tuner)

# need to install gpu in environment
    
