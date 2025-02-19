import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FNN

def fit_fnn(dataset, params):
    # define hyperparams
    input_size = dataset['train_input'].shape[1]
    hidden_size = params['hidden_size']
    num_classes = params['num_classes']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']

    # get train and test data from dataset
    train_data = torch.cat((dataset['train_input'], dataset['train_output']), 1)
    test_data = torch.cat((dataset['test_input'], dataset['test_output']), 1)

    # write dataloaders
    train_loader = torch.utils.data.DataLoader(datset=train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datset=test_dataset, batch_size=batchsize, shuffle=False)


if __name__=="__main__":
    dataset = get_htgr(cuda=True, quadrant=1)
    params = {
        'hidden_size' : 100,
        'num_classes' : 10,
        'num_epochs' : 2,
        'batch_size' : 100,
        'learning_rate' : 0.001
    }
    fit_fnn(dataset, params) 