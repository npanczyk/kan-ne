import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing import *
from torch.utils.data import DataLoader, TensorDataset
import os

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def fit_fnn(dataset, params):
    # define hyperparams
    input_size = dataset['train_input'].shape[1]
    hidden_size = params['hidden_size']
    output_size = dataset['train_output'].shape[1]
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']

    # get train and test data from dataset
    train_data = TensorDataset(dataset['train_input'], dataset['train_output'])
    test_data = TensorDataset(dataset['test_input'], dataset['test_output'])

    # write dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # define the model
    model = FNN(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    costval = []
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device).float(), y_train.to(device).float()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train.reshape(-1,1))
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%50 == 0:
            print(loss)
            costval.append(loss.item())

    # evaluate model performance


    return model 




if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_htgr(cuda=True, quadrant=1)
    params = {
        'hidden_size' : 100,
        'num_epochs' : 50,
        'batch_size' : 100,
        'learning_rate' : 0.001
    }
    fit_fnn(dataset, params) 