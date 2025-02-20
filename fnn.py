import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing import *
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from accessories import *

class FNN(nn.Module):
    def __init__(self, input_size, hidden_nodes, output_size):
        super(FNN, self).__init__()
        layers = []
        # define input layer
        layers.append(nn.Linear(input_size, hidden_nodes[0]))
        layers.append(nn.ReLU())
        # loop through layers in hidden nodes
        for i in range(1, len(hidden_nodes)):
            print(f'i: {i}, hidden_nodes[i-1]: {hidden_nodes[i-1]}, hidden_nodes[i]: {hidden_nodes[i]}')
            layers.append(nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
            layers.append(nn.ReLU())
        # define output layer
        layers.append(nn.Linear(hidden_nodes[-1], output_size))
        # stick all the layers in the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def fit_fnn(dataset, params, plot=False, save_as=None):
    # define hyperparams
    input_size = dataset['train_input'].shape[1]
    print(f'Input Size: {input_size}')
    hidden_nodes = params['hidden_nodes']
    output_size = dataset['train_output'].shape[1]
    print(f'Output Size: {output_size}')
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
    model = FNN(input_size, hidden_nodes, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    all_losses = []
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device).float(), y_train.to(device).float()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
        if epoch%10 == 0:
            print(loss)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(all_losses)
        ax.set_ylabel('Training Loss')
        ax.set_xlabel('Epoch')
        plt.savefig(f'figures/fnn_loss_{save_as}.png', dpi=300)

    # evaluate model performance
    y_preds, y_tests = get_metrics(model, test_loader, dataset['y_scaler'], save_as=save_as)
    return

def get_metrics(model, test_loader, scaler, save_as, p=5):
    model.eval()
    y_preds = []
    y_tests = []
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device).float(), y_test.to(device).float()
            # get prediction
            y_pred = model(x_test)
            # unscale y_test and y_pred
            y_test_unscaled = scaler.inverse_transform(y_test.cpu().detach().numpy())
            y_pred_unscaled = scaler.inverse_transform(y_pred.cpu().detach().numpy())
            # append the tests and predictions to lists
            y_tests.append(y_test_unscaled)
            y_preds.append(y_pred_unscaled)
        y_tests = np.concatenate(y_tests, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
    
    metrics = {
            'OUTPUT': dataset['output_labels'],
            'MAE':[],
            'MAPE':[],
            'MSE':[],
            'RMSE':[],
            'RMSPE':[],
            'R2':[]
        }
    for i in range(len(dataset['output_labels'])):
        # get metrics for each output
        yi_test = y_tests[:,i]
        yi_pred = y_preds[:,i]
        print(f'yi_test: {yi_test.shape}')
        print(f'yi_pred: {yi_pred.shape}')
        metrics['MAE'].append(round(mean_absolute_error(yi_test, yi_pred), p))
        metrics['MAPE'].append(round(mape(yi_test, yi_pred), p))
        metrics['MSE'].append(round(mean_squared_error(yi_test, yi_pred), p))
        metrics['RMSE'].append(round(np.sqrt(mean_squared_error(yi_test, yi_pred)), p))
        metrics['RMSPE'].append(round(rmspe(yi_test, yi_pred), p))
        metrics['R2'].append(round(r2_score(yi_test, yi_pred),p))
    metrics_df = pd.DataFrame.from_dict(metrics)
    # check to see if there 
    if not os.path.exists('results'):
        os.makedirs('results')
    metrics_df.to_csv(f'results/{save_as}.csv', index=False)

    return y_preds, y_tests


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_htgr(cuda=True)
    params = {
        'hidden_nodes' : [437, 258, 101, 75, 35],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 9e-4
    }
    fit_fnn(dataset, params, plot=True, save_as='HTGR_fnn') 