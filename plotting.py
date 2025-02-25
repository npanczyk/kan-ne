import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(importances, labels):
    fig, ax = plt.subplots(figsize =(8, 10))
    x_vals = np.arange(0, len(labels))
    ax.bar(x_vals, importances, width=0.4)
    ax.set_ylabel('Relative Feature Importances')
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(labels, rotation=45)
    return fig

def plot_overfitting(n_params, train_rmse, test_rmse, cont_train_rmse, cont_test_rmse, save_as):
    fig, ax1 = plt.subplots()
    ax1.plot(n_params, train_rmse, marker="o")
    ax1.plot(n_params, test_rmse, marker="o")
    ax1.legend(['train', 'test'], loc="lower left")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('RMSE')
    fig.savefig(f'figures/{save_as}_nparams.png', dpi=300)

    fig, ax2 = plt.subplots()
    ax2.plot(cont_train_rmse, linestyle='dashed')
    ax2.plot(cont_test_rmse)
    ax2.legend(['train', 'test'])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('RMSE')
    ax2.set_yscale('log')
    fig.savefig(f'figures/{save_as}.png', dpi=300)
    return

if __name__=="__main__":
    fig = plot_feature_importances([0.4, 0.3, 0.1], ['red', 'orange', 'yellow'])
    fig.savefig('test.png', dpi=300)
