import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(importances, labels):
    fig, ax = plt.subplots()
    x_vals = np.arange(0, len(labels))
    ax.bar(x_vals, importances, width=0.4, tick_label=labels)
    ax.set_ylabel('Relative Feature Importances')
    return fig

if __name__=="__main__":
    fig = plot_feature_importances([0.4, 0.3, 0.1], ['red', 'orange', 'yellow'])
    fig.savefig('test.png', dpi=300)
