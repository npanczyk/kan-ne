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
from hyperopt import tpe, hp, fmin
from main import NKAN

def tune(dataset, max_evals, algorithm=None):

    return

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_name = 'chf_123'
    dataset  = get_chf()
    test_kan = NKAN(dataset=dataset, seed=42, device=device)
    model = test_kan.get_model(test=False)
    #r2 = test_kan.get_model(test=False, tuning=True)