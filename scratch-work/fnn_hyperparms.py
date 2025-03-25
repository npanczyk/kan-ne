import numpy as np 
import pandas as pd 
from preprocessing import *
from functools import partial

'''
QUICK SCRIPT TO GENERATE FNN HYPERPARAMETER TABLE IN LATEX DOC
'''

pymaise_params = {
    'chf': {
        'hidden_nodes' : [231, 138, 267],
        'num_epochs' : 200,
        'batch_size' : 64,
        'learning_rate' : 0.0009311391232267503,
        'use_dropout': True,
        'dropout_prob': 0.4995897609454529, 
    },
    'bwr': {
        'hidden_nodes' : [511, 367, 563, 441, 162],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0009660778027367906,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'fp': {
        'hidden_nodes' : [66, 400],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.001,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'heat': {
        'hidden_nodes' : [251, 184, 47],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0008821712781015931,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'htgr': {
        'hidden_nodes' : [199, 400],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.00011376283985074373,
        'use_dropout': True,
        'dropout_prob': 0.3225718287912892,
    },
    'mitr': {
        'hidden_nodes' : [309],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0008321972582830564,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'rea': {
        'hidden_nodes' : [326, 127],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0009444837105276597,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'xs': {
        'hidden_nodes' : [95],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0003421585453407753,
        'use_dropout': False,
        'dropout_prob': 0,            
    },
    'mitr_a': {
        'hidden_nodes' : [309],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0008321972582830564,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'mitr_b': {
        'hidden_nodes' : [309],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0008321972582830564,
        'use_dropout': False,
        'dropout_prob': 0,
    },
    'mitr_c': {
        'hidden_nodes' : [309],
        'num_epochs' : 200,
        'batch_size' : 8,
        'learning_rate' : 0.0008321972582830564,
        'use_dropout': False,
        'dropout_prob': 0,
    },        
}
old_keys = pymaise_params.keys()
new_keys = [key.upper() for key in old_keys]
new_dict = dict(zip(new_keys, pymaise_params.values()))
df = pd.DataFrame(new_dict)
print(df.to_latex(longtable=True, escape='latex', caption='Hyperparameters used for FNN architecture and training.'))