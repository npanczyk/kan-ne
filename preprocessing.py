# figure out what you want to keep from the pyMAISE _handler.py file
# need to write a class to load datsets, split into training and testing, scale
# the datasets, convert them into Torch tensors, and then create a dataset
# directory

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

# putting this on the backburner for now
'''class data:
    def __init__(self, filepath, column_names, output_names, filepath2=None, train_split=0.7):
        self.filepath = filepath
        self.column_names = column_names
        self.output_names = output_names
        self.split = train_split
        if filepath2:
            self.filepath2 = filepath2

    def load_data(self.filepath, self.filepath2=None):
        if filepath2==None:
            df = pd.read_csv(filepath)'''


def get_chf(synthetic=False):
    """
    Gets data for CHF prediction.

    Features:
    - ``D (m)``: Diameter of the test section (:math:`0.002 - 0.016~m`),
    - ``L (m)``: Heated length (:math:`0.07 - 15.0~m`),
    - ``P (kPa)``: Pressure (:math:`100-20000~kPa`),
    - ``G (kg m-2s-1)``: Mass flux (:math:`17.7-7712.0~\\frac{kg}{m^2\\cdot s}`),
    - ``Tin (C)``: Inlet temperature length (:math:`9.0-353.62^\\circ C`),
    - ``Xe (-)``: Outlet equilibrium quality (:math:`-0.445-0.986`),

    Output:
    - ``CHF (kW m-2)``: Critical heat flux (:math:`130.0-13345.0~\\frac{kW}{m^2}`).
    Args:
        synthetic (bool, optional): Whether to use synthetic or real CHF data. Defaults to False.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_label, test_input, test_label)
    """
    if synthetic==False:
        train_df = pd.read_csv('datasets/chf_train.csv')
        test_df = pd.read_csv('datasets/chf_valid.csv')
    else:
        train_df = pd.read_csv('datasets/chf_train_synth.csv')
        test_df = pd.read_csv('datasets/chf_test_synth.csv')
    x_train = train_df.iloc[:, [0, 1, 2, 3, 4, 5]].values  # Input columns (1-6) D, L, P, G, T, Xe
    y_train = train_df.iloc[:, [6]].values  # CHF
    x_test = test_df.iloc[:, [0, 1, 2, 3, 4, 5]].values  
    y_test = test_df.iloc[:, [6]].values

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double)
    train_label = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_label = torch.tensor(Y_test, dtype=torch.double).unsqueeze(1)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    return dataset


def get_mitr(test_split=0.3, random_state=42):
    """Gets MIT microreactor data. Six features and 22 outputs.

    Args:
        test_split (float, optional): Ratio of test to training set. Defaults to 0.3.
        random_state (int, optional): Random state to allow for reproducible shuffling. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_label, test_input, test_label)
    """
    features_df = pd.read_csv('datasets/crx.csv')
    outputs_df = pd.read_csv('datasets/powery.csv')
    x_train, x_test, y_train, y_test = train_test_split(
    features_df, outputs_df, test_size=0.3, random_state=random_state)

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double)
    train_label = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_label = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    return dataset

def get_xs(test_split=0.3, random_state=42):
    """Gets reactor physics data ready for KAN.
    Features (cross sections): 
    - ``FissionFast``: fast fission,
    - ``CaptureFast``: fast capture,
    - ``FissionThermal``: thermal fission,
    - ``CaptureThermal``: thermal capture,
    - ``Scatter12``: group 1 to 2 scattering,
    - ``Scatter11``: group 1 to 1 scattering,
    - ``Scatter21``: group 2 to 1 scattering,
    - ``Scatter22``: group 2 to 2 scattering,
    Outputs:
    - k: criticality

    Args:
        test_split (float, optional): Ratio of test to training. Defaults to 0.3.
        random_state (int, optional): Random state to make reproducible results when shuffling the data. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_label, test_input, test_label)
    """
    features_df = pd.read_csv('datasets/xs.csv').iloc[:,[0,1,2,3,4,5,6,7]]
    outputs_df = pd.read_csv('datasets/xs.csv').iloc[:, [8]]
    x_train, x_test, y_train, y_test = train_test_split(
    features_df, outputs_df, test_size=0.3, random_state=random_state)

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double)
    train_label = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_label = torch.tensor(Y_test, dtype=torch.double).unsqueeze(1)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    return dataset

def get_fp(test_split=0.3, random_state=42):
    """
    Gets fuel performance data ready for KAN.

    Features:
     - ``fuel_dens``: fuel density :math:`[kg/m^3]`,
    - ``porosity``: porosity,
    - ``clad_thick``: cladding thickness :math:`[m]`,
    - ``pellet_OD``: pellet outer diameter :math:`[m]`,
    - ``pellet_h``: pellet height :math:`[m]`,
    - ``gap_thickness``: gap thickness :math:`[m]`,
    - ``inlet_T``: inlet temperature :math:`[K]`,
    - ``enrich``: U-235 enrichment,
    - ``rough_fuel``: fuel roughness :math:`[m]`,
    - ``rough_clad``: cladding roughness :math:`[m]`,
    - ``ax_pow``: axial power,
    - ``clad_T``: cladding surface temperature :math:`[K]`,
    - ``pressure``: pressure :math:`[Pa]`,

    Outputs:

    - ``fis_gas_produced``: fission gas production :math:`[mol]`,
    - ``max_fuel_centerline_temp``: max fuel centerline temperature :math:`[K]`,
    - ``max_fuel_surface_temperature``: max fuel surface temperature :math:`[K]`,
    - ``radial_clad_dia``: radial cladding diameter displacement after
      irradiation :math:`[m]`,

    Args:
        test_split (float, optional): Test to train ratio. Defaults to 0.3.
        random_state (int, optional): Makes shuffling reproducible. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_label, test_input, test_label)
    """
    features_df = pd.read_csv('datasets/fp_inp.csv')
    outputs_df = pd.read_csv('datasets/fp_out.csv')
    x_train, x_test, y_train, y_test = train_test_split(
    features_df, outputs_df, test_size=0.3, random_state=42)

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double)
    train_label = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_label = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    return dataset

def get_heat(test_split=0.3, random_state=42):
    """Gets heat conduction data:
    Features:
     - ``qprime``: linear heat generation rate :math:`[W/m]`,
    - ``mdot``: mass flow rate :math:`[g/s]`,
    - ``Tin``: temperature of the fuel boundary :math:`[K]`,
    - ``R``: fuel radius :math:`[m]`,
    - ``L``: fuel length :math:`[m]`,
    - ``Cp``: heat capacity :math:`[J/(g\\cdot K)]`,
    - ``k``: thermal conductivity :math:`[W/(m\\cdot K)]`,
    Outputs:
    - ``T``: fuel centerline temperature :math:`[K]`

    Args:
        test_split (float, optional): Ratio of test to train data. Defaults to 0.3.
        random_state (int, optional): Sets random state to allow for reproducible shuffling. Defaults to 42.

    Returns:
        _type_: _description_
    """
    features_df = pd.read_csv('datasets/heat.csv').iloc[:,[0,1,2,3,4,5,6]]
    outputs_df = pd.read_csv('datasets/heat.csv').iloc[:, [7]]
    x_train, x_test, y_train, y_test = train_test_split(
    features_df, outputs_df, test_size=0.3, random_state=random_state)

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double)
    train_label = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_label = torch.tensor(Y_test, dtype=torch.double).unsqueeze(1)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    return dataset

dataset = get_heat()
print( dataset['train_input'] )
print( len(dataset['train_input']) )
print( len(dataset['test_input']) )
