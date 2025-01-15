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
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['D', 'L', 'P', 'G', 'Tin', 'Xe'],
        'output_labels': ['CHF'],
        'y_scaler': scaler_Y
    }
    return dataset


def get_mitr(test_split=0.3, random_state=42):
    """Gets MIT microreactor data. Six features (six control blade hights) and 22 outputs (power produced by each fuel element in the core).

    Args:
        test_split (float, optional): Ratio of test to training set. Defaults to 0.3.
        random_state (int, optional): Random state to allow for reproducible shuffling. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['CR1', 'CR2', 'CR3', 'CR4', 'CR5', 'CR6'],
        'output_labels': ['A-2','B-1','B-2','B-4','B-5','B-7','B-8','C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10','C-11','C-12','C-13','C-14','C-15'],
        'y_scaler': scaler_Y
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
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double).unsqueeze(1)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['FissionFast', 'CaptureFast', 'FissionThermal', 'CaptureThermal', 'Scatter12', 'Scatter11', 'Scatter21', 'Scatter22'],
        'output_labels': ['k'],
        'y_scaler': scaler_Y
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
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['fuel_dens', 'porosity', 'clad_thick', 'pellet_OD', 'pellet_h', 'gap_thickness', 'inlet_T', 'enrich', 'rough_fuel', 'rough_clad', 'ax_pow', 'clad_T', 'pressure'],
        'output_labels': ['fission_gas', 'max_fuel_cl_T', 'max_fuel_surf_T', 'radial_clad_T'],
        'y_scaler': scaler_Y
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
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double).unsqueeze(1)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['qprime', 'mdot', 'Tin', 'R', 'L', 'Cp', 'k'],
        'output_labels': ['T'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_rea(test_split=0.3, random_state=42):
    """Gets rod ejection accident (REA) data.

    Features:
    - ``rod_worth``: reactivity worth of the ejected rod,
    - ``beta``: delayed neutron fraction,
    - ``h_gap``: gap conductance :math:`[W/(m^2\\cdot K)]`,
    - ``gamma_frac``: direct heating fraction

    Outputs:
    - ``max_power``: peak power reached during transient :math:`[\\%FP]`,
    - ``burst_width``: Width of power burst :math:`[s]`,
    - ``max_TF``: max fuel centerline temperature :math:`[K]`,
    - ``avg_Tcool``: average coolant outlet temperature :math:`[K]`.

    Args:
        test_split (float, optional): Ratio of test to train data. Defaults to 0.3.
        random_state (int, optional): Sets random state to allow for reproducible shuffling. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
    """
    features_df = pd.read_csv('datasets/rea_inputs.csv')
    outputs_df = pd.read_csv('datasets/rea_outputs.csv')
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['rod_worth', 'beta', 'h_gap', 'gamma_frac'],
        'output_labels': ['max_power', 'burst_width', 'max_TF', 'avg_Tcool'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_bwr(test_split=0.3, random_state=42):
    """Gets BWR data.

    Features:
    - ``PSZ``: Fuel bundle region Power Shaping Zone (PSZ),
    - ``DOM``:  Fuel bundle region Dominant zone (DOM),
    - ``vanA``: Fuel bundle region vanishing zone A (VANA),
    - ``vanB``: Fuel bundle region vanishing zone B (VANB),
    - ``subcool``: Represents moderator inlet conditions. Core inlet subcooling
      is interpreted to be at the steam dome pressure (i.e., not core-averaged
      pressure). The input value for subcooling will automatically be increased
      to account for this fact. (Btu/lb),
    - ``CRD``: Defines the position of all control rod groups (banks),
    - ``flow_rate``: Defines essential global design data for rated coolant mass
      flux for the active core, :math:`\\frac{kg}{(cm^{2}-hr)}`. Coolant   mass
      flux equals active core flow divided by core cross-section area. The core
      cross-section area is DXA 2 times the number of assemblies,
    - ``power_density``: Defines essential global design data for rated power
      density using cold dimensions, :math:`(\\frac{kw}{liter})`,
    - ``VFNGAP``: Defines the ratio of narrow water gap width to the sum of the
      narrow and wide water gap widths,

    Outputs:
    - ``K-eff``:  Reactivity coefficient k-effective, the effective neutron
      multiplication factor,
    - ``Max3Pin``: Maximum planar-averaged pin power peaking factor,
    - ``Max4Pin``: maximum pin-power peaking factor, :math:`F_{q}`, (which includes
      axial intranodal peaking),
    - ``F-delta-H``: Ratio of max-to-average enthalpy rise in a channel,
    - ``Max-Fxy``: Maximum radial pin-power peaking factor,

    Args:
        test_split (float, optional): Ratio of test to train data. Defaults to 0.3.
        random_state (int, optional): Sets random state to allow for reproducible shuffling. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
    """
    features_df = pd.read_csv('datasets/bwr_input.csv')
    outputs_df = pd.read_csv('datasets/bwr_output.csv')
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['PSZ', 'DOM', 'vanA', 'vanB', 'subcool', 'CRD', 'flow_rate', 'power_density', 'VFNGAP'],
        'output_labels': ['K-eff', 'Max3Pin', 'Max4Pin', 'F-delta-H', 'Max-Fxy'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_htgr(test_split=0.3, random_state=42):
    """Gets high temperature gas reactor (HTGR) data:
    Features:
    - ``theta_{1}``: Angle of control drum in quadrant 1 (degrees),
    - ``theta_{2}``: Angle of control drum in quadrant 1 (degrees),
    - ``theta_{3}``: Angle of control drum in quadrant 2 (degrees),
    - ``theta_{4}``: Angle of control drum in quadrant 2 (degrees),
    - ``theta_{5}``: Angle of control drum in quadrant 3 (degrees),
    - ``theta_{6}``: Angle of control drum in quadrant 3 (degrees),
    - ``theta_{7}``: Angle of control drum in quadrant 4 (degrees),
    - ``theta_{8}``: Angle of control drum in quadrant 4 (degrees),

    Outputs:
    - ``FluxQ1``: Neutron flux in quadrant 1 :math:`(\\frac{neutrons}{cm^{2} s})`,
    - ``FluxQ2``: Neutron flux in quadrant 2 :math:`(\\frac{neutrons}{cm^{2} s})`,
    - ``FluxQ3``: Neutron flux in quadrant 3 :math:`(\\frac{neutrons}{cm^{2} s})`,
    - ``FluxQ4``: Neutron flux in quadrant 4 :math:`(\\frac{neutrons}{cm^{2} s})`,

    Args:
        test_split (float, optional): Ratio of test to train data. Defaults to 0.3.
        random_state (int, optional): Sets random state to allow for reproducible shuffling. Defaults to 42.

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
    """
    features_df = pd.read_csv('datasets/microreactor.csv').iloc[:,[29,30,31,32,33,34,35,36]]
    outputs_df = pd.read_csv('datasets/microreactor.csv').iloc[:, [4,5,6,7]]
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
    train_output = torch.tensor(Y_train, dtype=torch.double)
    test_input = torch.tensor(X_test, dtype=torch.double)
    test_output = torch.tensor(Y_test, dtype=torch.double)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8'],
        'output_labels': ['FluxQ1', 'FluxQ2', 'FluxQ3', 'FluxQ4'],
        'y_scaler': scaler_Y
    }
    return dataset

'''dataset = get_chf()
hidden_nodes = dataset['train_input'].shape[1]
print( hidden_nodes )
print( type(hidden_nodes) )'''
