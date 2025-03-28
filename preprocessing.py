import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import torch
import xarray as xr


def get_chf(synthetic=False, cuda=False):
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
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
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
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['D', 'L', 'P', 'G', 'T_in', 'Xe'],
        'output_labels': ['CHF'],
        'y_scaler': scaler_Y
    }
    return dataset


def get_mitr(test_split=0.3, random_state=42, cuda=False, region='FULL'):
    """Gets MIT microreactor data. Six features (six control blade hights) and 22 outputs (power produced by each fuel element in the core).

    Args:
        test_split (float, optional): Ratio of test to training set. Defaults to 0.3.
        random_state (int, optional): Random state to allow for reproducible shuffling. Defaults to 42.
        cuda (bool): Allows for gpu usage if set to True
        region (str): A, B, or C region of the MIT microreactor (alphabetical moving out from the center of the core)

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels.
    """
    if region.upper() == 'FULL':
        output_cols = ['A-2','B-1','B-2','B-4','B-5','B-7','B-8','C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10','C-11','C-12','C-13','C-14','C-15']
    elif region.upper() == 'A':
        output_cols = ['A-2']
    elif region.upper() == 'B':
        output_cols = ['B-1','B-2','B-4','B-5','B-7','B-8']
    elif region.upper() == 'C':
        output_cols = ['C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10','C-11','C-12','C-13','C-14','C-15']

    features_df = pd.read_csv('datasets/crx.csv')
    outputs_df = pd.read_csv('datasets/powery.csv', usecols=output_cols)
    x_train, x_test, y_train, y_test = train_test_split(
    features_df, outputs_df, test_size=0.3, random_state=random_state)

    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5', 'CR_6'],
        'output_labels': output_cols,
        'y_scaler': scaler_Y
    }
    return dataset


def get_xs(test_split=0.3, random_state=42, cuda=False):
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

    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['fission_fast', 'capture_fast', 'fission_thermal', 'capture_thermal', 'scatter_12', 'scatter_11', 'scatter_21', 'scatter_22'],
        'output_labels': ['k'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_fp(test_split=0.3, random_state=42, cuda=False):
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

    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['fuel_density', 'porosity', 'thickness_clad', 'pellet_OD', 'pellet_h', 'thickness_gap', 'T_inlet', 'enrich', 'fuel_rough', 'clad_rough', 'power_ax', 'T_clad', 'pressure'],
        'output_labels': ['fission_gas', 'max_fuel_cl_T', 'max_fuel_surf_T', 'radial_clad_T'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_heat(test_split=0.3, random_state=42, cuda=False):
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
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['qprime', 'mdot', 'T_in', 'R', 'L', 'C_p', 'k'],
        'output_labels': ['T'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_rea(test_split=0.3, random_state=42, cuda=False):
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

    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

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

def get_bwr(test_split=0.3, random_state=42, cuda=False):
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

    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['PSZ', 'DOM', 'vanA', 'vanB', 'subcool', 'CRD', 'FlowRate', 'PowerDensity', 'VFNGAP'],
        'output_labels': ['K-eff', 'Max3Pin', 'Max4Pin', 'F-delta-H', 'Max-Fxy'],
        'y_scaler': scaler_Y
    }
    return dataset

def get_htgr(random_state=42, cuda=False, quadrant=None):
    train_df = pd.read_csv('datasets/htgr_train.csv')
    test_df = pd.read_csv('datasets/htgr_valid.csv')
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    if quadrant is None:
        flux_cols = ['fluxQ1','fluxQ2','fluxQ3','fluxQ4']
    else:
        flux_cols = [f'fluxQ{quadrant}']
    x_train = train_df.loc[:,['theta1','theta2','theta3','theta4','theta5','theta6','theta7','theta8']].values
    y_train = train_df.loc[:,flux_cols].values  # CHF
    x_test = test_df.loc[:,['theta1','theta2','theta3','theta4','theta5','theta6','theta7','theta8']].values  
    y_test = test_df.loc[:,flux_cols].values

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['theta_1','theta_2','theta_3','theta_4','theta_5','theta_6','theta_7','theta_8'],
        'output_labels': flux_cols,
        'y_scaler': scaler_Y
    }
    return dataset

def reflect_htgr(test_split=0.3, random_state=42, normalize=False):
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
    """
    theta_cols = [f"theta{i + 1}" for i in range(8)]
    flux_cols = [f"fluxQ{i + 1}" for i in range(4)]
    data = (
            pd.read_csv('datasets/microreactor.csv', header="infer")
            .to_xarray()
            .to_array()
            .transpose(..., "variable")
        )

    # slice the data based on features and outputs
    input_slice = slice(29, 37)
    output_slice = slice(4, 8)
    inputs = data.isel(variable=input_slice)
    outputs = data.isel(variable=output_slice)
    # combine slices into one xarray
    data = xr.concat([inputs, outputs], dim=data.dims[-1])
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(
    inputs, outputs, test_size=0.3, random_state=random_state)
    # merge into training and testing for mult_samples()
    train_data = xr.concat([x_train, y_train], dim=data.dims[-1])
    test_data = xr.concat([x_test, y_test], dim=data.dims[-1])
    # feed to mult_samples() to get reflected dataset (3004 samples)
    sym_train_data = mult_samples(train_data)
    sym_test_data = mult_samples(test_data)

    # save a CSV of the reflected data
    train_df = sym_train_data.to_pandas()
    test_df = sym_test_data.to_pandas()
    if normalize==True:
        for df in [train_df, test_df]:
            total = df['fluxQ1'].to_numpy(dtype=np.float64) + df['fluxQ2'].to_numpy(dtype=np.float64)+ df['fluxQ3'].to_numpy(dtype=np.float64)+ df['fluxQ4'].to_numpy(dtype=np.float64)
            df['fluxQ1'] = df['fluxQ1']/total
            df['fluxQ2'] = df['fluxQ2']/total
            df['fluxQ3'] = df['fluxQ3']/total
            df['fluxQ4'] = df['fluxQ4']/total
    train_df.to_csv('datasets/htgr_train.csv')
    test_df.to_csv('datasets/htgr_valid.csv')
    return

def mult_samples(data):
    # Credit to mult_sym from https://github.com/deanrp2/MicroControl/blob/main/pmdata/utils.py#L51
    # Credit to mult_samples from https://github.com/aims-umich/pyMAISE

    theta_cols = [f"theta{i + 1}" for i in range(8)]
    flux_cols = [f"fluxQ{i + 1}" for i in range(4)]
    # Create empty arrays
    ht = xr.DataArray(
        np.zeros(data.shape),
        coords={
            "index": [f"{idx}_h" for idx in data.coords["index"].values],
            "variable": data.coords["variable"],
        },
    )
    vt = xr.DataArray(
        np.zeros(data.shape),
        coords={
            "index": [f"{idx}_v" for idx in data.coords["index"].values],
            "variable": data.coords["variable"],
        },
    )
    rt = xr.DataArray(
        np.zeros(data.shape),
        coords={
            "index": [f"{idx}_r" for idx in data.coords["index"].values],
            "variable": data.coords["variable"],
        },
    )

    # Swap drum positions
    hkey = [f"theta{i}" for i in np.array([3, 2, 1, 0, 7, 6, 5, 4], dtype=int) + 1]
    vkey = [f"theta{i}" for i in np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=int) + 1]
    rkey = [f"theta{i}" for i in np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=int) + 1]

    ht.loc[:, hkey] = data.loc[:, theta_cols].values
    vt.loc[:, vkey] = data.loc[:, theta_cols].values
    rt.loc[:, rkey] = data.loc[:, theta_cols].values

    # Adjust angles
    ht.loc[:, hkey] = (3 * np.pi - ht.loc[:, hkey].loc[:, hkey]) % (2 * np.pi)
    vt.loc[:, vkey] = (2 * np.pi - vt.loc[:, hkey].loc[:, vkey]) % (2 * np.pi)
    rt.loc[:, rkey] = (np.pi + rt.loc[:, hkey].loc[:, rkey]) % (2 * np.pi)

    # Fill quadrant tallies
    hkey = [2, 1, 4, 3]
    vkey = [4, 3, 2, 1]
    rkey = [3, 4, 1, 2]

    ht.loc[:, [f"fluxQ{i}" for i in hkey]] = data.loc[:, flux_cols].values
    vt.loc[:, [f"fluxQ{i}" for i in vkey]] = data.loc[:, flux_cols].values
    rt.loc[:, [f"fluxQ{i}" for i in rkey]] = data.loc[:, flux_cols].values

    sym_data = xr.concat([data, ht, vt, rt], dim="index").sortby("index")

    # Normalize fluxes
    sym_data.loc[:, flux_cols].values = Normalizer().transform(sym_data.loc[:, flux_cols].values)

    # Convert global coordinate system to local
    loc_offsets = np.array(
        [3.6820187359906447, 4.067668586955522, 2.2155167202240653 - np.pi, 2.6011665711889425 - np.pi,
         0.5404260824008517, 0.9260759333657285, 5.3571093738138575 - np.pi, 5.742759224778734 - np.pi]
    )

    # Apply correct 0 point
    sym_data.loc[:, theta_cols] = sym_data.loc[:, theta_cols] - loc_offsets + 2 * np.pi

    # Reverse necessary angles
    sym_data.loc[:, [f"theta{i}" for i in [3,4,5,6]]] *= -1

    # Scale all to [0, 2 * np.pi]
    sym_data.loc[:, theta_cols] = sym_data.loc[:, theta_cols] % (2 * np.pi)

    return sym_data
