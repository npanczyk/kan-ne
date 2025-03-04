from main import *
from preprocessing import *
import shutil
import os
from kan import *
import datetime as dt

fp_best = {'depth': 1, 'grid': 8, 'k': 7, 'lamb': 2.0426962767412815e-05, 'lamb_entropy': 5.0346373804560525, 'lr_1': 1.5, 'lr_2': 1.75, 'reg_metric': 'edge_forward_sum', 'steps': 75}
bwr_best = {'depth': 1, 'grid': 7, 'k': 2, 'lamb': 0.0008912210456241697, 'lamb_entropy': 7.488094627223641, 'lr_1': 1.75, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 125}
heat_best = {'depth': 1, 'grid': 7, 'k': 3, 'lamb': 0.00018986520634595234, 'lamb_entropy': 8.209205342922996, 'lr_1': 1.5, 'lr_2': 2, 'reg_metric': 'edge_forward_spline_u', 'steps': 150}
htgr_best = {'depth': 1, 'grid': 8, 'k': 3, 'lamb': 1.2166058649376336e-05, 'lamb_entropy': 7.665809430995193, 'lr_1': 0.75, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 25}
mitr_a_best = {'depth': 1, 'grid': 4, 'k': 7, 'lamb': 4.2843417969236176e-05, 'lamb_entropy': 0.025403293360453105, 'lr_1': 1.5, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 25}
mitr_b_best = {'depth': 1, 'grid': 3, 'k': 2, 'lamb': 1.1115022163426145e-06, 'lamb_entropy': 0.9626840845416629, 'lr_1': 0.75, 'lr_2': 1, 'reg_metric': 'edge_forward_sum', 'steps': 250}
mitr_c_best = {'depth': 1, 'grid': 3, 'k': 3, 'lamb': 9.739068562318698e-06, 'lamb_entropy': 7.047425350169997, 'lr_1': 1, 'lr_2': 1.5, 'reg_metric': 'edge_forward_spline_n', 'steps': 150}
mitr_best = {'depth': 1, 'grid': 7, 'k': 6, 'lamb': 0.00033697982852750485, 'lamb_entropy': 5.7732177455173055, 'lr_1': 1.75, 'lr_2': 0.5, 'reg_metric': 'edge_forward_spline_u', 'steps': 150}
chf_best = {'depth': 1, 'grid': 9, 'k': 2, 'lamb': 5.123066656699474e-06, 'lamb_entropy': 7.716618914050463, 'lr_1': 2, 'lr_2': 1.75, 'reg_metric': 'edge_forward_spline_u', 'steps': 100}
rea_best = {'depth': 1, 'grid': 6, 'k': 8, 'lamb': 3.79496703629217e-05, 'lamb_entropy': 0.006504868427044119, 'lr_1': 2, 'lr_2': 0.5, 'reg_metric': 'edge_forward_sum', 'steps': 25}
xs_best = {'depth': 2, 'grid': 9, 'k': 4, 'lamb': 0.00039029273996368227, 'lamb_entropy': 0.42860645226254324, 'lr_1': 1.25, 'lr_2': 1.25, 'reg_metric': 'edge_forward_spline_u', 'steps': 100}

def run_model(device, dataset, params, run_name, lib=None):
    kan = NKAN(dataset, 42, device, params)
    model = kan.get_model()
    spline_metrics = kan.get_metrics(model, run_name)
    equation = kan.get_equation(model, run_name, simple=0, lib=None)
    importances = kan.get_importances(model, run_name)
    return


if __name__=="__main__":
    if os.path.exists('model'):
        shutil.rmtree("model")
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    datasets_dict = {
        'fp': [get_fp, fp_best],
        # 'bwr': [get_bwr, bwr_best],
        # 'heat': [get_heat, heat_best],
        # 'htgr': [get_htgr, htgr_best],
        # 'mitr_a': [partial(get_mitr, region='A'), mitr_a_best],
        # 'mitr_b': [partial(get_mitr, region='B'), mitr_b_best],
        # 'mitr_c': [partial(get_mitr, region='C'), mitr_c_best],
        # 'mitr': [partial(get_mitr, region='FULL'), mitr_best],
        # 'chf': [get_chf, chf_best],
        # 'rea': [get_rea, rea_best],
        # 'xs': [get_xs, xs_best]
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model, info in datasets_dict.items():
        run_model(
            device=device, 
            dataset=info[0](cuda=True), 
            params=info[1], 
            run_name=f"{model.upper()}_{str(dt.date.today())}")

