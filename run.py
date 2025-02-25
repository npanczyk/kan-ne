from main import *
from preprocessing import *
import shutil
import os

fp_best = {'depth': 1, 'grid': 4, 'k': 8, 'lamb': 1.4831135449453957e-07, 'lamb_entropy': 1.8089227877236496, 'lr_1': 1.5, 'lr_2': 1.25, 'steps': 125}
bwr_best = {'depth': 1, 'grid': 10, 'k': 2, 'lamb': 3.0698178578908114e-05, 'lamb_entropy': 0.886893553328925, 'lr_1': 1.5, 'lr_2': 1, 'steps': 225}
heat_best = {'depth': 1, 'grid': 5, 'k': 2, 'lamb': 0.00013095361343762514, 'lamb_entropy': 4.352418097964702, 'lr_1': 1.75, 'lr_2': 1.25, 'steps': 150}
htgr_best = {'depth': 3, 'grid': 5, 'k': 4, 'lamb': 2.63584570163326e-06, 'lamb_entropy': 3.02497466094173, 'lr_1': 0.75, 'lr_2': 1.5, 'steps': 75}
mitr_a_best = {'depth': 2, 'grid': 5, 'k': 3, 'lamb': 6.901006219885579e-06, 'lamb_entropy': 2.677506734155999, 'lr_1': 2, 'lr_2': 1.25, 'steps': 150}
mitr_b_best = {'depth': 1, 'grid': 6, 'k': 2, 'lamb': 1.8321755739060752e-06, 'lamb_entropy': 2.6597667019837576, 'lr_1': 2, 'lr_2': 0.5, 'steps': 75}
mitr_c_best = None
mitr_best = None
chf_best = {'depth': 3, 'grid': 8, 'k': 8, 'lamb': 3.3619635794339965e-07, 'lamb_entropy': 2.078997799175118, 'lr_1': 1.25, 'lr_2': 1.25, 'steps': 125}
rea_best = {'depth': 2, 'grid': 5, 'k': 3, 'lamb': 3.844422322524488e-07, 'lamb_entropy': 3.4113460498010846, 'lr_1': 0.5, 'lr_2': 2, 'steps': 250}
xs_best = {'depth': 1, 'grid': 5, 'k': 6, 'lamb': 6.618155426602294e-06, 'lamb_entropy': 2.8227427065240978, 'lr_1': 0.75, 'lr_2': 2, 'steps': 225}

def run_model(device, dataset, params, run_name):
    kan = NKAN(dataset, 42, device, params)
    model = kan.get_model()
    spline_metrics = kan.get_metrics(model, run_name)
    lib0 = ['x','x^2','x^3','x^4', '1/x', 'exp','log','sqrt','sin', 'cos','tan','abs', 'sgn']
    lib1 = ['x', 'x^2', 'x^3', 'x^4', 'x^5', '1/x', '1/x^2', '1/x^3', '1/x^4', '1/x^5', 'sqrt', 'x^0.5', 'x^1.5', '1/sqrt(x)', '1/x^0.5', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'sgn']
    lib2 = ['x', 'x^2', 'x^3', 'x^4', 'x^5', '1/x', '1/x^2', '1/x^3', '1/x^4', '1/x^5', 'sqrt', 'x^0.5', 'x^1.5', '1/sqrt(x)', '1/x^0.5', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sgn', 'arcsin', 'arccos', 'arctan', 'arctanh', '0', 'gaussian']
    libs = [lib0, lib1, lib2]
    for i, lib in enumerate(libs):
        equation = kan.get_equation(model, f'{run_name}_lib{i}', lib, metrics=True)
    importances = kan.get_importances(model, run_name)
    shutil.rmtree("model")
    return

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_model(device, get_fp(cuda=True), fp_best, 'FP_250224')
    run_model(device, get_bwr(cuda=True), bwr_best, 'BWR_250224')
    run_model(device, get_heat(cuda=True), heat_best, 'HEAT_250224')
    run_model(device, get_htgr(cuda=True), htgr_best, 'HTGR_250224')
    run_model(device, get_chf(cuda=True), chf_best, 'CHF_250224')
    run_model(device, get_rea(cuda=True), rea_best, 'REA_250224')
    run_model(device, get_xs(cuda=True), xs_best, 'XS_250224')
