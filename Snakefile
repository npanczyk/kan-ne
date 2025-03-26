import datetime as dt
model = 'MITR_FULL'
run_name = f"{model.upper()}_{str(dt.date.today())}"
# run_name = f"{model.upper()}_2025-03-03"
path = f"hyperparameters/{run_name}/"

rule targets:
    input: 
        fig_file = "figures.log",
        miter_full = f'{path}{run_name}_params.txt'



rule collect_figures:
    output: "figures.log"
    shell:
        "touch figures.log"

# hyperparams: params, r2, pruned per dataset
# kan: spline results, sym results, eq.txt, eq.tex
# fnn: results
# fnn xai: shap-values, png
# kan xai: shap-values, png

rule hyperparams:
    output: 
        miter_full = f'{path}{run_name}_params.txt'
    script:
        "hypertuning.py"
