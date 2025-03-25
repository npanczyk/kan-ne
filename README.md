# kan-ne
This repository serves to reproduce the results found in *{citation here}*. 
## Paper Citation
TBD

## Getting Started
### Requirements
- python

All necessary packages included in `environment.yml`

### Environment Setup and Activation

```bash
conda env create -f environment.yml

conda activate pykan-env
```

## How to Generate Results
### KAN   
KAN results can be reproduced by running the following three scripts, in order: `hypertuning.py`, `run.py`, `explainability.py`. 

* **Hyperparameter Tuning** 

    To start, run:
    ```bash
    python hypertuning.py
    ```
    This will create a `\hyperparameters` directory with a subdirectory for each model. Each subdirectory will have a `params.txt`, a `pruned.txt`, and an `R2.txt` file. These files contain the hyperparameter sets, whether the model successfully pruned or not, and the symbolic and spline R2 scores for each trial. To find the top hyperparameter results, you will need to modify the `hyperparams_dict` dictionary to correspond to your local best hyperparameter file paths in `hypersort.py`. Then, to generate a table with the results, run: 
    ```bash
    python hypersort.py 
    ```

* **Model Fitting, Metrics, and Equations**   

    Once you've generated your best hyperparameters, modify the dictionaries of the form `model_best` to contain the best hyperparameters for each model in the script `run.py`. Then to fit each model, generate spline and symbolic metrics, and generate equations, run: 
    ```bash 
    python run.py
    ```
    The metrics will be stored in `\results` and the equations will be stored in `\equations`, named after their respective models. 

* **Explainability Analysis**   

    After generating your equations, you can start the explainability analysis. First, open `explainability.py` and under the main function, modify the `datasets_dict` to correspond to your equation filepaths. Then, you will need to generate the shap values and save them for plotting later. To do this, run:
    ```bash
    python explainability.py
    ```
    Then, using the path dictionary printed to the command line, modify the `shap_path_dict` to match your local filepaths to the saved shap values (these are .pkl objects). Next, comment out the `paths_dict = ` line and uncomment the for loop and the `plot_shap()` function call lines. Re-run:
    ```bash
    python explainability.py
    ```
    Optionally, to save your generated shap values to a csv file and to print them to a latex table, uncomment and run the last for loop in the file.

### FNN   

To reproduce the FNN results, you will run 

```bash
python fnn.py
```
three times, uncommenting a unique step in the main function each time. In Step 1, you will generate a model path dictionary. Modify `model_path_dict` after running Step 1 to correspond to your local file path before running Step 2. In Step 2, you will generate a shap path dictionary. Modify `shap_path_dict` after running Step 2 to correspond to your local file path before running Step 3. In Step 3, you will plot the shap values generated in Step 2. Optionally, if you would like to print the shap values and save them to separate csv files, you can run Step 4 simulataneously as Step 3 by uncommenting the associated lines. Otherwise, each step should be run in isolation (all other steps commented out). 
## License

[MIT](https://choosealicense.com/licenses/mit/)