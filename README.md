# MPRA_prediction
Repository containing DL models to predict expression levels from MPRA data
## Installation
There are two environments that are needed for running the github repository. 
The environments can be installed from a .yml file : **`tf_MPRA`** for training the models and **`contribution_scores`** for running the shap values.

`conda env create -f tf_MPRA.yml`

`conda env create -f contribution_scores.yml`


All scripts contain comments or jupyter notebook cells with specifications on which environment to use to compute them.
The environments were created by running the `MPRA_prediction/envs/create_env.sh script`

## Regression models
The regression folder contains scripts and notebooks to predict MPRA activity from sequence. 
Within this folder there are two sets of models:

### Regression from whole sequences
- **`cv_simple_regression.ipynb`** :\
CNN+DENSE model with 10-fold cv. Trains and saves the model for each fold.

- **`cv_simple_regression_ensemble.ipynb`** :\
CNN+DENSE model with 10-fold cv. Trains each fold 10 times and saves the predictions.

- **`contribution_scores.ipynb`** :\
Computes contribution scores

- **`evaluate_regression.ipynb`** :\
Compares a random forest predictor vs the cv_simple_model_regression_ensemble.ipynb predictions
 
### Regression from background sequences
The scripts are very similar for background sequences, the only difference is the encoding of the data and the addition of a masking layer that masks the positions where there are motifs.

- **`cv_simple_regression_background`** :\
CNN+DENSE model with 10-fold cv. Trains and saves the model for each fold.

- **`cv_simple_regression_ensemble_background.ipynb`** :\
CNN+DENSE model with 10-fold cvn. Trains each fold 10 times and saves the predictions.

- **`contribution_scores_background.ipynb`** :\
Computes contribution scores for background samples.

- **`evaluate_regression_background.ipynb`** :\
Compares a random forest predictor vs the cv_simple_model_regression_ensemble_background.ipynb predictions


