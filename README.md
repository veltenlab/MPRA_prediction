# MPRA_prediction
Repository containing DL models to predict expression levels from MPRA data

## Installation
There are two environments that are needed for running the github repository. 
The environments can be installed from a .yml file : **`tf_MPRA`** for training the models and **`contribution_scores`** for running the shap values.

`conda env create -f tf_MPRA.yml`

`conda env create -f contribution_scores.yml`


All scripts contain comments or jupyter notebook cells with specifications on which environment to use to compute them.
The environments were created by running the `MPRA_prediction/envs/create_env.sh script`

## Steps to follow
- **`prepareData.R`** :\
Downloads the training data (of library A, B, C and F) from figshare, puts it into the right format and creates train-test splits. 

- **`train_model.ipynb`** :\
CNN+DENSE model with 10-fold cv. Trains each fold 10 times and saves the predictions and weights.

- **`query_model_server.py`** :\
Reads in the weights of the trained model and creates a simple server which waits for sequences as input and returns model predictions as output

- **`query_model_client.R`** :\
Once the model server is running, you can use this function to query it from R.

## Sequence design

See subfolder `sequence_design` for automated design of sequences with cell state specific activity patterns
