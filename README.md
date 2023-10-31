# MPRA_prediction
Repository containing DL models to predict expression levels from MPRA data

## Regression models
The regression folder contains scripts and notebooks to predict MPRA activity from sequence. 
Within this folder there are two sets of models:

### Regression from whole sequences
- cv_simple_regression.ipynb : Computes a CNN+DENSE model with 10-fold cross validation. Saves the model weights for each of the folds and prediction of each fold
- cv_simple_regression_ensemble.ipynb : Computes a CNN+DENSE model with 10-fold cross validation. Saves the model weights for each of the folds and prediction of each fold
- contribution_scores.ipynb : Computes contribution scores
- evaluate_regression : compares a random forest predictor vs the cv_simple_model_regression_ensemble.ipynb predictions
- 
### Regression from background sequences
The scripts are very similar for background sequences, the only difference is the encoding of the data and the addition of a masking layer that masks the positions where there are motifs.
- cv_simple_regression_background : Computes a CNN+DENSE model with 10-fold cross validation. Saves the model weights for each of the folds and prediction of each fold
- cv_simple_regression_ensemble_background.ipynb : Computes a CNN+DENSE model with 10-fold cross validation. Saves the model weights for each of the folds and prediction of each fold
- contribution_scores_background.ipynb : Computes contribution scores
- evaluate_regression_background : compares a random forest predictor vs the cv_simple_model_regression_ensemble_background.ipynb predictions


