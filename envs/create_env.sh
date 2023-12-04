
conda create --name tf_MPRA python=3.9.7
conda activate tf_MPRA
pip install tensorflow[and-cuda]
conda install -c anaconda ipykernel 
conda install -c anaconda pandas
conda install -c anaconda numpy
conda install -c anaconda scikit-learn 
conda install -c anaconda matplotlib


conda create --name contribution_scores python=3.7
conda activate contribution_scores
conda install -c anaconda tensorflow-gpu=1.14.0
conda install -c anaconda keras-gpu=2.2.4
conda install shap=0.29.3
conda install h5py=2.10.0
pip install ipykernel
conda install pandas
conda install matplotlib
