# log_Sw_prediction
prediction of water solubility

This git repository contains the Supplementary Information to the publication “Prediction of the water solubility by a graph convolutional-based neural network on a highly curated dataset”:

•	main.py

•	mygraphconvmodel.py

•	dataset.xlxs

We used DeepChem library for model development. The full code for the DNN development is given at the GIT repository of DeepChem https://github.com/deepchem/deepchem. Our adapted code is provided here in main.py. The keras model implementation is given in mygraphconvmodel.py.

The dataset for model development was taken from Sorkun et al. [1]. 

All corrections made during the data curation procedure are included in the dataset. 

The predictions for the E dataset are included as a second tab in dataset.xlsx.


[1] Sorkun MC, Khetan A, Er S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Scientific Data (2019) 6:143. https://doi.org/10.1038/s41597-019-0151-1
