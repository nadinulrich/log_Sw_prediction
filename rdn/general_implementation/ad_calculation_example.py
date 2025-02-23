import pandas as pd
import rdn
import numpy as np
from pathlib import Path 


basis_dir =  str(Path(__file__).parent.absolute())
#read training and test data sets
train_data = basis_dir+"here your file"
test_data = basis_dir+"here your file"
train = pd.read_excel(train_data)
test = pd.read_excel(test_data)

#extract all needed parameters for rdn functions
knearest = 12

smiles_train = train.iloc[:, 10]
values_exp_train = train["Sexp"]
values_pred_train = train[["Spred1", "Spred2", "Spred3", "Spred4", "Spred5"]]

smiles_test = test["SMILES"]

#apply rdn applicability domain
W = rdn.get_reliability(values_exp_train, values_pred_train)
in_ad, in_ad_percent = rdn.get_ad(smiles_train, smiles_test, knearest, W)

#create a new column for test with ad info
status_col = np.where(in_ad, "in", "out")
test["status"] = status_col
test.to_csv("ad_test.csv", index=False)

#rmse in/out
values_exp_test = test["SExp"].values 
values_pred_test = test[["Split1", "Split2", "Split3", "Split4", "Split5"]].values.mean(axis=1)

abs_err = np.abs(values_exp_test-values_pred_test)

rmse_in = np.sqrt(np.mean(abs_err[in_ad]**2))
rmse_out = np.sqrt(np.mean(abs_err[~in_ad]**2))

#see additional information
print(f"{in_ad.sum()}/{len(smiles_test)} molecules from test set are in the applicability domain. That is {in_ad_percent:.2f}%.")