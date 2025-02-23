import pandas as pd
from pathlib import Path 
import rdn

k=1
basis=  str(Path(__file__).parent.absolute())
train_data = basis+ "here your file"

#read data
train = pd.read_excel(train_data)


#extract all needed parameters for rdn functions
knearest_arr = list(range(1,21))+[22,25,27,30,35,40,45,50,55,60]

smiles_train = train.iloc[:, 10]
values_exp_train = train["Sexp"]
values_pred_train = train[["Spred1", "Spred2", "Spred3", "Spred4", "Spred5"]]

#apply find_k()
rdn.find_k(smiles_train, values_exp_train, values_pred_train, knearest_arr, n_splits=1000, test_size=0.2, random_state=42)