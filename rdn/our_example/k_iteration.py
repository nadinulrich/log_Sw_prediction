import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit 
import pickle
from pathlib import Path 
import rdn

k=1
basis=  str(Path(__file__).parent.absolute())
train_data = basis+ "here your file"

#read data
train = pd.read_excel(train_data)

#calculate reliability
values_exp_train = train["Sexp"]
values_pred_train = train[["Spred1", "Spred2", "Spred3", "Spred4", "Spred5"]]
W = rdn.get_reliability(values_exp_train, values_pred_train)

#tune k
splitter = ShuffleSplit(n_splits=1000, test_size=0.2, random_state=42)
smiles_train = train.iloc[:, 10]
indices_train = np.arange(len(smiles_train))
in_ad_percent_arr_k = []
accuracy_arr_k=[]
for knearest in range(k, k+1):
    in_ad_percent_arr = []
    accuracy_arr = []
    for idx_tr, idx_te in splitter.split(indices_train):
        #determine parameters for get_ad()
        smiles_tr = smiles_train.iloc[idx_tr]
        smiles_te = smiles_train.iloc[idx_te]
        values_exp_te = values_exp_train.values[idx_te]
        values_pred_te = values_pred_train.values.mean(axis=1)[idx_te]
        #calculate the ad
        _, in_ad_percent, accuracy = rdn.get_ad(smiles_tr, smiles_te, knearest, W[idx_tr], values_exp_te, values_pred_te, get_accuracy=True)
        in_ad_percent_arr.append(in_ad_percent)
        accuracy_arr.append(accuracy)
    in_ad_percent_arr_k.append(in_ad_percent_arr)
    accuracy_arr_k.append(accuracy_arr)

plt.style.use("fivethirtyeight")

#plot the in_ad percantage for each k
plt.figure(figsize=(8, 6))
flierprops = dict(marker='o', markerfacecolor="red")
meanprops=dict(markerfacecolor="green", markeredgecolor="green")
medianprops=dict(color="blue")
box = plt.boxplot(in_ad_percent_arr_k, positions=range(k, k+1), showmeans=True, flierprops=flierprops, meanprops=meanprops, medianprops=medianprops)
plt.xticks(range(k, k+1))
plt.xlabel("knearest")
plt.ylabel("Percantage of test in ad")
plt.title("Box-Whisker-Plot")

plt.tight_layout()
plt.savefig(basis+"k_boxplot.pdf")



#save important data
boxplot_data = {
    "whiskers": [whisker.get_ydata().tolist() for whisker in box["whiskers"]],
    "medians": [median.get_ydata()[0] for median in box["medians"]],
    "fliers": [flier.get_ydata().tolist() for flier in box["fliers"]],
    "means": [mean.get_ydata()[0] for mean in box["means"]] if "means" in box else None,
}


with open(basis+"boxplot_data.pkl", "wb") as f:
    pickle.dump(boxplot_data, f)

pd.DataFrame(accuracy_arr_k).to_csv(basis+"k_accuracy.csv", index=False)






    
