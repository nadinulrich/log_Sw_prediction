import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit 
import pickle
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from scipy.spatial.distance import cdist, pdist, squareform


"""-------------------------------hleper functions------------------------------------"""
def get_fps(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    hmols = [Chem.AddHs(mol) for mol in mols]
    return [rdMolDescriptors.GetMACCSKeysFingerprint(hmol) for hmol in hmols]

def get_fps_mtrx(fps):
    arr = []
    for fp in fps:
        a = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, a)
        arr.append(a)
    return np.array(arr)


def get_dist_mtrx(fps1, fps2=None):
    if isinstance(fps2, np.ndarray):
        return cdist(fps1, fps2, metric="jaccard")
    else:
        dist_mtrx = squareform(pdist(fps1, metric="jaccard"))
        return np.sort(dist_mtrx, axis=1)

def get_reliability(values_exp: pd.Series, values_pred: pd.DataFrame):
    values_exp = values_exp.values 
    values_pred = values_pred.values
    #consensus std
    std = values_pred.std(ddof=0, axis=1)/(np.abs(values_pred.mean(axis=1))+1e-6)
    std[std>1] = 1
    #agreement
    values_pred = values_pred.mean(axis=1)
    rel_errors = np.abs(values_pred - values_exp)/(np.abs(values_exp)+1e-6)
    agreement = 1 - rel_errors
    agreement[agreement<0] = 0
    #reliability
    return (1-std)*agreement
"""-----------------------------------------------------------------------------------"""

def get_ad(smiles_train, smiles_test, knearest, W):
    #distance to nearest k
    fps_train = get_fps(smiles_train)
    fps_mtrx_train = get_fps_mtrx(fps_train)
    dist_mtrx_train = get_dist_mtrx(fps_mtrx_train)
    mean_knn_dist = np.mean(dist_mtrx_train[:, 1:knearest+1], axis=1)
    #reference value
    Q3 = np.percentile(mean_knn_dist, 75)
    Q1 = np.percentile(mean_knn_dist, 25)
    ref_val = Q3 + 1.5*(Q3-Q1)
    #row applicability domain threashholds
    indices = np.apply_along_axis(np.searchsorted, 1, dist_mtrx_train, ref_val, side="right")
    D = np.array([np.mean(row[1:idx]) if idx>1 else np.nan for row, idx in zip(dist_mtrx_train, indices)])
    D[np.isnan(D)] = np.nanmin(D) #we assume D is not n/a only
    #final adjusted applicability domain threashholds
    r = D*W

    #apply the application domain
    fps_test = get_fps(smiles_test)
    fps_mtrx_test = get_fps_mtrx(fps_test)
    dist_mtrx_train_test = get_dist_mtrx(fps_mtrx_train, fps_mtrx_test)
    ad_filt = dist_mtrx_train_test <= r[:, np.newaxis]
    #NNcounts = ad_filt.sum(axis=0)
    #in_ad = NNcounts>0
    in_ad = ad_filt.any(axis=0)

    #performance
    in_ad_counts = in_ad.sum()
    in_ad_percent = 100*in_ad_counts/len(smiles_test)

    return (in_ad, in_ad_percent)


def find_k(smiles_train, values_exp_train, values_pred_train, knearest_arr, **spliter_params):
    #reliability
    W = get_reliability(values_exp_train, values_pred_train)

    #split train and determine applicability domain for each split
    splitter = ShuffleSplit(**spliter_params)
    indices_train = np.arange(len(smiles_train))
    in_ad_percent_arr_k = []
    for knearest in knearest_arr:
        in_ad_percent_arr = []
        for idx_tr, idx_te in splitter.split(indices_train):
            #determine parameters for get_ad()
            smiles_tr = smiles_train.iloc[idx_tr]
            smiles_te = smiles_train.iloc[idx_te]
            #calculate the ad
            _, in_ad_percent = get_ad(smiles_tr, smiles_te, knearest, W[idx_tr])
            in_ad_percent_arr.append(in_ad_percent)
        in_ad_percent_arr_k.append(in_ad_percent_arr)

    #plot the in_ad percentage for each k
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(10, 6))
    flierprops = dict(marker='o', markerfacecolor="red")
    meanprops = dict(markerfacecolor="green", markeredgecolor="green")
    medianprops = dict(color="blue")
    box = plt.boxplot(in_ad_percent_arr_k, positions=knearest_arr, showmeans=True, flierprops=flierprops, meanprops=meanprops, medianprops=medianprops)
    plt.xticks(knearest_arr)
    plt.xlabel("knearest")
    plt.ylabel("Percantage of test in ad")
    plt.title("Box-Whisker-Plot")
    plt.tight_layout()
    plt.savefig("k_boxplot.pdf")

    #save boxplots data
    boxplot_data = {
        "whiskers": [whisker.get_ydata().tolist() for whisker in box["whiskers"]],
        "medians": [median.get_ydata()[0] for median in box["medians"]],
        "fliers": [flier.get_ydata().tolist() for flier in box["fliers"]],
        "means": [mean.get_ydata()[0] for mean in box["means"]] if "means" in box else None,
    }

    with open("boxplot_data.pkl", "wb") as f:
        pickle.dump(boxplot_data, f)

