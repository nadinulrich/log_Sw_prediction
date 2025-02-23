import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from scipy.spatial.distance import cdist, pdist, squareform
"""-------------------------------helper functions------------------------------------"""
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

def get_ad(smiles_train, smiles_test, knearest, W, values_exp_te:np.ndarray =None, values_pred_te:np.ndarray=None, get_accuracy=False): #values_pred_te should already be a numpy array of predicted single values not the array of arrays of ensemble predictions
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

    #apply the applicability domain
    fps_test = get_fps(smiles_test)
    fps_mtrx_test = get_fps_mtrx(fps_test)
    dist_mtrx_train_test = get_dist_mtrx(fps_mtrx_train, fps_mtrx_test)
    ad_filt = dist_mtrx_train_test <= r[:, np.newaxis]
    in_ad = ad_filt.any(axis=0)

    #performance
    in_ad_counts = in_ad.sum()
    in_ad_percent = 100*in_ad_counts/len(smiles_test)

    if get_accuracy:
        correct_in_ad = (abs(values_exp_te[in_ad] - values_pred_te[in_ad]) <= 0.55).sum()
        correct_out_ad = (abs(values_exp_te[~in_ad] - values_pred_te[~in_ad]) >= 0.55).sum()
        accuracy = (correct_in_ad + correct_out_ad)/len(values_exp_te)
        return (in_ad, in_ad_percent, accuracy)
    else:
        return (in_ad, in_ad_percent)

