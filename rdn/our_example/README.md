# The exact Implementation of RDN AD (Aniceto et al., Sahigara et al.) we used for our model

This directory contains the implementation of the **Reliability Density Neighbourhood (RDN) method** from **Aniceto et al.** to determine the applicability domain (AD) of a **water solubility prediction model**. The approach is **partially based on Sahigara et al.**, with modifications for our specific regression use case.

## Directory Contents

### **Python Scripts**

#### **1 `rdn.py`**

- Contains the core functions for determining the **RDN applicability domain (AD)**.

##### **Functions in `rdn.py`**

- **Helper functions**

  - Compute molecular **fingerprints**
  - Construct **fingerprint matrices**
  - Calculate **Tanimoto distance matrices**

- **`get_reliability()`**

  - Computes the **reliability correction factor (W)** for each training compound.
  - Inspired by Aniceto et al., but adapted for **regression models** (instead of classification).
  - **Reliability calculation:**  
    $W = (1 - STD_{rel}) ⋅ (1 - Δx_{rel})$
    - **$STD_{rel}$** = relative standard deviation of predictions
    - **$Δx_{rel}$** = relative deviation of predictions from experimental values
    - If **$STD_{rel}$ or $Δx_{rel}$ > 1**, reliability is set to **0** (indicating the lowest confidence).

- **`get_ad()`**
  - Follows the scheme from **Sahigara and Aniceto** to calculate local **AD radii** based on k-nearest neighbors.
  - Uses **reliability (W) to adjust the AD radii**.
  - Checks whether each test compound is within at least one **training AD radius**.
  - **Returns:**
    - AD filter for test compounds
    - Percentage of test compounds **within the AD**
    - The accuracy of the applicability domain calculated customly(see the code after "if accuracy_on:") as the proportion of test molecules that are correctly inside the applicability domain (AD) and correctly outside the AD.

#### **2 `k_iteration.py`**

- Serves to select the **optimal number of k-nearest neighbors** for AD calculation:
  - Splits the training set **1000 times (80/20 train/test)**
  - Calculates the **percentage of test' compounds within AD** for different **k-values**
  - Saves the results for **boxplot analysis** (analogous to Sahigara et al.)
  - Was run separately for defferent $k$-nearest values ranging from 1 to 60.

#### **3 `k_evaluation.py`**

- Reads the boxplot data from `k_iteration.py` and creates two **boxplots**:
  - One with raw values
  - One with **rounded integer values** (for easier selection of optimal k-nearest)
- Reads the accuracy data from `k_iteration.py` and plots the mean accuracy over $k$-nearest.

#### **4 `ad_calculation.py`**

- Computes the **applicability domain for k = 12** (optimal k).
- Outputs:
  - AD statistics
  - Marks test compounds as **inside or outside the AD**

---

## Notes

- The **RDN method** was originally developed for **classification models** (Aniceto et al., Sahigara et al.), but we adapted it for **regression models**.
- The **relative standard deviation** and **relative deviation from experimental values** are used as key measures of prediction reliability.
- The distance metric used is tanimoto distance and not the euclidian distance like in Aniceto et al. and Sahigara et al.

For more details, refer to the main [`README.md`](../README.md) of the repository.
