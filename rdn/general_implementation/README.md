# General Implementation of the RDN AD Method from Aniceto et al.

This directory contains a general implementation of the **Reliability Density Neighbourhood (RDN) method** for determining the applicability domain (AD), as described in **Aniceto et al.** (_A novel applicability domain technique
for mapping predictive reliability across the
chemical space of a QSAR: reliability‑density
neighbourhood_) based on Sahigara et al.(_Defining a novel k-nearest neighbours approach
to assess the applicability domain of a QSAR
model for reliable predictions_)

## Directory Contents

### Python Scripts

#### `rdn.py`

- Similar to `rdn.py` in `our_example/`, but with some differences:
  - Does not include accuracy calculation
  - Includes an additional function: `find_k()`, which automates the process of finding the optimal k-nearest neighbors
  - `find_k()`
    - Runs the k loop from `k_optimize.py` (from `our_example/`) directly on the whole $k$ range which is to be examined.
    - Generates and saves a boxplot with the results

#### `k_optimization_example.py`

- Uses the `find_k()` function to determine the optimal k-nearest for the example dataset.

#### `ad_calculation_example.py`

- Applies `get_ad()` to the example dataset to help to select the optimal $k$-nearest value for the applicability domain (AD).

## Notes

- This directory provides a generalized version of the RDN AD method, without the additional accuracy assessment included in `our_example/`.

For more details, refer to the main [`README.md`](../README.md).
