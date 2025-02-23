# RDN Applicability Domain

This repository contains the Python implementation of the **Reliability Density Neighbourhood (RDN) method** for determining the applicability domain of a QSAR model. The method is described in the paper:  

> **"A novel applicability domain technique for mapping predictive reliability across the chemical space of a QSAR: reliability-density neighbourhood"**  
> by Aniceto et al. based on the work of Sahigara et al. in:  

> **"Defining a novel k-nearest neighbours approach to assess the applicability domain of a QSAR model for reliable predictions"**  

Aniceto et al. have also introduced an R implementation of the method, which can be found in their GitHub repository (see their paper for details). 

As their method is constructed for QSAR classification models based on chosen molecular descriptors, we had to make a couple adjustments to apply RDN to our regression model. First we use the tanimoto distance derived from MACCSKeys fingerprints instead of euclidian distance in the descriptor space. Second we adjusted the calculation of the reliability($W$) described in Aniceto et al. For more detail on that see the function get_reliability() from rdn.py in `our_example/`.

##  Directory Structure

The repository consists of two main directories:  

- **`our_example/`**  
  Contains the exact scripts used to determine the applicability domain of our **water solubility prediction model**.

- **`general_implementation/`**  
  Provides a **general implementation** of the RDN method. 

- The main difference of the above directories is, that in our_example we didn't run a vary time consuming loop for determining the optimal $k$-nearest for the whole $k$-range at once, but instead ran it parallel for different $k$-nearest values seperately. Whereas it is the case in general_implementation.
- You should start with README.md from `our_example/` before moving to `general_implementation/`, as its README.md only describes the differences to `our_example/`.

## How to Use  

Each subdirectory contains its own description (`README.md`) with specific details on its contents and usage.  

