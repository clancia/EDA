# EDA
Approximate Computation of Joint Stationary Probability for Exponentially Delayed Arrivals (EDA)

## Overview
The code in this repository generates the figures appearing on [1].

* **plotBound.py** Figure 2 and Table 1 
* **condno_rhoq.py** Figure 3
* **condno_alpha.py** Figure 4
* **mtxspy.py** Figure 5

## Dependencies
1. numpy 1.9.1
2. scipy 0.15.1
3. matplotlib 1.4.2
4. mpmath 0.19

## Caveat
The parameter **ALPHA** is set to **100** in many of the scripts. This value guarantees that the a priori error on the approximate stationary distribution is small for **rho** and **q** smaller than **0.98**.

The computational time for **ALPHA = 100** (and especially the amount of memory required) is quite large, so increasing **ALPHA** is not recommended.

## References
[1] C. Lancia, G. Guadagni, S. Ndreca, and B. Scoppola. Advances on the Late Arrivals Problem. *arXiv preprint [1302.1999](http://arxiv.org/abs/1302.1999)*, 2015.