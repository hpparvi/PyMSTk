PyMSTk
======

Python Model Selection Toolkit (spelled PyMiSTake)

A small Python toolkit for the estimation of global likelihoods (Bayesian evidence).

Currently implemented routines

 - Bridge sampling with a proxy distribution
   - Estimates the absolute global likelihood given 
     - a set of posterior sample locations
     - a set of posterior samples corresponding to the locations
     - posterior distribution function
     - proxy distribution (optional)
 
Routines to be implemented

 - Nested sampling
 - Truncated posterior mixture estimate

Trivial routines that may be implemeted

 - Basic MC integration
 - Importance sampling MC integration
 
Nontrivial routines that may (or may not) be implemented

 - Bayesian quadrature

Authors
-------

 - Hannu Parviainen <hpparvi@gmail.com>
