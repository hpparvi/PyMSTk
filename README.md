PyMSTk
======

Python Model Selection Toolkit (spelled PyMiSTake)

A small Python toolkit for the estimation of global likelihoods (Bayesian evidence).

Currently implemented routines

 - Bridge sampling against a proxy distribution
   - Estimates the absolute global likelihood given 
     - a set of posterior sample locations
     - a set of posterior samples corresponding to the locations
     - posterior distribution function
     - proxy distribution parameters
 
Routines to be implemented

 - Nested sampling
 - Bayesian quadrature
 - Truncated posterior mixture estimate

Trivial routines that may be implemeted

 - Basic MC integration
 - Importance sampling MC integration
 
Authors
-------

 - Hannu Parviainen <hpparvi@gmail.com>
