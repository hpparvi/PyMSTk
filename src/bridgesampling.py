from __future__ import division

import math as m
import numpy as np
from scipy.stats import norm, uniform
from scipy import linalg

def bs(x, q, q1fun, means=None, sigmas=None, n2=100, niter=5, return_all=False, guess=1):
    
    means = means if means is not None else np.mean(x, axis=0)
    sigmas = sigmas if sigmas is not None else 2*np.std(x, axis=0)
    
    print means, sigmas
    q2fun = MVN(means, sigmas)
    
    x1  = asarray(x)
    x2  = q2fun.rvs(n2)

    q11 = q
    q12 = q1fun(x2)
    q22 = q2fun(x2)
    q21 = q2fun(x1)

    n1 = q11.size
    s1 = n1/float(n1+n2)
    s2 = n2/float(n1+n2)
    
    l1 = q11 / q21
    l2 = q12 / q22
    
    r = ones(niter)*guess
    for i in range(1,niter):
        r[i] = sum(l2/(s1*l2+s2*r[i-1]))/n2 / (sum(1./(s1*l1+s2*r[i-1]))/n1)
    
    return r if return_all else r[-1]


class MVN(object):
    def __init__(self, mu, sigma):
        self.mu = asarray(mu)
        self.sigma = asarray(sigma)
        self.covariance = diagflat(self.sigma**2)
        self.precision = linalg.inv(self.covariance)
        self.det = linalg.det(self.covariance)
        self.nd = self.mu.size
        self.norm = (2*m.pi)**(-0.5*self.nd) / m.sqrt(self.det)
        
    def _pdf(self, x):
        r = x - self.mu
        return self.norm * m.exp(-0.5*(r.dot(self.precision.dot(r.T)) ))

    def pdf(self, x):
        x = asarray(x)
        if x.ndim == 1:
            return self._pdf(x)
        else:
            return array([self._pdf(xx) for xx in x])
    
    def rvs(self, size=1):
        return multivariate_normal(self.mu, self.covariance, size=size)
    
    def __call__(self, x):
        return self.pdf(x)
    
