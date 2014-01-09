from __future__ import division

import math as m
import numpy as np
from scipy.stats import norm, uniform
from scipy import linalg

__al__ = ['bs']

## TODO: Find out why the density functions must return 0-dimensional arrays!

def bs(x, q, q1fun, q2fun=None, means=None, sigmas=None, n2=100, niter=5, return_all=False, guess=1):
    """

    Parameters
    ----------

    x      : samples from the density function ``q1fun``
    q      : log-density function values corresponding to ``x``
    q1fun  : density function
    q2fun  : density function (optional) 
    means  : multivariate normal means (optional) 
    sigmas : multivariate normal sigmas (optional)
    n2     : number of samples to draw
    niter  : number of iterations
    return_all :
    guess  :

    """

    if not q2fun:
        means = means if means is not None else np.mean(x, axis=0)
        sigmas = sigmas if sigmas is not None else 2*np.std(x, axis=0)
        q2fun = MVN(means, sigmas)
    
    x1s  = np.asarray(x)
    x2s  = q2fun.rvs(n2)

    lq11 = np.asarray(q)
    lq12 = np.array([q1fun(x2) for x2 in x2s])
    lq22 = np.array([q2fun(x2) for x2 in x2s])
    lq21 = np.array([q2fun(x1) for x1 in x1s])

    print lq11.shape
    print lq12.shape

    n1, ln1 = lq11.size, m.log(lq11.size)
    n2, ln2 = n2, m.log(n2)
    ls1 = m.log(n1/float(n1+n2))
    ls2 = m.log(n2/float(n1+n2))
    
    ll1 = lq11 - lq21
    ll2 = lq12 - lq22
    
    lr = np.ones(niter)*guess
    for i in range(1,niter):
        A = ll2 - np.logaddexp(ls1+ll2, ls2+lr[i-1])
        Am = A.max()
        A = np.log(np.sum(np.exp(A-Am))) + Am - np.log(n2)

        B = 0.0 - np.logaddexp(ls1+ll1, ls2+lr[i-1])
        Bm = B.max()
        B = np.log(np.sum(np.exp(B-Bm)))+ Bm - np.log(n1)

        lr[i] = A-B
    
    return lr if return_all else lr[-1]


class MVN(object):
    def __init__(self, mu, sigma):
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        self.covariance = np.diagflat(self.sigma**2)
        self.precision = linalg.inv(self.covariance)
        self.det = linalg.det(self.covariance)
        self.nd = self.mu.size
        self.norm = (2*m.pi)**(-0.5*self.nd) / m.sqrt(self.det)
        
    def rvs(self, size=1):
        return np.random.multivariate_normal(self.mu, self.covariance, size=size)

    def _pdf(self, x):
        r = x - self.mu
        return self.norm * m.exp(-0.5*(r.dot(self.precision.dot(r.T)) ))

    def pdf(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return self._pdf(x)
        else:
            return np.array([self._pdf(xx) for xx in x])
        
    def logpdf(self, x):
        x = np.asarray(x)
        return np.array([np.log(self._pdf(x))])
 

    def __call__(self, x):
        return self.logpdf(x)


class MVU(object):
    """Multivariate uniform distribution

    Parameters
    ----------

    lims : 2D array [ndim,2], where each row contains the min and max limits.
    """
    def __init__(self, lims):
        lims = np.asarray(lims)
        
        self.ndim = lims.shape[0]
        self.means = lims.mean(1)
        self.widths = np.asarray(lims[:,1] - lims[:,0])
        self.pdf = np.prod(1./self.widths)
        self.lpdf = np.array([np.log(self.pdf)])
    
    def rvs(self, size=1):
        return (np.random.uniform(size=(size,self.means.size))-0.5)*self.widths + self.means

    def logpdf(self, x):
        if np.all(np.abs(x-self.means) < 0.5*self.widths):
            return self.lpdf
        else:
            return np.array([-1e18])

    def __call__(self, x):
        return self.logpdf(x)
