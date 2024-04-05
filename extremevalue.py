import numpy as np
import torch
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from scipy.optimize import minimize
from math import log,floor
import pandas as pd
import os
import matplotlib.pyplot as plt
import tqdm

def _rootsFinder(fun,jac,bounds,npoints,method, eps=1e-8):
        """
        Find possible roots of a scalar function
        
        Parameters
        ----------
        fun : function
		    scalar function 
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval
        
        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1]-bounds[0])/(npoints+1)
            if step <= eps:
                X0 = np.array([0.0])
            else:
                X0 = np.arange(bounds[0]+step,bounds[1],step)
            # except:
            #     print(peaks)
            #     print('Ym:',Ym)
            #     print('YM:',YM)
            #     print('Ymean:',Ymean)
            #     print('a:',a)
            #     print('b:',b)
            #     print('c:',c)
            #     import sys
            #     sys.exit(0)

        elif method == 'random':
            X0 = np.random.uniform(bounds[0],bounds[1],npoints)
        
        def objFun(X,f,jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g+fx**2
                j[i] = 2*fx*jac(x)
                i = i+1
            return g,j
        
        opt = minimize(lambda X:objFun(X,fun,jac), X0, 
                       method='L-BFGS-B', 
                       jac=True, bounds=[bounds]*len(X0))
        
        X = opt.x
        np.round(X,decimals = 5)
        return np.unique(X)

def _log_likelihood(Y,gamma,sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
        
        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   
        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma/sigma
            L = -n * log(sigma) - ( 1 + (1/gamma) ) * ( np.log(1+tau*Y) ).sum()
        else:
            L = n * ( 1 + log(Y.mean()) )
        return L

def grimshaw(peaks, epsilon = 1e-8, n_points = 10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        
        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """
        def u(s):
            return 1 + np.log(s).mean()
            
        def v(s):
            return np.mean(1/s)
        
        def w(Y,t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            return us*vs-1
        
        def jac_w(Y,t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            jac_us = (1/t)*(1-vs)
            jac_vs = (1/t)*(-vs+np.mean(1/s**2))
            return us*jac_vs+vs*jac_us
            
    
        Ym = peaks.min()
        YM = peaks.max()
        Ymean = peaks.mean()
        
        
        a = -1/YM
        if abs(a)<2*epsilon:
            epsilon = abs(a)/n_points
        
        a = a + epsilon
        b = 2*(Ymean-Ym)/(Ymean*Ym)
        c = 2*(Ymean-Ym)/(Ym**2)

        # We look for possible roots
        left_zeros = _rootsFinder(lambda t: w(peaks,t),
                                 lambda t: jac_w(peaks,t),
                                 (a+epsilon,-epsilon),
                                 n_points,'regular')
        
        right_zeros = _rootsFinder(lambda t: w(peaks,t),
                                  lambda t: jac_w(peaks,t),
                                  (b,c),
                                  n_points,'regular')
    
        # all the possible roots
        zeros = np.concatenate((left_zeros,right_zeros))
        
        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = _log_likelihood(peaks,gamma_best,sigma_best)
        
        # we look for better candidates
        for z in zeros:
            gamma = u(1+z*peaks)-1
            sigma = gamma/z
            ll = _log_likelihood(peaks,gamma,sigma)
            if ll>ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
    
        return gamma_best,sigma_best,ll_best

def percentage(t, Zq, gamma, sigma, n, Nt):
    '''
    t : numpy.array
        initial threshold computed during the calibration step
    
    n : int
        number of observed values
    
    Nt : numpy.array
        number of observed peaks
    '''
    
    return (Nt / n) * np.where(gamma != 0, np.power(1 / (1 + (gamma / sigma)*(Zq - t)), 1/gamma), np.exp(-(Zq - t)/sigma))


def initialize(init_data, level = 0.90, verbose = True):
        """
        Run the calibration (initialization) step
        
        Parameters
	    ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t 
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        """
        level = level-np.floor(level)
        
        n_init = init_data.shape[0]
        
        S = np.sort(init_data, axis=0)     # we sort X to get the empirical quantile
        init_threshold = S[int(level*n_init), :] # t is fixed for the whole algorithm

        # initial peaks
        peaks = []
        Nt = []


        for i in range(init_data.shape[1]):
            peaks.append((init_data[:, i])[init_data[:, i] > init_threshold[i]] - init_threshold[i])
            Nt.append(peaks[i].size)

        Nt = np.array(Nt)
        n = n_init
        
        if verbose:
            print('Initial threshold shape: %s' % init_threshold.shape)
            print('number of 10 peaks : %s' % Nt[:10])
            print('Grimshaw maximum log-likelihood estimation ... ', end = '')
            
        gammas, sigmas, lls = [], [], []    
        for i in range(init_data.shape[1]):
            if peaks[i].size == 0:
                g, s, l = 1e9, 1e9, 1e9
            else:
                g,s,l = grimshaw(peaks[i])

            assert not np.isnan(g) and not np.isnan(s)

            gammas.append(g)
            sigmas.append(s)
            lls.append(l)
        
        gammas = np.array(gammas)
        sigmas = np.array(sigmas)
        lls = np.array(lls)

        return gammas, sigmas, lls, peaks, Nt, init_threshold
