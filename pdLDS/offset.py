from scipy.spatial import cKDTree
from math import floor
import torch
import numpy as np

# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
# from botorch.optim.optimize import optimize_acqf

from skopt import gp_minimize

    
    
        
from skopt import gp_minimize
from scipy.spatial import cKDTree
from math import floor
import torch
import numpy as np


import warnings

def tricube_weight(d):
    return (1-d**3)**3


class LoessInference:
    def __init__(self,  T, x = None, best=None, init=True):
        # self.s_bounds = s_bounds
        self.T = T
        self.t = torch.arange(self.T)
        self.kd = cKDTree(self.t[:,None])
        
#         self.n_initial_points = n_inital_points
    
        if init:
            assert x is not None, "Need to give x if init"
#             self.T = len(x)
# Ω
            self.x = x
#             self.init_knee()
        
            self.best = best
            self.bestx = self.best
#         if self.best is not None:
#             self.besty = self.knee(self.bestx)
            
#         else:
#             self.bestx = None
#             self.besty = None
            
            
        
    def Loess(self, 
                y, 
                S = 2, 
                # s = 0.33, 
                weight = tricube_weight):
        '''
        First order local linear regression, but a bit faster

        Parameters:
        -----------
        x:
        y:
        s: float (0,1]
            smoothing parameter

        Returns: 
        --------
        values, (M,B)
        '''


        T , d = y.shape
        # k = min(floor(T*s), T-1)
        k = S
#         print("k", k)


        di, ix = self.kd.query(self.t[:,None], np.arange(1,k+2))
        ix = torch.tensor(ix)
        di = torch.tensor(di)
#         print(ix.shape)

        
#         d = 10
        di = di
        sd = di/di.max(1).values[:,None]
        w = torch.sqrt(weight(sd))

        X_b = torch.ones([T, k+1,2]) #temp var
        Y = torch.ones([T, k+1,d])
        for i, (k_ix,w_) in enumerate(zip(ix,w)):
            X_b[i,:,0] = self.t[k_ix]
            Y[i,:,:] =  y[k_ix]

        # print("OFFSET COMPUTING", w_.shape)

        # speedup due to batch operation
        lstsq = torch.linalg.lstsq((X_b* w_[None,:,None]), (Y* w_[None,:,None]))
        M = lstsq.solution[:,0,:]
        B = lstsq.solution[:,1,:]
        values = M * self.t[:,None] + B
        return values, (M,B)

    def loess_detrend(self, 
        y, 
        # s=0.1
        S = 3
        ):
        trend, (m, b) = self.Loess( y, S = S)
        lds = y - trend
        return trend, lds
    
    def skknee(self, s):
        '''
        Objective to optimize
        
        '''
        s = s[0]
        loss = self.compute_loess_score(self.x, s=s)
#         print(loss)
        score = loss - (self.Sm * s + self.Sb)
        return -score.item()
    
    def knee(self, s):
        '''
        Objective to optimize
        
        '''
        s = s[0]
        loss = self.compute_loess_score(self.x, s=s)
#         print(loss)
        score = loss - (self.Sm * s + self.Sb)
        return score
    
    def compute_loess_score(self, y, s=0.1):
        '''
        Compute Loess. 
        '''
        trend, lds = self.loess_detrend(y, s=s)
        a,_,_,_ = torch.linalg.lstsq(lds[:-1], lds[1:])
        a_loss = torch.mean((lds[1:] - lds[:-1] @ a)**2)
        return a_loss

        
    

    def estimate_smoothing_sk(self,  n_calls=21, n_initial_points=5):
        
        res = gp_minimize(self.skknee,                  # the function to minimize
                  [self.s_bounds],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=n_calls,         # the number of evaluations of f
                  n_random_starts=n_initial_points,  # the number of random initialization points
                  model_queue_size=1, 
                x0 =self.bestx)   # the random seed
    
        return res
    
    
    
