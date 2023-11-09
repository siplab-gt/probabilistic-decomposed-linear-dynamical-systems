import numpy as np
import torch

def obs_ll(y, x, C, S):
    
    T, M = y.shape
    X = (y - (C @ (x.T)).T)[:,None,:]

    log_like_i = (torch.linalg.solve(S, X.squeeze().T).T[:,None,:]  @ X.transpose(2,1)).squeeze()
    log_like = log_like_i
    log_like += T*torch.logdet(S) + M * torch.log(torch.tensor(2*torch.pi))
    log_like *= -1/2

    return log_like.sum()


def dyn_ll(x0, x1, F, S):
    logdet = torch.logdet(S)
    T, N = x0.shape
    X = (x1 - (F @ x0[:,:,None]).squeeze())[:,None,:]
    log_like_i = ((torch.linalg.solve(S, X.squeeze().T).T)[:,None,:]  @ X.transpose(2,1)).squeeze()
    log_like = log_like_i
    log_like += T*logdet + N * torch.log(torch.tensor(2*torch.pi))
    log_like *= -1/2
    return log_like.sum()





    
