import torch 
import torch.nn as nn

def kalman_filter(y, F, C, x0, Sy, Sx, S0):
    '''
    Numpy Kalman filter
    y: 
    F: 
    C: 
    x0: inital point (N,)
    S0: initial point noise (N,N)
    Sy: measurement noise (T,M,M) or (M,M)
    Sx: process noise (T,N,N) or (M,M)
    
    '''
    T = len(y)
    N = F.shape[-1]
    

    # init filter variables
    x_pred = torch.zeros([T,N])
    p_pred = torch.zeros([T,N,N])

    x_filt = torch.zeros([T,N])
    p_filt = torch.zeros([T,N,N])


    x_filt[0] = x_pred[0] = x0
    
    # TODO: initializing to infinite covariance can be avoided via Information Filter    
    p_filt[0] = p_pred[0] = S0  

    # Kalman forward
    for t in range(1,T):
        # predict
        A = F[t-1]
        x_pred[t] = A @ x_filt[t-1]
        p_pred[t] = A @ p_filt[t-1] @ A.T + Sx

        # update
        r = y[t] - C @ x_pred[t]
        K = torch.linalg.solve( C @ p_pred[t] @ C.T + Sy, (C @ p_pred[t])).T

        x_filt[t] = x_pred[t] + K @ r
        # joseph form for numerical stability
        p_filt[t] = (torch.eye(N) - K @ C ) @ p_pred[t] @ (torch.eye(N) - K @ C ).T + K @ Sy @ K.T         
        
    return x_filt, x_pred, p_filt, p_pred, K


def rts_smoother(x_filt, x_pred, p_filt, p_pred, F, KT, C):
    '''
    x_filt: (T,N)
    x_pred: (T,N)
    p_filt: (T,N,N)
    p_pred: (T,N,N)
    F:      (T,N,N)
    '''
    T = len(x_filt)
    N = F.shape[-1]

    x_smooth = torch.zeros([T,N])
    p_smooth = torch.zeros([T,N,N])
    p2_smooth = torch.zeros([T,N,N])

    x_smooth[-1] = x_filt[-1]
    p_smooth[-1] = p_filt[-1]

    # pairwise covariance
    p2_smooth[-1] = (torch.eye(N) - KT @ C) @ F[-1] @ p_filt[-2]


    Gts = torch.zeros([T-1,N,N])

    for t in range(T-1,0, -1):
        # TODO: could make smoother more efficient by using F @ p_filt.T from filtering distribution
        Gt = torch.linalg.solve(p_pred[t], (F[t-1] @ p_filt[t-1].T)).T
        Gts[t-1] = Gt
        x_smooth[t-1] = x_filt[t-1] + Gt @ (x_smooth[t] -  x_pred[t])
        S = (torch.eye(N) - Gt @ F[t-1])
        p_smooth[t-1] = Gt @ p_smooth[t] @ Gt.T + S @ p_filt[t-1] @ S.T

    for t in range(T-1,1, -1):
        p2_smooth[t-1] = Gts[t-2] @ p_smooth[t]
    p2_smooth = p2_smooth[1:] 

    return x_smooth, p_smooth, p2_smooth


# def backward_sample(x_filt, x_pred, p_filt, p_pred, F):
#     T, N = x_filt.shape 
#     xs = torch.zeros((T, N))
#     p_smooth = torch.zeros([T,N,N])
#     p_smooth[-1] = p_filt[-1]

#     noise = torch.randn(T, N)
#     xs[-1] = sample_gaussian(x_filt[-1], p_filt[-1], noise[-1])
#     for t in range(T-1,0, -1):
#         # TODO: could make smoother more efficient by using F @ p_filt.T from filtering distribution
#         Gt = torch.linalg.solve(p_pred[t] ,(p_filt[t-1] @ F[t-1].T).T).T 
#         m = x_filt[t-1] + Gt @ (xs[t] - x_pred[t]) # condition on sample
#         S = (torch.eye(N) - Gt @ F[t-1])
#         p_smooth[t-1] = Gt @ p_smooth[t] @ Gt.T + S @ p_filt[t-1] @ S.T

#         xs[t-1] = sample_gaussian(m, p_smooth[t-1], noise[t-1])

#     return xs

def sample_gaussian(m, S, n):
    L = torch.linalg.cholesky(S)
    return m + L @ n


def sma(x, w):
    '''
    Simple moving average with same edge padding
    
    Parameters
    ----------
    x: (T, N)
    w: window size
    
    Returns
    -------
    s: smoothed signal
    
    '''


    SMA = nn.Conv1d(1, 1, kernel_size=w, padding="same", padding_mode="replicate", bias=False)
    list(SMA.parameters())[0].data = torch.ones(1,1,w)/w
    return SMA(x.T[:,None,:]).squeeze().T


# def rts_smoother(x_filt, x_pred, p_filt, p_pred, F, KT, C):
#     '''
#     x_filt: (T,N)
#     x_pred: (T,N)
#     p_filt: (T,N,N)
#     p_pred: (T,N,N)
#     F:      (T,N,N)
#     '''
#     T = len(x_filt)
#     N = F.shape[-1]

#     x_smooth = torch.zeros([T,N])
#     p_smooth = torch.zeros([T,N,N])
#     p2_smooth = torch.zeros([T,N,N])

#     x_smooth[-1] = x_filt[-1]
#     p_smooth[-1] = p_filt[-1]

#     # pairwise covariance
#     p2_smooth[-1] = (torch.eye(N) - KT @ C) @ F[-1] @ p_filt[-2]


#     Gts = torch.zeros([T-1,N,N])

#     for t in range(T-1,0, -1):
#         # TODO: could make smoother more efficient by using F @ p_filt.T from filtering distribution
#         Gt = torch.linalg.solve(p_pred[t], (F[t-1] @ p_filt[t-1].T)).T
#         Gts[t-1] = Gt
#         x_smooth[t-1] = x_filt[t-1] + Gt @ (x_smooth[t] -  x_pred[t])
#         S = (torch.eye(N) - Gt @ F[t-1])
#         p_smooth[t-1] = Gt @ p_smooth[t] @ Gt.T + S @ p_pred[t-1] @ S.T

#     for t in range(T-1,1, -1):
#         p2_smooth[t-1] = p_smooth[t] @ Gts[t-2].T
#     p2_smooth = p2_smooth[1:] 

#     return x_smooth, p_smooth, p2_smooth

def backward_sample(x_filt, x_pred, p_filt, p_pred, F):
    
    # pml2 murphy. section 8.2.3.5.     

    T, N = x_filt.shape 
    xs = torch.zeros((T, N))

    noise = torch.randn(T, N)
    xs[-1] = sample_gaussian(x_filt[-1], p_filt[-1], noise[-1])
    for t in range(T-1,0, -1):
        # TODO: could make smoother more efficient by using F @ p_filt.T from filtering distribution. or compute batch wise
        Gt = torch.linalg.solve(p_pred[t] ,(p_filt[t-1] @ F[t-1].T).T).T 
        m = x_filt[t-1] + Gt @ (xs[t] - x_pred[t]) # condition on sample
        s = (p_filt[t-1] - (Gt @ p_pred[t] @ Gt.T)) # Equation 8.87
        try:
            xs[t-1] = sample_gaussian(m, s, noise[t-1])
        except:
            s = project_pd(s)
            xs[t-1] = sample_gaussian(m, s, noise[t-1])
        
        
    return xs


def project_pd(A):
    B = (A + A.T)/2
    e,v = torch.linalg.eig(B)
    e_ = torch.max(torch.real(e), torch.zeros_like(torch.real(e)))
    A_pd = torch.real(v @ torch.diag(e_).cfloat() @ v.T)
    A_pd += torch.eye(len(A_pd))*1e-3
    return A_pd

