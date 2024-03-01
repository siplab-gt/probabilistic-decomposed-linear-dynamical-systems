import torch
import torch.nn as nn

def SBLDFInference(x_sample, xdot, G, xi = 10, max_iter = 1000, tol = 10e-5, ab_init = 1):
    
    T = len(xdot)
    D, N, _ = G.shape
    M = N
    
    # init SBL parameters
    a = torch.ones(D)*ab_init
    b = torch.ones(D)*ab_init

    # SBL through time
    g_i = torch.ones(D)*10 # gamma

    a_ = []
    b_ = []
    l_i = 1
    c_filt = []
    gammas = []
    lambdas = []
    sigmas = []
    # c_pred = []
    for t in range(T):
        a_.append(a)
        b_.append(b)

        l_i = 0.1
        # Inner SBL Loop per time point
        mu = torch.ones(D)*10e8
        Sigma = torch.ones(D)*10e8


        # INFER MU and SIGMA on q(c_t)
        for i in range(max_iter): # while has not converged
            old_m = mu
            old_S = Sigma

            A_ = (G @ x_sample[t][None,:,None]).squeeze()
            Phi = A_.T 

            # compute source posterior
            Sigma = torch.linalg.inv(torch.diag(1/g_i) + 1/l_i * Phi.T @ Phi)
            mu = 1/l_i * Sigma @ Phi.T @ xdot[t]

            # HP updates
            g_i = ((torch.diag(Sigma) + mu**2) + 2*b)/(1+2*a)
            l_i = (((xdot[t] - Phi @ mu)**2).sum() + torch.trace(Phi.T @ Phi @ Sigma))/M

            # check convergence
            change = torch.linalg.norm(old_S - Sigma) + torch.linalg.norm(old_m - mu)
            if change < tol:
                break

        c_filt.append(mu)
        gammas.append(g_i)
        lambdas.append(l_i)
        sigmas.append(Sigma)
        

        if t < T-1:
            if (mu**2).sum() < 10e-5: # reset gammas if mu goes to 0. 
                b = torch.ones(D)*ab_init
                a = torch.ones(D)*ab_init
                g_i = torch.ones(D)*10
            else:
                b = torch.ones(D)*xi*((mu + 1e-6) ** 2)
                a = torch.ones(D)*xi
                g_i = b / a

#         if np.isnan(mu).sum():
#             break


    c_filt = torch.vstack(c_filt)
    gammas = torch.vstack(gammas)
    lambdas = torch.stack(lambdas)
    a = torch.vstack(a_)
    b = torch.vstack(b_)
    sigmas  = torch.dstack(sigmas).transpose(2,0)
    
    return c_filt, gammas, lambdas, a, b, sigmas


def smooth_support_channelwise(support, w):
    support_smooth = torch.zeros_like(support)
    for i in range(len(support)-w):
        support_smooth[i+w//2] = torch.round(support[i:i+w+1].float().mean(0))

    for i in range(w//2):
        support_smooth[i] = support[i]
        support_smooth[-i] = support[-i]

    return support_smooth

def smooth_support_acrosschannels(support, w, rate_threshold=0.05 ):
    rates = torch.zeros(len(support))
    # support_smooth = torch.zeros_like(support)
    for i in range(len(support)-w):
    #     support_smooth[i+w//2] = torch.round(support[i:i+w+1].float().mean(0))

        b_ = support[i:i+w+1]
        rate = (b_.float()[1:] - b_.float()[:-1] != 0).sum(1).float().mean()
        rates[i+w//2] = rate


    support_smooth = torch.clone(support)
    for i in torch.where(rates > rate_threshold)[0]:
        support_smooth[i] = support_smooth[i - 1]

    for i in range(len(support_smooth)-3):
        l = support_smooth[i:i+3].float().mean(0)
        support_smooth[i+1] = torch.round(l)
    support_smooth[0] = support_smooth[1]
    support_smooth[-2] = support_smooth[-1]
    return support_smooth

def group_consecutive_ixs(x):
    ixs_ = torch.where((x.float()[1:]-x.float()[:-1]).abs().sum(1) != 0)[0]
    ixs = torch.zeros(len(ixs_)+2)
    ixs[1:-1] = ixs_
    ixs[-1] = len(x)-1
    return ixs

def solve_segmented_restricted_lstsq(basis, y, dynamics, x):

    ixs = group_consecutive_ixs(basis)

    values = []
    for k, (i,j) in enumerate(zip(ixs[:-1], ixs[1:])):
        
        ix1 = i.int()
        ix2 = j.int()
        if ix2 - ix1 == 1: # TODO: this is a hack to avoid extremely short sequences
            ix2 = np.min([j.int().item() +1, len(basis)-1])
            if ix2 == len(basis)-1:
                ix1 = i.int() - 2



        n_basis = basis[ix2].sum().item()
        obs = y[ix1:ix2] 
        dos = dynamics[basis[ix2]].data
        if n_basis == 1:

                phis = (dos @ x[ix1:ix2][:,:,None]).squeeze()
                phis_ = phis.flatten()
                obs_ = obs.flatten()
                c_ = torch.linalg.lstsq( phis_[:,None], obs_[:,None])[0].flatten()
    #             c_ = torch.nn.Parameter(torch.randn(1))

        # phis = phis.transpose(1,2)
        else:
            # print(ix2-ix1)
            
            phis = (dos[None,:] @ x[ix1:ix2][:,None,:, None]).squeeze()
            phis = phis.transpose(1,2)
            phis_ = phis.reshape(phis.shape[0]*phis.shape[1],n_basis)
            obs_ = obs.flatten()[:,None]
            c_ = torch.linalg.lstsq(phis_, obs_)[0].flatten()
            
#         print(k, i, j, c_)

        values.append([ix1.item(), ix2.item(), c_, basis[ix2]])

    c_infer = torch.zeros_like(basis.float())

    for (ix1, ix2, value, bs) in values:
        temp = torch.zeros_like(c_infer[ix1:ix2])
        temp[:,bs] = temp[:,bs]+value
        c_infer[ix1:ix2] = temp


    c_infer[-1] = c_infer[-2]
    c_infer *= basis
    return c_infer

def sparseSmoother(c_filt, sigmas, basis):
    sigmas = torch.diagonal(sigmas, dim1=1, dim2=2)
    ix = torch.where(sigmas.sum(1) == 0)[0]
    sigmas[ix] = torch.ones_like(sigmas[ix])*1e-8

    #     c_filt = dlds.coefs_filt[0]

    c_smooth = torch.zeros_like(c_filt)
    s_smooth = torch.zeros_like(sigmas)
    c_smooth[-1] = c_filt[-1]
    s_smooth[-1] = sigmas[-1]

    T = len(c_smooth)

    for t in range(T-1, 0, -1):

        c_t1 = c_smooth[t]
        c_t0 = c_filt[t-1]

        s_t1 = basis[t]
        s_t0 = basis[t-1]


        sig_t1 = s_smooth[t]
        sig_t0 = sigmas[t-1] + 1

        s_smooth[t-1] = s_t1.float()*sig_t1 * sig_t0/(sig_t1 + sig_t0)+(1-s_t1.float())*sig_t0*10000
        c_smooth[t-1] = s_t1.float()*(c_t1 * sig_t0 + c_t0 * sig_t1)/(sig_t1 + sig_t0)+(1-s_t1.float())*c_t0

    return c_smooth, s_smooth



def impute_c_dynamics(c_filt, basis, impute_val, thresh=1e-2):
    c_mod = torch.zeros_like(c_filt)
    c_mod[((c_filt * basis).abs() < thresh)] = impute_val[((c_filt * basis).abs() < thresh)]
    c_mod[((c_filt * basis).abs()>=  thresh)] = c_filt[((c_filt * basis).abs()>=  thresh)]
    return c_mod


from numba import jit
import numba
import numpy as np



from numba import jit
@jit(nopython=True)
def fastSBLDFInference(x_sample, xdot, G, xi = 10, max_iter = 1000, tol = 10e-5, ab_init = 0):

    '''
    via SBL-DF
    '''
   
    T = len(xdot)
    D, N, _ = G.shape
    M = N
    
#     print(T)
    
    # init SBL parameters
    a = np.ones(D)*ab_init
    b = np.ones(D)*ab_init

    # SBL through time
    g_i = np.ones(D,)*10 # gamma

    a_ = []
    b_ = []
    l_i = 1
    c_filt = np.zeros((T,D))
    sigmas = np.zeros((T,D,D))
    gammas = []
    lambdas = []
#     sigmas = []
    for t in range(T):
        a_.append(a)
        b_.append(b)

        
        l_i = 0.1
#         # Inner SBL Loop per time point
        mu = np.ones((D))*10e8
        Sigma = np.ones((D,D))*10e8
        

        A_ = np.zeros((D, N))
        for d in range(D):
            A_[d] = G[d] @ x_sample[t]
        Phi = A_.T 

#         # INFER MU and SIGMA on q(c_t)
        for i in range(max_iter): # while has not converged
            old_m = np.ascontiguousarray(mu)
            old_S = np.ascontiguousarray(Sigma)

             # compute source posterior
            Sigma = np.ascontiguousarray(np.linalg.inv(np.diag(1/g_i) + 1/l_i * Phi.T @ Phi))
            mu = np.ascontiguousarray(1/l_i * (Sigma @ Phi.T) @ xdot[t])

             # HP updates
            g_i = ((np.diag(Sigma) + mu**2) + 2*b)/(1+2*a)
            l_i = (((xdot[t] - Phi @ mu)**2).sum() + np.trace(Phi.T @ Phi @ Sigma))/M
            
             # check convergence
            change = np.linalg.norm(old_S - Sigma) + np.linalg.norm(old_m - mu)
            if change < tol:
                break

      
        

        if t < T-1:
            if (mu**2).sum() < 1e-6: # reset gammas if mu goes to 0. 
                b = np.ones(D)*ab_init
                a = np.ones(D)*ab_init
                g_i = np.ones(D)*10
                mu = c_filt[-1]
                Sigma = sigmas[-1]
            
                
            else:
                b = np.ones(D)*xi*((mu + 1e-6) ** 2)
                a = np.ones(D)*xi
                g_i = b / a

        c_filt[t] = mu
        sigmas[t] = Sigma
#         c_filt.append(mu)
        gammas.append(g_i)
        lambdas.append(l_i)
#         sigmas.append(Sigma)
    
    return c_filt, gammas, lambdas, a, b, sigmas



def convertfastSBL(fastOutput):
    c_filt, gammas, lambdas, a, b, sigmas = fastOutput
    
    c_filt = np.c_[c_filt]
    gammas = np.c_[gammas]
    lambdas = np.array(lambdas)
    a = np.c_[a]
    b = np.c_[b]
    sigmas = np.c_[sigmas]
    
    c_filt = torch.tensor(c_filt).float()
    gammas = torch.tensor(gammas).float()
    lambdas = torch.tensor(lambdas).float()
    a = torch.tensor(a).float()
    b = torch.tensor(b).float()
    sigmas = torch.tensor(sigmas).float()
    
    
    return c_filt, gammas, lambdas, a, b, sigmas
    

def sparseSmoother_opt(model, c_filt, sigmas, basis, latent):
    x0, x1 = latent[:-1], latent[1:]
    dx = x1 - x0
    Phis = (model.dynamics[None] @ x0[:,None,:,None]).squeeze()
    b0, b1 = basis.float()[:-1], basis.float()[1:]
    bs = (b1 - b0).abs().sum(1)

    cs = nn.Parameter(c_filt.clone())
    c_opt = torch.optim.Adam([cs], lr=1e-2)

    bb2 = torch.roll(bs == 0, -1)
    bb1 = torch.roll(bs == 0, 1)
    bb0 = bs == 0

    bbs = bb1*bb0*bb2
    for i in range(100):
        c_opt.zero_grad()
        cs_ = cs*basis

        dx_pred = (Phis.transpose(1,2).data @ cs_[:,:,None]).squeeze()
        loss = ((dx - dx_pred)**2).sum() + ((cs_[1:] - cs_[:-1])[bbs]**2).sum() + ((cs_[1:] - cs_[:-1])[bbs].abs()).sum()
        loss.backward()
        c_opt.step()
    # + (cs.abs()).sum() + ((cs[1:] - cs[:-1]).abs()).sum()

#         print(loss.item())
    c_smooth = cs.data
    s_smooth = torch.zeros_like(c_smooth)
    return c_smooth, s_smooth


