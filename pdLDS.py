from tqdm.auto import trange

from offset import LoessInference
from dynamics import kalman_filter, rts_smoother, backward_sample, sma
from cDynamics import SBLDFInference, fastSBLDFInference, convertfastSBL
from cDynamics import smooth_support_channelwise, smooth_support_acrosschannels, group_consecutive_ixs, solve_segmented_restricted_lstsq, sparseSmoother4, impute_c_dynamics
import numpy as np
from likelihood import obs_ll, dyn_ll


from optimizers import AdaBelief
# from adabelief_pytorch import AdaBeliefe

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

def ensure_torch_tensor_decorator(f):
    def inner(self, Y, *args, **kwargs):
        Y_ = []
        for y in Y:
            if type(y) != torch.Tensor:
                y = torch.tensor(y).float()
            Y_.append(y)
        f(self, Y_, args[0], **kwargs)

    return inner


def ensure_input_torch_tensor(Y):
    Y_ = []
    for y in Y:
        if type(y) != torch.Tensor:
            y = torch.tensor(y).float()
        Y_.append(y)
    return Y_



    
class pdLDS:
    def __init__(self, emissions_dim, 
                        latent_dim, 
                        n_dynamic_operators, seed=0):
        '''
        M: (int) Observation dim
        N: (int) latent dim
        K: (int) Number of dynamics. (size of dynamics dictionary)
        '''
        
        
        self.M = emissions_dim
        self.N = latent_dim
        self.K = n_dynamic_operators

        self.dynamics = None
        
        
        self.seed = seed
        # self.min_var = 1e-3

        # self.initiated = False
        
        
    def _init_param(self,y):
        # self.initiated = True

        # TODO: consider all data
        u, s, v = torch.pca_lowrank(y, q=self.N)
        x = y @ torch.linalg.pinv(v).T
        self.x_init = x
        
        # initialize parameters
        torch.manual_seed(self.seed)
        self.dynamics = nn.Parameter(torch.randn(self.K, self.N,self.N)*0.05, requires_grad=True)
        with torch.no_grad():
            # data driven intialization
            self.emissions = nn.Parameter(v, requires_grad=True)
#             self.emissions.data = self.emissions.data/self.emissions.data.norm(dim=0)
            self.Sy = torch.var(y - (self.emissions @ x.T).T,axis=0).float()

            # random initialization
            self.Sx = torch.ones(self.N).float()
            self.S0 = torch.eye(self.N).float()
            
    def _init_opt(self,):
        self.opt = AdaBelief([self.dynamics], 
                             lr=self.dynamics_lr, 
                             eps=1e-8,
                             rectify = False, 
                             print_change_log = False, 
                             weight_decay=self.weight_decay)
        
        # self.opt_C = AdaBelief([self.emissions], 
        #                        lr=self.emissions_lr, 
        #                        rectify = False,
        #                         eps=1e-8, 
        #                         print_change_log = False)


        self.lr_ = self.dynamics_lr
        self.sched_dynamics = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.985)
        # self.sched_dynamics_emissions = torch.optim.lr_scheduler.ExponentialLR(self.opt_C, gamma=0.925)

        
    def _infer_offset(self, x, best=None):
        # debugging code
        # best_s = self.S
                
        trend, lds = self.loess.loess_detrend(x, self.S)
        trend, lds = trend.float(), lds.float()

        self.lds = lds
        self.trend = trend
        
        
        # experimental              
#         self.mid = mid = (lds.max(0)[0] + lds.min(0)[0])/2
        
#         trend += mid
#         lds -= mid
        
        y_trend = (self.emissions @ trend.T).T
        y_lds = (self.emissions @ lds.T).T
        
        return y_trend, y_lds, trend, lds
                
                
    def _infer_latent(self, 
                      y_lds, 
                      F, 
                      lds0, 
                      ix):
        # self.Sx.data = torch.clamp(self.Sx.data, np.log(self.min_var))
        # self.Sy.data = torch.clamp(self.Sy.data, np.log(self.min_var))
        
        x_filt, x_pred, p_filt, p_pred, KT = kalman_filter(y_lds, F, self.emissions, lds0, torch.diag(self.Sy), torch.diag(self.Sx), self.S0)
        x_smooth, p_smooth, p2_smooth = rts_smoother(x_filt, x_pred, p_filt, p_pred, F, KT, self.emissions)
        
        
        # xs = backward_sample(x_filt, x_pred, p_filt, p_pred, F)
        xs = x_smooth
        
        x_sm = sma(x_smooth, 3)
        xdot = x_sm[1:] - x_sm[:-1]
        xdot = sma(xdot, 3).data
        
        return x_filt, x_smooth, xs, xdot, p_smooth, p2_smooth, p_filt, p_pred
    
    def _op_x_c(self, c):
        T = c.shape[0]
        return (self.dynamics.reshape(self.K,-1).T @ c.T).T.reshape(T,self.N,self.N)
    
    def _I_plus_opc(self, opxc):
        T = opxc.shape[0]
        return torch.tile(torch.eye(self.N),(T,1,1)) + opxc
    
    def _get_transitions(self,c):
        T = c.shape[0]
        opc = self._op_x_c(c)
        F = self._I_plus_opc(opc)
        return F
        

        
    
    def _infer_coef(self, x_smooth, xdot):
        fastOutput = fastSBLDFInference(x_smooth.numpy().astype(np.float64),
                                        xdot.numpy().astype(np.float64),
                                        self.dynamics.data.numpy().astype(np.float64), 
                                        xi=self.xi, 
                                        max_iter=200, 
                                        tol = 10e-5, 
                                        ab_init = 0)
        
        c_filt, gammas, lambdas, a, b, sigmas = convertfastSBL(fastOutput)
        self.c_filt, self.sigmas = c_filt, sigmas
        self.gammas = gammas
        self.sigmas = sigmas
        if self.smooth_c:
            # c_smooth, s_smooth = sparseSmoother(c_filt, sigmas, x_smooth,xdot, self.dynamics)
            # c_filt = self.initial_smooth(c_filt, x_smooth, xdot)
            # c_smooth, _ = sparseSmoother3(c_filt, sigmas)
            w = 4
            support = (c_filt.abs() > 1e-4)
            ss = smooth_support_channelwise(support, w)
            basis = smooth_support_acrosschannels(ss,w)

            impute_val = solve_segmented_restricted_lstsq(basis, xdot, self.dynamics, x_smooth)
            c_mod = impute_c_dynamics(c_filt, basis, impute_val)
            c_smooth, s_smooth = sparseSmoother4(c_mod, sigmas, basis)
#         c_smooth, s_smooth = sparseSmoother3(c_smooth, s_smooth)
        else:
            if self.init:
                c_smooth = self.initial_smooth(c_filt, x_smooth, xdot, self.init_smoothness)
            else:
                # c_filt = pdlds.coefs_filt[ix]
                # latent = pdlds.latent[ix]
                # xdot = latent[1:] - x_s[:-1]
                # xdot
                # x_smooth
                # sigmas = self.sigmas

               
                c_smooth = c_filt
            
        c_smooth[np.abs(c_smooth) < 1e-4] = 0 # machine precision sparsity
        self.c_smooth = c_smooth
        
        
        return c_smooth, c_filt



    def fit(self,Y, 
            S, 
            dynamics_lr=1, 
            emissions_lr=1,
            weight_decay=1, 
            train_iters=100,
            xi = 0.1,
            
            init = True,
            init_iters = 25, 
            smooth_c=True,
            init_smoothness = 20
            ):

        assert S > 1, "Offset Window Size must be > 1"
        assert 0 <= emissions_lr <=1, "emissions_lr must be between [0,1]"
        self.S = S

        if not init:
            assert self.dynamics is not None, "Dynamics is not initialized"

        Y = ensure_input_torch_tensor(Y)
        self.xi = xi
        self.emissions_lr = emissions_lr
        self.n_samples = len(Y)
        self.train_iters = train_iters
        self.dynamics_lr = dynamics_lr
        self.weight_decay = weight_decay
        self.smooth_c = smooth_c


        y = Y[0]
        
        if init:
            print(f"Initializing weights with {init_iters} iterations")
            self.loss_curve = []
            self.lrs = []
            self.offsets = [torch.zeros([i.shape[0], self.N]) for i in Y] 
            self.latent = [torch.zeros([i.shape[0], self.N]) for i in Y] 
            self.coefs = [torch.zeros([i.shape[0]-1, self.K]) for i in Y] 
            self.coefs_filt = [torch.zeros([i.shape[0]-1, self.K]) for i in Y] 
            self._init_param(y)
            self._init_opt()
            # self.c_smooth = False
            self.emissions_lr = 0
            self.init = init
            self.smooth_c = False
            if init_smoothness % 2 == 0:
                self.init_smoothness = init_smoothness
            else:
                self.init_smoothness = init_smoothness + 1

            # self.dynamics_lr = 0
            # self.init_iters = 25

            self._fit(Y, S, init_iters,)


        # self.c_smooth = smooth_c
        self.init = False
        self.smooth_c = smooth_c
        # self.dynamics_lr = dynamics_lr
        self.emissions_lr = emissions_lr
        print(f"Training with  {train_iters} iterations")
        self._fit(Y, S, self.train_iters)


    
    
    def _fit(self, 
            Y, 
            S, 
            epochs
            # dynamics_lr=1, 
            # emissions_lr=1,
            # weight_decay=1, 
            # num_iters=100,
            # xi = 0.1,
            # # s_bounds = (0.9, 0.9),
            # init = True, 
            # smooth_c=True,
           ):

        '''
        learn dynamics matrices.
        
        Y: (list of B x T x M np.arrays). observations
        S: (int > 1) smoothing window size
        dynamics_lr: (float). learning rate
        emissions_lr: [0,1] float. learning rate
        '''

        
        y = Y[0]
        
        
        pbar_outer = trange(epochs, desc="ELBO: --", position=0)
        
        
        i = 0
        while i < epochs:


            loss = 0     
            x = y @ torch.linalg.pinv(self.emissions).T
            # self.loess = LoessInference(self.s_bounds, T=len(x), x=x, best=None)
            self.loess = LoessInference( T=len(x), x=x, best=None)

            pbar_inner = trange(self.n_samples, desc="ELBO (-): --", position=1, leave=False)


            # self.opt_C.zero_grad()
            

            self.opt.zero_grad()
            for ix in pbar_inner:


                with torch.no_grad():
                    y = Y[ix].float()
                    xs, xdot, trend, p_smooth, p2_smooth = self._infer(y, 
                                                    self.latent, 
                                                    self.coefs, 
                                                    self.coefs_filt, 
                                                    self.offsets, 
                                                    ix)

                    self.p_smooth = p_smooth
                    self.p2_smooth = p2_smooth

                    

                with torch.set_grad_enabled(True):
                    F = self._get_transitions(self.coefs[ix])


                    # compute ll    
                    nll = 0
                    nll += -dyn_ll(xs[:-1],xs[1:], F, torch.diag(self.Sx))/(self.coefs[ix].shape[0])
                    nll += -obs_ll(y[1:], (F @ xs[:-1][:,:,None]).squeeze()+ trend[:-1], self.emissions, torch.diag(self.Sy))/(self.coefs[ix].shape[0])


                    nll /= self.n_samples


                    nll.backward()

                    loss += nll
                    pbar_inner.set_description(f"ELBO ({ix}): {nll:.4f}\t")
                    
            
            # print("DYNAMICS GRAD", self.dynamics.grad)
            torch.nn.utils.clip_grad_norm_(self.dynamics, 5)
            if np.sum([c.isnan().sum() for c in self.coefs]):
                print("COEFS NAN", )
                break
            self.opt.step()
            self.lrs.append(self.sched_dynamics.get_last_lr())
            # self.opt_C.step()
            if not self.init:
                self.update_emissions(Y)
            
            # self.emissions.data = self.emissions.data/self.emissions.data.norm(dim=0)

            self.sched_dynamics.step()
            # self.sched_dynamics_emissions.step()
            ratio_c = np.sum([(c.abs().sum(1) < 1e-2).sum() for c in self.coefs])/np.sum([len(c) for c in self.coefs])



            pbar_inner.close()
            self.loss_curve.append(loss.item())
            pbar_outer.set_description(f"ELBO: {loss.item():.4f} c:{ratio_c:.4f}")

            with torch.no_grad():
                self.S0 = p_smooth[0]

            
                self.update_dynamics_err()


            i += 1
            pbar_outer.update(1)

            
    def _infer(self, y, latent, coefs, coefs_filt, offsets, ix):
        
        x = y @ torch.linalg.pinv(self.emissions).T
        with torch.no_grad():

            y_trend, y_lds, trend, lds = self._infer_offset(x)
            offsets[ix] = trend
            c = coefs[ix]
            
            
            F = self._get_transitions(c)
            x_filt, x_smooth, xs, xdot, p_smooth, p2_smooth, p_filt, p_pred = self._infer_latent(y_lds, F, lds[0], 
                                                                                               ix)
            
            self.xdot  = xdot
            self.xs  = xs
            latent[ix] = x_smooth

            # print(latent[ix])
            
            c_smooth, c_filt = self._infer_coef(x_smooth, xdot)
            
        
            coefs[ix] = c_smooth
            coefs_filt[ix] = c_filt
            
            
        return xs, xdot, trend, p_smooth, p2_smooth
    
    def infer(self, Y, epochs=15):
        '''
        given learned dynamics matrices and data, infer the offsets and coefficients.

        y: (list of T x M np.arrays)
        '''

        assert hasattr(self, 'f') and hasattr(self, 'C'), "parameters are not fit"

        offsets = [torch.zeros([i.shape[0], self.N]) for i in Y] 
        latent = [torch.zeros([i.shape[0], self.N]) for i in Y] 
        coefs = [torch.zeros([i.shape[0]-1, self.K]) for i in Y] 
        coefs_filt = [torch.zeros([i.shape[0]-1, self.K]) for i in Y] 


        n_samples = len(Y)
        pbar = trange(n_samples, desc="ELBO: --", position=0)

        for ix in pbar:
            # self.loess = LoessInference(self.s_bounds, T=len(Y[ix]), x=None, best=None, init=False)
            self.loess = LoessInference( T=len(Y[ix]), x=None, best=None, init=False)
            for j in range(epochs):
                self._infer(Y[ix], latent, coefs, coefs_filt, offsets, ix)

        return offsets, latent, coefs, coefs_filt

    def update_emissions(self, Y):
        prior_xx = 1e-4 * torch.eye(self.N)
        prior_xy = torch.zeros([self.N, self.M])
        prior_yy = torch.zeros([self.M, self.M])


        ExxT = prior_xx
        ExyT = prior_xy
        EyyT = prior_yy
        weight_sum = 1

        for ys, xs, os in zip(Y, self.latent, self.offsets):
            weight_sum += len(xs)
            xs_ = xs+os

            ExxT += (xs_.T @ xs_)
            ExyT += (xs_.T @ ys)
            EyyT += (ys.T @ ys)

        alpha=self.emissions_lr



        emissions_update = torch.linalg.solve(ExxT, ExyT).T
        expected_err = EyyT - emissions_update @ ExyT - ExyT.T @ emissions_update.T + emissions_update @ ExxT @ emissions_update.T
        Sigma = (expected_err +  np.eye(self.M)) / (weight_sum + self.M )
        # with torch.no_grad():
        self.emissions.data = alpha*emissions_update + (1-alpha) * self.emissions.data


        self.Sy = torch.diag(Sigma).float()
        self.Sy = torch.maximum(self.Sy, torch.tensor(1e-3))

        # print("updating emissions", self.emissions_lr)
        self.emissions_lr *= 0.9 # lr decay
        # alpha *= 0.9 # lr decay

    def update_dynamics_err(self):
        weight = 1        
        Sx = torch.zeros(self.N)
        for xs,cs in zip(self.latent, self.coefs):
            weight += len(xs)

            F = self._get_transitions(cs)
            sq_err = (xs[1:] - (F.data @ xs[:-1][:,:,None]).squeeze()) **2 + 1e-4
            Sx += sq_err.sum(0)

        Sx /= weight

        Sx = torch.maximum(Sx, torch.tensor(1e-3))
        # print("Sx", Sx)
        self.Sx = Sx.float()

    def initial_smooth(self, c_filt, latent, xdot, w=20, rate_threshold=0.05):
        support = (c_filt.abs() > 1e-4)
        support = smooth_support_channelwise(support, w)
        support = smooth_support_acrosschannels(support,w, rate_threshold=rate_threshold)
        c_infer = solve_segmented_restricted_lstsq(support, xdot, self.dynamics, latent)
        return c_infer

        





