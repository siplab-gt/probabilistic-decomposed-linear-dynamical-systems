# generate data - Lorenz
import numpy as np
from scipy.integrate import solve_ivp
import numpy.random as npr

from scipy.linalg import expm
import torch

def get_state(x):

    x_b = 0.8
    if x[0] > x_b:
        return 0
    elif x[0] < -x_b:
        return 1
    elif x[1] > 0:
        return 2
    elif x[1] < 0:
        return 3
    

def generate_NASCAR(n_trials,
                    T = 1000,
                    seed = 1,
                    measurement_noise = 0.5,
                    process_noise=0.01
                   ):
    
    As = np.array([[[0,0.1],
                    [-0.1, 0.0]],
                   [[0,0.1],
                    [-0.1, 0.0]],
                   [[0,  0],
                    [-0, 0.0]],
                   [[0, 0.],
                    [-0, 0.]]])

    bs = np.array([[0.   , 0.005],
                    [0.,  -0.005],
                   [ 0.1       ,  0.        ],
                   [-0.1     ,  0.        ]])


    N = 2
    npr.seed(seed)
    D = npr.randn(10, 2)

    X = []
    Z = []
    Y = []
    speed = []

    for seed in range(n_trials):
        npr.seed(seed)

        z = 0

        t = npr.rand()*0.9+0.1 # speed
        ts = []
        zs = []
        x0 = npr.randn(N)*0.5
        x = [x0]
        s = []
        for i in range(T):
            last_state = np.copy(z)
            z = get_state(x[-1])
            zs.append(z)
            x1 = expm(t*As[z])@x[-1] + t*bs[z] + npr.randn(N)*process_noise
            x1_true = expm(t*As[z])@x[-1] + t*bs[z]

            s_ = x1_true - x[-1]

            s.append(s_)
            x.append(x1)
            ts.append(t)
            if last_state != z:
                # sample random speed
                t = npr.rand()*0.9+0.1


        s = np.c_[s]
        x = np.c_[x]
        X.append(x)
        Z.append(zs)
        speed.append(s)

        y = (D @ x.T).T + npr.randn(x.shape[0], 10)*measurement_noise
        Y.append(y)
        
    return Y, X, Z, speed


def lorenz_xdot(t, x, sig, rho, beta):
#     x0 = np.array([0., 1., 1.05])
#     print(x)
    dx = sig*(x[1] - x[0])
    dy = x[0]*(rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])


def generate_ramped_lorenz(n_trials,
                           M = 10,
                           T = 1000,
                           measure_noise_std = 1,
                           seed = 0,
                          ):
    sig = 10
    beta = 8/3
    rho = 28

    args = (sig, rho, beta) # lorenz

    N = 3
    npr.seed(seed)
    true_D = npr.randn(M, N) *0.1
#     * 0.1

    Y = []
    X = []
    speed = []
    for i in range(n_trials):
        npr.seed(i)
        x0 = npr.randn(N)*10
        x = [x0]
        s = []
        for i in range(int(T/100)):
            if i == 0:
                x0 = x[-1]
            else:
                x0 = x[-1][-1]

            speed_ = npr.rand()*1 + 0.25
            t = np.linspace(0,speed_,100)
            t_ = (np.exp(t) - 1)
            s.append(t_)
            
            sol = solve_ivp(lorenz_xdot, [0, t_.max()], x0, args=args, t_eval=t_)
            true_x = sol.y.T[1:]
            x.append(true_x)
        speed.append(np.concatenate(s))

        true_x = np.vstack(x)
        X.append(true_x)

        true_y = (true_D @ true_x.T).T 
        y = true_y + npr.randn(*true_y.shape) * measure_noise_std
        Y.append(torch.tensor(y).float())
    return Y, X, speed



def generateLorenz(x0 = np.array([1., 10., 10.]), noise_std = 5, T = 2000):

    
    a = 0.2

    sig = 10
    beta = 8/3
    rho = 28

    # lorenz_xdot(x, t, sig, rho, beta)
    # args = (0.1, 0.1, 14)
    args = (sig, rho, beta) # lorenz
    # args = (0.2, -0.01, 1, -0.4, -1, -1 ) # 4 wings

    
    sol = solve_ivp(lorenz_xdot, [0, 30], x0, args=args, t_eval=np.linspace(0,30,T))
    true_x = sol.y.T

    
    # y = true_x + npr.randn(*true_x.shape)*noise_std
    M, N = 10, 3
    true_D = npr.randn(M, N) * 1
    true_y = (true_D @ true_x.T).T 
    y = true_y + npr.randn(*true_y.shape) * noise_std
    return y, (true_y, true_x, true_D)



