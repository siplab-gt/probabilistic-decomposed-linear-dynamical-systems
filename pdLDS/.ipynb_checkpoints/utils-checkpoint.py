import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import patches


def get_phase_portrait(df, grid, t):
    '''
    df: function with parameters (x,t) for the time derivative \dot{x}
    grid: list of tuples containing (min, max, step) for each dimension
    t: value for t in df
    '''
    
    mgrid = []
    for i in grid:
        mgrid.append(np.arange(i[0], i[1]+i[2], i[2]))

    X = np.meshgrid(*mgrid)
    coords = [i.flatten()for i in X]
    coords = np.c_[coords].T
    dX = df(coords, t)
    dX = [j.reshape(i.shape) for i, j  in zip(X, dX.T)]

    return X, dX






def sum_abs_err(x_true, x_pred):
    return ((x_true - x_pred)).abs().sum(1)


def get_IQR(x):
    x = x.flatten().sort()[0]
    n = int(len(x)/2)
    Q1 = x[:n].median()
    Q3 = x[n:].median()

    IQR = Q3 - Q1

    lower = Q1 - (1.5*IQR)
    upper = Q3 + (1.5*IQR)

    return lower,upper, IQR, Q1, Q3
