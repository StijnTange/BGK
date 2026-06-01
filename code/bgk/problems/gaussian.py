import numpy as np

R = 1.0


def u_of_x(x):
    scale = 1.0
    return (
        scale
        * 0.1
        * (np.exp(-((10.0 * x - 1.0) ** 2)) - 2.0 * np.exp(-((10.0 * x + 3.0) ** 2)))
    )


def maxwellian_func(rho, u, T, v):
    pref = rho / np.sqrt(2.0 * np.pi * R * T)
    return pref * np.exp(-((v - u) ** 2) / (2.0 * R * T))


def f0_func(x, v):
    rho0 = 1.0
    T0 = 1.0
    return maxwellian_func(rho0, u_of_x(x), T0, v)
