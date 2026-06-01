import numpy as np

R = 1.0


def rho0_func(x):
    return np.where(x <= 0.5, 1.0, 0.125)


def u0_func(x):
    return np.zeros_like(x)


def T0_func(x):
    return np.where(x <= 0.5, 5.0, 4.0)


def maxwellian_func(rho, u, T, v):
    pref = rho / np.sqrt(2 * np.pi * R * T)
    return pref * np.exp(-((v - u) ** 2) / (2.0 * R * T))


def f0_func(x, vx):
    rho = np.where(x <= 0.5, 1.0, 0.125)
    ux = np.zeros_like(x)
    T = np.where(x <= 0.5, 5.0, 4.0)
    R = 1.0

    pref = rho / (np.sqrt(2.0 * np.pi * R * T))

    c_squared = (vx - ux) ** 2

    return pref * np.exp(-c_squared / (2.0 * R * T))
