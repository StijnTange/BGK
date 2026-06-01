import numpy as np

R = 1.0


def f0_func(x, vx, vy):
    # Reference dimensionless macroscopic state
    rho = 1.0
    T = 1.0
    ux = 0.0
    uy = 0.0
    R = 1.0

    # 2D Maxwellian
    pref = rho / (2.0 * np.pi * R * T)
    return pref * np.exp(-((vx - ux) ** 2 + (vy - uy) ** 2) / (2.0 * R * T))
