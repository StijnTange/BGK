import numpy as np


def get_fourier_f0_func(config):
    T_L = config.physics.T_L
    T_R = config.physics.T_R
    R = config.physics.R
    dim_v = config.grid.dim_v

    if dim_v == 1:

        def f0_func(x, vx):
            T = np.ones_like(x)
            rho = np.ones_like(T)
            pref = rho / np.sqrt(2.0 * np.pi * R * T)
            return pref * np.exp(-(vx**2) / (2.0 * R * T))

    elif dim_v == 2:

        def f0_func(x, vx, vy):
            xL = x.min()
            xR = x.max()
            T = T_L + (T_R - T_L) * (x - xL) / (xR - xL)
            rho = np.ones_like(T)
            pref = rho / (2.0 * np.pi * R * T)
            return pref * np.exp(-(vx**2 + vy**2) / (2.0 * R * T))

    else:
        raise ValueError(f"dim_v={dim_v} not supported for Fourier. Use 1 or 2.")

    return f0_func
