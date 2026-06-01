import numpy as np


def couette_velocity_continuum(x_nd, u_w_nd, L_nd=1.0):
    return (x_nd / L_nd) * u_w_nd


def couette_temperature_continuum(x_nd, T_w_nd, u_w_nd, Pr, cp_nd, L_nd=1.0):
    return T_w_nd + (Pr * u_w_nd**2 / (2.0 * cp_nd * L_nd**2)) * x_nd * (L_nd - x_nd)


def couette_shear_stress_continuum(u_w_nd, mu_0_nd, L_nd=1.0):
    return mu_0_nd * u_w_nd / L_nd


def couette_velocity_free_molecular(u_w_nd):
    return u_w_nd / 2.0


def couette_temperature_free_molecular(T_w_nd, u_w_nd, R_nd, n_v):
    return T_w_nd + u_w_nd**2 / (4.0 * n_v * R_nd)


def couette_shear_stress_free_molecular(rho_w_nd, u_w_nd, R_nd, T_w_nd):
    return rho_w_nd * u_w_nd * np.sqrt(R_nd * T_w_nd / (2.0 * np.pi))
