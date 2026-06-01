import numpy as np


def fourier_temperature_continuum(x_nd, T_L_nd, T_R_nd, omega, L_nd=1.0):
    exp = omega + 1.0
    return (T_L_nd**exp + (T_R_nd**exp - T_L_nd**exp) * x_nd / L_nd) ** (1.0 / exp)


def fourier_heat_flux_continuum(
    T_L_nd, T_R_nd, omega, mu_0_nd, R_nd, n_v, Pr=1.0, L_nd=1.0
):
    cp_nd = (n_v / 2.0 + 1.0) * R_nd
    hf_coeff = cp_nd / Pr
    T_ref_nd = 1.0
    return (
        -hf_coeff
        * mu_0_nd
        * (T_R_nd ** (omega + 1) - T_L_nd ** (omega + 1))
        / ((omega + 1.0) * T_ref_nd**omega * L_nd)
    )


def fourier_temperature_free_molecular(T_L_nd, T_R_nd):
    return np.sqrt(T_L_nd * T_R_nd)


def fourier_heat_flux_free_molecular(rho_L_nd, T_L_nd, T_R_nd, R_nd, n_v):
    rho_R_nd = rho_L_nd * np.sqrt(T_L_nd / T_R_nd)
    coeff = (n_v + 1.0) / 2.0
    q_plus = coeff * R_nd * rho_L_nd * np.sqrt(R_nd * T_L_nd / (2.0 * np.pi)) * T_L_nd
    q_minus = -coeff * R_nd * rho_R_nd * np.sqrt(R_nd * T_R_nd / (2.0 * np.pi)) * T_R_nd
    return q_plus + q_minus
