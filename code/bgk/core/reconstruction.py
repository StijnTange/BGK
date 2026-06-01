import numpy as np


def linear_reconstructor(y, x, x_desired):
    Nx = len(x)
    dx = x[1] - x[0]

    orig_shape = x_desired.shape
    xd_flat = x_desired.ravel()

    j = np.searchsorted(x, xd_flat) - 1
    j = np.clip(j, 0, Nx - 2).reshape(orig_shape)

    w_right = (x_desired - x[j]) / dx
    w_right = np.clip(w_right, 0.0, 1.0)
    w_left = 1.0 - w_right

    y_left = np.take_along_axis(y, j, axis=0)
    y_right = np.take_along_axis(y, j + 1, axis=0)

    return w_left * y_left + w_right * y_right


def get_indices(vi, j_left, order, nx):
    half_order = order // 2

    if order % 2 == 0:
        shifts = np.where(vi >= 0, -half_order, -(half_order - 1))
        start_indices = j_left + shifts
    else:
        start_indices = j_left - half_order

    start_indices = np.clip(start_indices, 0, nx - 1 - order)

    stencil_indices = start_indices[..., None] + np.arange(order + 1)

    return stencil_indices


def lagrangian_reconstructor(y, x, x_desired, v_star, order=2):
    nx = len(x)
    orig_shape = x_desired.shape

    j_left = np.searchsorted(x, x_desired.ravel()) - 1
    j_left = np.clip(j_left, 0, nx - 2).reshape(orig_shape)

    indices = get_indices(v_star, j_left, order, nx)
    k_plus_1 = order + 1

    x_s = x[indices]
    y_s = np.empty_like(x_s)
    y_s_shape = y.shape + (k_plus_1,)
    y_s = np.empty(y_s_shape, dtype=y.dtype)

    for k in range(k_plus_1):
        y_s[..., k] = np.take_along_axis(y, indices[..., k], axis=0)

    xi = x_desired[..., None]
    l_basis = np.ones_like(x_s)

    for j in range(k_plus_1):
        xj = x_s[..., j : j + 1]
        for m in range(k_plus_1):
            if m != j:
                xm = x_s[..., m : m + 1]
                l_basis[..., j : j + 1] *= (xi - xm) / (xj - xm)

    y_desired = np.sum(y_s * l_basis, axis=-1)

    return y_desired


def weno3_reconstructor(y, x, x_desired):
    Nx = len(x)
    dx = x[1] - x[0]
    dx2 = dx**2

    orig_shape = x_desired.shape

    j = np.searchsorted(x, x_desired.ravel()) - 1
    j = np.clip(j, 0, Nx - 2).reshape(orig_shape)

    j_m1 = np.maximum(j - 1, 0)
    j_p1 = np.minimum(j + 1, Nx - 1)
    j_p2 = np.minimum(j + 2, Nx - 1)

    v_jm1 = np.take_along_axis(y, j_m1, axis=0)
    v_j = np.take_along_axis(y, j, axis=0)
    v_jp1 = np.take_along_axis(y, j_p1, axis=0)
    v_jp2 = np.take_along_axis(y, j_p2, axis=0)

    xj = x[j]
    xd = x_desired

    L0_L = (xd - xj) * (xd - (xj + dx)) / (2 * dx2)
    L1_L = -(xd - (xj - dx)) * (xd - (xj + dx)) / dx2
    L2_L = (xd - (xj - dx)) * (xd - xj) / (2 * dx2)
    p_l = v_jm1 * L0_L + v_j * L1_L + v_jp1 * L2_L

    L0_R = (xd - (xj + dx)) * (xd - (xj + 2 * dx)) / (2 * dx2)
    L1_R = -(xd - xj) * (xd - (xj + 2 * dx)) / dx2
    L2_R = (xd - xj) * (xd - (xj + dx)) / (2 * dx2)
    p_r = v_j * L0_R + v_jp1 * L1_R + v_jp2 * L2_R

    bl = (
        (13 / 12) * v_jm1**2
        + (16 / 3) * v_j**2
        + (25 / 12) * v_jp1**2
        - (13 / 3) * v_jm1 * v_j
        + (13 / 6) * v_jm1 * v_jp1
        - (19 / 3) * v_j * v_jp1
    )

    br = (
        (13 / 12) * v_jp2**2
        + (16 / 3) * v_jp1**2
        + (25 / 12) * v_j**2
        - (13 / 3) * v_jp2 * v_jp1
        + (13 / 6) * v_jp2 * v_j
        - (19 / 3) * v_jp1 * v_j
    )

    c_l = (xj + 2 * dx - xd) / (3 * dx)
    c_r = (xd - (xj - dx)) / (3 * dx)

    eps = 1e-6
    al = c_l / (bl + eps) ** 2
    ar = c_r / (br + eps) ** 2

    al = np.where(j == 0, 0.0, al)
    ar = np.where(j == Nx - 2, 0.0, ar)

    w_sum = al + ar
    return (al * p_l + ar * p_r) / w_sum
