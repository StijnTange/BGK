import numpy as np
from bgk.core.grid import Grid
from bgk.core.reconstruction import lagrangian_reconstructor, weno3_reconstructor


class DistributionFunction:
    def __init__(self, grid: Grid, config):
        self.grid = grid
        self.dim_v = grid.dim_v
        self.config = config

    def initialize(self, f0_func):
        if self.dim_v == 1:
            x_mesh, vx_mesh = np.meshgrid(self.grid.x, self.grid.v, indexing="ij")
            self.f = f0_func(x_mesh, vx_mesh)
        elif self.dim_v == 2:
            x_mesh, vx_mesh, vy_mesh = np.meshgrid(
                self.grid.x, self.grid.vx, self.grid.vy, indexing="ij"
            )
            self.f = f0_func(x_mesh, vx_mesh, vy_mesh)
        else:
            raise ValueError("f0_func should have either 2 or 3 parameters.")

    def _integrate(self, array):
        if self.dim_v == 1:
            return np.sum(array, axis=-1) * self.grid.dv
        elif self.dim_v == 2:
            return np.sum(array, axis=(-2, -1)) * (self.grid.dvx * self.grid.dvy)
        else:
            raise ValueError(f"Unsupported velocity dimension: {self.dim_v}")

    def compute_moments(self, f_input=None):
        dim_v = self.dim_v
        f = self.f if f_input is None else f_input

        if dim_v == 1:
            v = self.grid.v[None, :]
            density = self._integrate(f)
            momentum = self._integrate(f * v)
            energy = self._integrate(f * 0.5 * v**2)

            return np.stack([density, momentum, energy])

        elif dim_v == 2:
            vx = self.grid.vx_mesh[None, :, :]
            vy = self.grid.vy_mesh[None, :, :]

            density = self._integrate(f)
            momentum_x = self._integrate(f * vx)
            momentum_y = self._integrate(f * vy)
            energy = self._integrate(f * 0.5 * (vx**2 + vy**2))

            return np.stack([density, momentum_x, momentum_y, energy])

        else:
            raise ValueError(f"Unsupported velocity dimension: {dim_v}")

    def compute_macroscopics(self, w: np.ndarray = None, R: float = 1.0, f_input=None):
        if w is None:
            w = self.compute_moments(f_input=f_input)

        dim_v = self.dim_v

        rho = w[0]
        E = w[-1]

        rho_min = 1e-12
        T_min = 1e-8
        valid = rho > rho_min
        rho_safe = np.maximum(rho, rho_min)

        if dim_v == 1:
            p_x = w[1]

            u = np.zeros_like(rho)
            u[valid] = p_x[valid] / rho_safe[valid]

            e = np.zeros_like(rho)
            e[valid] = (E[valid] / rho_safe[valid]) - 0.5 * u[valid] ** 2

            T = np.ones_like(rho) * T_min
            T[valid] = np.maximum((2.0 / dim_v) * e[valid] / R, T_min)

            return np.vstack([rho, u, T])

        elif dim_v == 2:
            p_x = w[1]
            p_y = w[2]

            u_x = np.zeros_like(rho)
            u_y = np.zeros_like(rho)

            u_x[valid] = p_x[valid] / rho_safe[valid]
            u_y[valid] = p_y[valid] / rho_safe[valid]

            e = np.zeros_like(rho)
            e[valid] = (E[valid] / rho_safe[valid]) - 0.5 * (
                u_x[valid] ** 2 + u_y[valid] ** 2
            )

            T = np.ones_like(rho) * T_min
            T[valid] = np.maximum((2.0 / dim_v) * e[valid] / R, T_min)

            return np.vstack([rho, u_x, u_y, T])

        else:
            raise ValueError(f"Unexpected number of moments provided: {len(w)}")

    def compute_heat_flux(self, u: np.ndarray = None, f_input=None):
        f = self.f if f_input is None else f_input
        dim_v = self.dim_v

        if u is None:
            macros = self.compute_macroscopics()
            u = macros[1 : 1 + dim_v]

        if dim_v == 1:
            v = self.grid.v[None, :]
            u_col = u.reshape(-1, 1)
            c = v - u_col
            c_squared = c**2
            integrand = f * c * 0.5 * c_squared
            q = self._integrate(integrand)
            q = q.reshape(1, -1)

            return q

        elif dim_v == 2:
            u_x = u[0]
            u_y = u[1]

            vx = self.grid.vx_mesh[None, :, :]
            vy = self.grid.vy_mesh[None, :, :]

            cx = vx - u_x[:, None, None]
            cy = vy - u_y[:, None, None]
            c_squared = cx**2 + cy**2

            qx = self._integrate(f * cx * 0.5 * c_squared)
            qy = self._integrate(f * cy * 0.5 * c_squared)

            return qx, qy

        else:
            raise ValueError(f"Unsupported velocity dimension: {dim_v}")

    def maxwellian(self, macros: np.ndarray = None, R: float = 1.0, f_input=None):
        f = self.f if f_input is None else f_input
        dim_v = self.dim_v

        if macros is None:
            macros = self.compute_macroscopics(R=R, f_input=f)

        rho = macros[0].reshape(-1, 1)
        T = macros[-1].reshape(-1, 1)

        if dim_v == 1:
            u_b = macros[1].reshape(-1, 1)

            c_squared = (self.grid.v[None, :] - u_b) ** 2
            rho_b = rho
            T_b = T

        elif dim_v == 2:
            ux_b = macros[1].reshape(-1, 1, 1)
            uy_b = macros[2].reshape(-1, 1, 1)

            c_squared = (self.grid.vx_mesh[None, :, :] - ux_b) ** 2 + (
                self.grid.vy_mesh[None, :, :] - uy_b
            ) ** 2

            rho_b = rho.reshape(-1, 1, 1)
            T_b = T.reshape(-1, 1, 1)

        else:
            raise ValueError(f"Unsupported velocity dimension: {dim_v}")

        pref = rho_b / ((2.0 * np.pi * R * T_b) ** (dim_v / 2.0))
        return pref * np.exp(-c_squared / (2.0 * R * T_b))

    def advect(
        self,
        dt: float,
        bc_type: str,
        config=None,
        f_input=None,
        lagrangian=False,
        mass_correction=True,
    ) -> np.ndarray:
        f_prev = self.f if f_input is None else f_input

        Nx = f_prev.shape[0]
        v_shape = f_prev.shape[1:]

        if self.dim_v == 1:
            vx = self.grid.v
        elif self.dim_v == 2:
            vx = self.grid.vx
        dx, x = self.grid.dx, self.grid.x
        xL, xR = self.grid.xL, self.grid.xR

        x_desired_shape = (Nx, len(vx)) + (1,) * (self.dim_v - 1)

        if bc_type == "periodic":
            L = xR - xL
            x_star = ((x[:, None] - vx[None, :] * dt - xL) % L) + xL
        elif bc_type == "specular":
            L = xR - xL
            x_u = x[:, None] - vx[None, :] * dt
            y_fold = (x_u - xL) % (2.0 * L)
            no_flip = y_fold <= L
            x_star = np.where(no_flip, xL + y_fold, xL + (2.0 * L - y_fold))
            _specular_no_flip = no_flip
        else:
            x_star, _ = backtrace_open(x, vx, dt, xL, xR)

        x_star_bcast = x_star.reshape(x_desired_shape)

        if bc_type in ["diffusive", "open", "K_steady"]:
            x_star_clamped = np.clip(x_star_bcast, x[0], x[-1])

            if bc_type == "diffusive":
                mass_before = np.trapezoid(self._integrate(f_prev), dx=dx, axis=0)

            if lagrangian:
                vx_reshaped = vx.reshape((len(vx),) + (1,) * (self.dim_v - 1))
                v_star_bcast = np.broadcast_to(vx_reshaped, x_star_clamped.shape)

                f_next = lagrangian_reconstructor(
                    y=f_prev,
                    x=x,
                    x_desired=x_star_clamped,
                    v_star=v_star_bcast,
                    order=1,
                )
            else:
                f_next = weno3_reconstructor(y=f_prev, x=x, x_desired=x_star_clamped)
            if bc_type == "diffusive":
                TL, TR = config.physics.T_L, config.physics.T_R
                VL, VR = config.physics.u_L, config.physics.u_R
                R = config.physics.R

                if self.dim_v == 1:
                    v_adv = self.grid.v
                    c_sq_L = (v_adv - VL) ** 2
                    c_sq_R = (v_adv - VR) ** 2
                    pref_L = 1.0 / np.sqrt(2 * np.pi * R * TL)
                    pref_R = 1.0 / np.sqrt(2 * np.pi * R * TR)
                else:
                    v_adv = self.grid.vx_mesh
                    c_sq_L = v_adv**2 + (self.grid.vy_mesh - VL) ** 2
                    c_sq_R = v_adv**2 + (self.grid.vy_mesh - VR) ** 2
                    pref_L = 1.0 / (2 * np.pi * R * TL)
                    pref_R = 1.0 / (2 * np.pi * R * TR)

                g_unit_L = pref_L * np.exp(-c_sq_L / (2 * R * TL))
                g_unit_R = pref_R * np.exp(-c_sq_R / (2 * R * TR))

                m_in_L = v_adv < 0
                m_out_L = v_adv > 0
                m_in_R = v_adv > 0
                m_out_R = v_adv < 0

                flux_in_L = self._integrate(np.where(m_in_L, v_adv * f_next[0], 0.0))
                flux_in_R = self._integrate(np.where(m_in_R, v_adv * f_next[-1], 0.0))

                flux_out_unit_L = self._integrate(
                    np.where(m_out_L, v_adv * g_unit_L, 0.0)
                )
                flux_out_unit_R = self._integrate(
                    np.where(m_out_R, v_adv * g_unit_R, 0.0)
                )

                rho_w_L = -flux_in_L / (flux_out_unit_L + 1e-15)
                rho_w_R = -flux_in_R / (flux_out_unit_R - 1e-15)

                f_next[0] = np.where(m_out_L, rho_w_L * g_unit_L, f_next[0])
                f_next[-1] = np.where(m_out_R, rho_w_R * g_unit_R, f_next[-1])

                if mass_correction:
                    mass_after_bc = np.trapezoid(self._integrate(f_next), dx=dx, axis=0)
                    mass_diff = mass_before - mass_after_bc

                    density_out_L = self._integrate(np.where(m_out_L, f_next[0], 0.0))
                    density_out_R = self._integrate(np.where(m_out_R, f_next[-1], 0.0))

                    mass_out_L = density_out_L * (dx / 2.0)
                    mass_out_R = density_out_R * (dx / 2.0)
                    mass_out_total = mass_out_L + mass_out_R

                    if mass_out_total > 1e-15:
                        alpha = 1.0 + (mass_diff / mass_out_total)
                        alpha = max(0.0, alpha)

                        f_next[0] = np.where(m_out_L, alpha * f_next[0], f_next[0])
                        f_next[-1] = np.where(m_out_R, alpha * f_next[-1], f_next[-1])

            elif bc_type == "inflow/outflow":
                max_displacement = np.max(np.abs(vx)) * dt / dx
                pad = max(3, int(np.ceil(max_displacement)) + 1)

                x_ext = np.concatenate(
                    [
                        x[0] - np.arange(1, pad + 1)[::-1] * dx,
                        x,
                        x[-1] + np.arange(1, pad + 1) * dx,
                    ]
                )

                f_padded = np.zeros((Nx + 2 * pad, *v_shape))
                f_padded[pad:-pad, ...] = f_prev

                v_adv = self.grid.v if self.dim_v == 1 else self.grid.vx
                v_pos = v_adv > 0
                v_neg = v_adv < 0
                R_n = self.config.physics.reflectance_left

                is_cell_centered = getattr(self.config.grid, "is_cell_centered", True)
                # ───────────────────────────────────────────────────────────────────

                for p in range(1, pad + 1):
                    if is_cell_centered:
                        idx_mirror_L = min(p - 1, Nx - 1)
                    else:
                        idx_mirror_L = min(p, Nx - 1)

                    f_reflected_L = f_prev[idx_mirror_L, ::-1]

                    if hasattr(self, "f_flow_left"):
                        f_padded[pad - p, v_pos] = (
                            self.f_flow_left[v_pos] + R_n * f_reflected_L[v_pos]
                        )
                    else:
                        f_padded[pad - p, v_pos] = R_n * f_reflected_L[v_pos]

                    f_padded[pad - p, v_neg] = f_prev[0, v_neg]

                    if hasattr(self, "f_flow_right"):
                        f_padded[pad + Nx - 1 + p, v_neg] = self.f_flow_right[v_neg]
                    else:
                        f_padded[pad + Nx - 1 + p, v_neg] = 0.0

                    f_padded[pad + Nx - 1 + p, v_pos] = f_prev[-1, v_pos]

                f_next = weno3_reconstructor(
                    y=f_padded, x=x_ext, x_desired=x_star_bcast
                )

        elif bc_type == "periodic":
            x_ext = np.concatenate(
                [[x[0] - 2 * dx, x[0] - dx], x, [x[-1] + dx, x[-1] + 2 * dx]]
            )
            f_padded = np.empty((Nx + 4, *v_shape))
            f_padded[2:-2, ...] = f_prev
            f_padded[1, ...] = f_prev[-2, ...]
            f_padded[0, ...] = f_prev[-3, ...]
            f_padded[-2, ...] = f_prev[1, ...]
            f_padded[-1, ...] = f_prev[2, ...]
            if lagrangian:
                v_star_bcast = np.broadcast_to(vx, x_star_bcast.shape)

                f_next = lagrangian_reconstructor(
                    y=f_padded,
                    x=x_ext,
                    x_desired=x_star_bcast,
                    v_star=v_star_bcast,
                    order=2,
                )
            else:
                f_next = weno3_reconstructor(
                    y=f_padded, x=x_ext, x_desired=x_star_bcast
                )

        elif bc_type == "specular":
            if self.dim_v == 1:
                f_flipped = f_prev[:, ::-1]
            else:
                f_flipped = f_prev[:, ::-1, :]

            x_ext = np.concatenate(
                [[x[0] - 2 * dx, x[0] - dx], x, [x[-1] + dx, x[-1] + 2 * dx]]
            )

            f_padded = np.empty((Nx + 4, *v_shape))
            f_padded[2:-2, ...] = f_prev
            f_padded[1, ...] = f_flipped[1, ...]
            f_padded[0, ...] = f_flipped[2, ...]
            f_padded[-2, ...] = f_flipped[-2, ...]
            f_padded[-1, ...] = f_flipped[-3, ...]

            f_padded_flip = np.empty((Nx + 4, *v_shape))
            f_padded_flip[2:-2, ...] = f_flipped
            f_padded_flip[1, ...] = f_prev[1, ...]
            f_padded_flip[0, ...] = f_prev[2, ...]
            f_padded_flip[-2, ...] = f_prev[-2, ...]
            f_padded_flip[-1, ...] = f_prev[-3, ...]

            r_no_flip = weno3_reconstructor(y=f_padded, x=x_ext, x_desired=x_star_bcast)
            r_flip = weno3_reconstructor(
                y=f_padded_flip, x=x_ext, x_desired=x_star_bcast
            )

            no_flip_bcast = _specular_no_flip.reshape(x_desired_shape)
            f_next = np.where(no_flip_bcast, r_no_flip, r_flip)

        elif bc_type == "inflow/outflow":
            max_displacement = np.max(np.abs(vx)) * dt / dx
            pad = max(3, int(np.ceil(max_displacement)) + 1)

            x_ext = np.concatenate(
                [
                    x[0] - np.arange(1, pad + 1)[::-1] * dx,
                    x,
                    x[-1] + np.arange(1, pad + 1) * dx,
                ]
            )

            f_padded = np.zeros((Nx + 2 * pad, *v_shape))
            f_padded[pad:-pad, ...] = f_prev

            v_adv = self.grid.v if self.dim_v == 1 else self.grid.vx
            v_pos = v_adv > 0
            v_neg = v_adv < 0
            R_n = self.config.physics.reflectance_left

            f_reflected_L = f_prev[0, ::-1]

            for p in range(1, pad + 1):
                if hasattr(self, "f_flow_left"):
                    f_padded[pad - p, v_pos] = (
                        self.f_flow_left[v_pos] + R_n * f_reflected_L[v_pos]
                    )
                else:
                    f_padded[pad - p, v_pos] = R_n * f_reflected_L[v_pos]

                f_padded[pad - p, v_neg] = f_prev[0, v_neg]

                if hasattr(self, "f_flow_right"):
                    f_padded[pad + Nx - 1 + p, v_neg] = self.f_flow_right[v_neg]
                else:
                    f_padded[pad + Nx - 1 + p, v_neg] = 0.0

                f_padded[pad + Nx - 1 + p, v_pos] = f_prev[-1, v_pos]

            f_next = weno3_reconstructor(y=f_padded, x=x_ext, x_desired=x_star_bcast)
        else:
            max_displacement = np.max(np.abs(vx)) * dt / dx
            pad = max(3, int(np.ceil(max_displacement)) + 1)
            x_ext = np.concatenate(
                [
                    x[0] - np.arange(1, pad + 1)[::-1] * dx,
                    x,
                    x[-1] + np.arange(1, pad + 1) * dx,
                ]
            )

            f_padded = np.zeros((Nx + 2 * pad, *v_shape))
            f_padded[pad:-pad, ...] = f_prev

            if bc_type == "inflow/outflow":
                v_adv = self.grid.v if self.dim_v == 1 else self.grid.vx
                v_pos = v_adv > 0
                v_neg = v_adv < 0
                R_n = self.config.physics.reflectance_left

                f_reflected_L = f_prev[0, ::-1]

                for p in range(1, pad + 1):
                    if hasattr(self, "f_flow_left"):
                        f_padded[pad - p, v_pos] = (
                            self.f_flow_left[v_pos] + R_n * f_reflected_L[v_pos]
                        )
                    else:
                        f_padded[pad - p, v_pos] = R_n * f_reflected_L[v_pos]

                    f_padded[pad - p, v_neg] = f_prev[0, v_neg]

                    if hasattr(self, "f_flow_right"):
                        f_padded[pad + Nx - 1 + p, v_neg] = self.f_flow_right[v_neg]
                    else:
                        f_padded[pad + Nx - 1 + p, v_neg] = 0.0

                    f_padded[pad + Nx - 1 + p, v_pos] = f_prev[-1, v_pos]
            f_next = weno3_reconstructor(y=f_padded, x=x_ext, x_desired=x_star_bcast)

        if f_input is None:
            self.f = f_next
        return f_next


def backtrace_periodic(x: np.ndarray, v: np.ndarray, dt: float, xL: float, xR: float):
    L = xR - xL
    x_star = ((x[:, None] - v[None, :] * dt - xL) % L) + xL
    return x_star, np.tile(v, (len(x), 1))


def backtrace_specular(x: np.ndarray, v: np.ndarray, dt: float, xL: float, xR: float):
    L = xR - xL

    x_u = x[:, None] - v[None, :] * dt

    y = (x_u - xL) % (2.0 * L)

    mask = y <= L

    x_star = np.where(mask, xL + y, xL + (2.0 * L - y))
    v_star = np.where(mask, v, -v)

    return x_star, v_star


def backtrace_open(x: np.ndarray, v: np.ndarray, dt: float, xL: float, xR: float):
    x_star = x[:, None] - v[None, :] * dt
    return x_star, np.tile(v, (len(x), 1))
