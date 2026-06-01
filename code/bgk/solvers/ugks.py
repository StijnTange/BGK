import numpy as np
from bgk.core.distribution import DistributionFunction
from bgk.solvers.base import Solver


class UGKSSolver(Solver):
    def __init__(self, config):
        super().__init__(config)

    def step(
        self,
        df: DistributionFunction,
        dt: float,
        **kwargs,
    ) -> DistributionFunction:
        dx = df.grid.dx
        pad = 2

        f_padded = self._pad_with_ghosts(df, pad=pad)

        f_face, df_face, f_L, f_R = self._compute_interface_reconstruction(
            f_padded, dx, df
        )

        w_face = self._compute_interface_w(df, f_L, f_R)

        w_cells = df.compute_moments(f_input=f_padded)
        if w_cells.shape[0] != df.dim_v + 2:
            w_cells = w_cells.T

        macros_cells = df.compute_macroscopics(w=w_cells, R=self.R)
        rho_cells = np.maximum(macros_cells[0], 1e-10)
        T_cells = np.maximum(macros_cells[-1], 1e-10)
        tau_n = self._compute_tau(rho_cells, T_cells)

        macros_face = df.compute_macroscopics(w=w_face, R=self.R)
        rho_face = np.maximum(macros_face[0], 1e-10)
        T_face = np.maximum(macros_face[-1], 1e-10)
        tau_face_phys = self._compute_tau(rho_face, T_face)

        p_cells = rho_cells * self.R * T_cells
        p_L = p_cells[:-1]
        p_R = p_cells[1:]
        shock_sensor = np.abs(p_L - p_R) / (p_L + p_R + 1e-15)
        tau_shape = (-1,) + (1,) * df.dim_v
        tau_face = tau_face_phys + (shock_sensor * dt).reshape(tau_shape)

        g_face = self._compute_interface_g(df, w_face)

        a_left = 2.0 * (w_face - w_cells[:, :-1]) / dx
        a_right = 2.0 * (w_cells[:, 1:] - w_face) / dx
        dgx_face = self._compute_dg_from_dw_upwind(df, g_face, w_face, a_left, a_right)

        dwdt = self._compute_dwdt(df, df_face)
        dgt_face = self._compute_dg_from_dw(df, g_face, w_face, dwdt)

        f_hat_padded = self._compute_time_averaged_f(
            df, f_face, df_face, g_face, dgx_face, dgt_face, dt, tau_face
        )

        f_hat_phys = f_hat_padded[pad - 1 : -pad + 1].copy()

        if self.bc_type == "diffusive":
            TL = self.config.physics.T_L
            TR = self.config.physics.T_R
            VL = self.config.physics.u_L
            VR = self.config.physics.u_R

            p_inner_L = p_cells[pad]
            p_inner_R = p_cells[-(pad + 1)]

            f_inner = df.f
            slope_0 = (f_inner[1] - f_inner[0]) / dx
            slope_m1 = (f_inner[-1] - f_inner[-2]) / dx
            f0_left = f_inner[0] - 0.5 * dx * slope_0
            f0_right = f_inner[-1] + 0.5 * dx * slope_m1

            w_inner_L = w_cells[:, pad]
            w_inner_R = w_cells[:, -(pad + 1)]

            f_hat_phys[0] = self._apply_wall_bc(
                df=df,
                f_hat_face=f_hat_phys[0],
                T_wall=TL,
                V_wall=VL,
                p_inner=p_inner_L,
                is_left=True,
                dt=dt,
                tau_face=tau_face_phys[pad - 1],
                f_face_0=f0_left,
                df_face_0=slope_0,
                w_inner_cell=w_inner_L,
            )
            f_hat_phys[-1] = self._apply_wall_bc(
                df=df,
                f_hat_face=f_hat_phys[-1],
                T_wall=TR,
                V_wall=VR,
                p_inner=p_inner_R,
                is_left=False,
                dt=dt,
                tau_face=tau_face_phys[-(pad)],
                f_face_0=f0_right,
                df_face_0=slope_m1,
                w_inner_cell=w_inner_R,
            )

        macro_flux_phys = self._compute_macroscopic_flux(df, f_hat_phys)
        w_inner = w_cells[:, pad:-pad]
        w_n1 = self._update_macroscopics(w_inner, macro_flux_phys, dt, dx)

        macros_n1 = df.compute_macroscopics(w=w_n1, R=self.R)
        g_n1_inner = df.maxwellian(macros=macros_n1, R=self.R)
        tau_n1 = self._compute_tau(macros_n1[0], macros_n1[-1])

        macros_n = df.compute_macroscopics(w_cells[:, pad:-pad], R=self.R)
        g_n_inner = df.maxwellian(macros=macros_n, R=self.R)

        f_inner_n = f_padded[pad:-pad]
        tau_n_inner = tau_n[pad:-pad]

        f_new = self._final_microscopic_update(
            df=df,
            f_n=f_inner_n,
            f_hat_plus=f_hat_phys[1:],
            f_hat_minus=f_hat_phys[:-1],
            g_n=g_n_inner,
            g_n1=g_n1_inner,
            dt=dt,
            dx=dx,
            tau_n=tau_n_inner,
            tau_n1=tau_n1,
        )

        df.f = f_new
        return df

    def _wall_flux(
        self,
        df: DistributionFunction,
        f_hat_face: np.ndarray,
        T_wall: float,
        V_wall: float,
        p_inner: float,
        is_left: bool,
        dt: float,
        tau_face: np.ndarray,
        f_face_0: np.ndarray,
        df_face_0: np.ndarray,
        w_inner_cell: np.ndarray,
    ) -> np.ndarray:
        dim_v = df.dim_v
        v_adv = df.grid.v if dim_v == 1 else df.grid.vx_mesh
        dx = df.grid.dx

        if dim_v == 1:
            pref_w = 1.0 / np.sqrt(2.0 * np.pi * self.R * T_wall)
            c_sq_w = (df.grid.v - V_wall) ** 2
        else:
            pref_w = 1.0 / (2.0 * np.pi * self.R * T_wall)
            c_sq_w = df.grid.vx_mesh**2 + (df.grid.vy_mesh - V_wall) ** 2
        g_unit_wall = pref_w * np.exp(-c_sq_w / (2.0 * self.R * T_wall))

        rho_wall_g = p_inner / (self.R * T_wall)
        g_wall = rho_wall_g * g_unit_wall

        g_wall_face = g_wall[None, ...]
        w_wall = df.compute_moments(f_input=g_wall_face)
        if w_wall.shape[0] != dim_v + 2:
            w_wall = w_wall.T
        w_wall = w_wall.flatten()

        if is_left:
            inc_mask_pre = v_adv < 0
        else:
            inc_mask_pre = v_adv > 0
        f_face_upwind = np.where(inc_mask_pre, f_face_0, g_wall)
        w_off = df.compute_moments(f_input=f_face_upwind[None, ...])
        if w_off.shape[0] != dim_v + 2:
            w_off = w_off.T
        w_off = w_off.flatten()

        if is_left:
            dwdx_wall = 2.0 * (w_inner_cell - w_off) / dx
        else:
            dwdx_wall = 2.0 * (w_off - w_inner_cell) / dx

        integrand = (-v_adv * df_face_0)[None, ...]
        dwdt_wall = df.compute_moments(f_input=integrand)
        if dwdt_wall.shape[0] != dim_v + 2:
            dwdt_wall = dwdt_wall.T
        dwdt_wall = dwdt_wall.flatten()

        w_wall_2d = w_wall.reshape(-1, 1)
        dwdx_wall_2d = dwdx_wall.reshape(-1, 1)
        dwdt_wall_2d = dwdt_wall.reshape(-1, 1)
        dgx_wall = self._compute_dg_from_dw(df, g_wall_face, w_wall_2d, dwdx_wall_2d)[0]
        dgt_wall = self._compute_dg_from_dw(df, g_wall_face, w_wall_2d, dwdt_wall_2d)[0]

        tau_bc = np.atleast_1d(tau_face).reshape((1,) * dim_v)
        C1, C2, C3, C4, C5 = self._get_time_averaged_coeffs(dt, tau_bc)

        f_hat_wall_full = (
            C1 * f_face_0
            + C2 * df_face_0 * v_adv
            + C3 * g_wall
            + C4 * dgx_wall * v_adv
            + C5 * dgt_wall
        )

        if is_left:
            inc_mask = v_adv < 0
            out_mask = v_adv > 0
        else:
            inc_mask = v_adv > 0
            out_mask = v_adv < 0

        flux_in = df._integrate(np.where(inc_mask, v_adv * f_hat_wall_full, 0.0))
        flux_out_unit = df._integrate(np.where(out_mask, v_adv * g_unit_wall, 0.0))
        rho_w = -flux_in / (flux_out_unit + 1e-15)

        return np.where(inc_mask, f_hat_wall_full, rho_w * g_unit_wall)

    def _apply_wall_bc(self, **kwargs) -> np.ndarray:
        return self._wall_flux(**kwargs)

    def _get_slopes(self, f: np.ndarray, dx: float) -> np.ndarray:
        s1 = (f[1:-1, ...] - f[0:-2, ...]) / dx
        s2 = (f[2:, ...] - f[1:-1, ...]) / dx
        eps = 1e-15
        slopes_inner = (
            (np.sign(s1) + np.sign(s2))
            * (np.abs(s1) * np.abs(s2))
            / (np.abs(s1) + np.abs(s2) + eps)
        )
        slopes = np.zeros_like(f)
        slopes[1:-1, ...] = slopes_inner
        return slopes

    def _compute_interface_reconstruction(self, f, dx, df):
        slopes = self._get_slopes(f, dx)
        f_left_limit = f[:-1, ...] + 0.5 * dx * slopes[:-1, ...]
        f_right_limit = f[1:, ...] - 0.5 * dx * slopes[1:, ...]
        vx = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_pos = vx[None, ...] > 0
        f_face = np.where(v_pos, f_left_limit, f_right_limit)
        df_face = np.where(v_pos, slopes[:-1, ...], slopes[1:, ...])
        return f_face, df_face, f_left_limit, f_right_limit

    def _get_time_averaged_coeffs(self, dt, tau):
        x = dt / tau
        small = x < 1e-3

        x_safe = np.where(small, 1.0, x)
        inv_x = 1.0 / x_safe
        exp_x = np.exp(-x)
        omx = 1.0 - exp_x

        C1_d = inv_x * omx
        C2_d = dt * (inv_x * exp_x - inv_x**2 * omx)
        C3_d = 1.0 - inv_x * omx
        C4_d = dt * (2.0 * inv_x**2 * omx - inv_x - inv_x * exp_x)
        C5_d = dt * (0.5 - inv_x + inv_x**2 * omx)

        C1_t = 1.0 - x / 2.0 + x**2 / 6.0 - x**3 / 24.0 + x**4 / 120.0
        C2_t = dt * (-0.5 + x / 3.0 - x**2 / 8.0 + x**3 / 30.0 - x**4 / 144.0)
        C3_t = x / 2.0 - x**2 / 6.0 + x**3 / 24.0 - x**4 / 120.0
        C4_t = dt * (-x / 6.0 + x**2 / 12.0 - x**3 / 40.0 + x**4 / 180.0)
        C5_t = dt * (x / 6.0 - x**2 / 24.0 + x**3 / 120.0 - x**4 / 720.0)

        C1 = np.where(small, C1_t, C1_d)
        C2 = np.where(small, C2_t, C2_d)
        C3 = np.where(small, C3_t, C3_d)
        C4 = np.where(small, C4_t, C4_d)
        C5 = np.where(small, C5_t, C5_d)

        return C1, C2, C3, C4, C5

    def _compute_time_averaged_f(
        self, df, f_face, df_face, g_face, dgx_face, dgt_face, dt, tau_face
    ):
        C1, C2, C3, C4, C5 = self._get_time_averaged_coeffs(dt, tau_face)
        v_adv = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_adv_b = v_adv[None, ...]
        f_hat_fr = C1 * f_face + C2 * df_face * v_adv_b
        f_hat_eq = C3 * g_face + C4 * dgx_face * v_adv_b + C5 * dgt_face
        return f_hat_fr + f_hat_eq

    def _compute_interface_w(self, df, f_L, f_R):
        vx = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_pos = vx[None, ...] > 0
        v_neg = vx[None, ...] < 0
        f_upw = np.where(v_pos, f_L, 0.0) + np.where(v_neg, f_R, 0.0)
        w = df.compute_moments(f_input=f_upw)
        if w.shape[0] != df.dim_v + 2:
            w = w.T
        return w

    def _compute_dg_from_dw_upwind(self, df, g_face, w_face, a_left, a_right):
        vx = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_pos = vx > 0

        dim_v = df.dim_v

        dgx_left = self._compute_dg_from_dw(df, g_face, w_face, a_left)
        dgx_right = self._compute_dg_from_dw(df, g_face, w_face, a_right)

        if dim_v == 1:
            v_pos_b = v_pos[None, :]
        else:
            v_pos_b = v_pos[None, :, :]

        return np.where(v_pos_b, dgx_left, dgx_right)

    def _compute_interface_g(self, df, w_face):
        macros = df.compute_macroscopics(w_face, R=self.R)
        rho = np.maximum(macros[0], 1e-10)
        T = np.maximum(macros[-1], 1e-10)
        dim_v = df.dim_v
        if dim_v == 1:
            u = macros[1]
            rho_b = rho[:, None]
            T_b = T[:, None]
            u_b = u[:, None]
            c_sq = (df.grid.v[None, :] - u_b) ** 2
        else:
            u_x = macros[1]
            u_y = macros[2]
            rho_b = rho[:, None, None]
            T_b = T[:, None, None]
            ux_b = u_x[:, None, None]
            uy_b = u_y[:, None, None]
            c_sq = (df.grid.vx_mesh[None, :, :] - ux_b) ** 2 + (
                df.grid.vy_mesh[None, :, :] - uy_b
            ) ** 2
        pref = rho_b / ((2.0 * np.pi * self.R * T_b) ** (dim_v / 2.0))
        return pref * np.exp(-c_sq / (2.0 * self.R * T_b))

    def _compute_dwdx(self, w_cells, w_face, dx):
        a_left = 2.0 * (w_face - w_cells[:, :-1]) / dx
        a_right = 2.0 * (w_cells[:, 1:] - w_face) / dx
        return a_left, a_right

    def _compute_dwdt(self, df, df_face):
        v_adv = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_adv_b = v_adv[None, ...]
        integrand = -v_adv_b * df_face
        w = df.compute_moments(f_input=integrand)
        if w.shape[0] != df.dim_v + 2:
            w = w.T
        return w

    def _compute_dg_from_dw(self, df, g, w_face, dw):
        dim_v = df.dim_v
        macros = df.compute_macroscopics(w_face, R=self.R)
        rho = np.maximum(macros[0], 1e-10)
        T = np.maximum(macros[-1], 1e-10)
        d_rho = dw[0]
        cv_fac = dim_v / 2.0

        if dim_v == 1:
            u = macros[1]
            d_rhou = dw[1]
            d_E = dw[2]
            d_u = (d_rhou - u * d_rho) / rho
            d_T = (
                d_E - cv_fac * self.R * T * d_rho - 0.5 * u**2 * d_rho - rho * u * d_u
            ) / (rho * cv_fac * self.R)
            rho_b = rho[:, None]
            T_b = T[:, None]
            d_rho_b = d_rho[:, None]
            d_T_b = d_T[:, None]
            u_b = u[:, None]
            d_u_b = d_u[:, None]
            c = df.grid.v[None, :] - u_b
            c_sq = c**2
            term1 = d_rho_b / rho_b
            term2 = (c * d_u_b) / (self.R * T_b)
            term3 = (c_sq / (2.0 * self.R * T_b**2) - cv_fac / T_b) * d_T_b
        else:
            u_x = macros[1]
            u_y = macros[2]
            d_rhoux = dw[1]
            d_rhouy = dw[2]
            d_E = dw[3]
            d_ux = (d_rhoux - u_x * d_rho) / rho
            d_uy = (d_rhouy - u_y * d_rho) / rho
            d_T = (
                d_E
                - cv_fac * self.R * T * d_rho
                - 0.5 * (u_x**2 + u_y**2) * d_rho
                - rho * u_x * d_ux
                - rho * u_y * d_uy
            ) / (rho * cv_fac * self.R)
            rho_b = rho[:, None, None]
            T_b = T[:, None, None]
            d_rho_b = d_rho[:, None, None]
            d_T_b = d_T[:, None, None]
            ux_b = u_x[:, None, None]
            uy_b = u_y[:, None, None]
            d_ux_b = d_ux[:, None, None]
            d_uy_b = d_uy[:, None, None]
            cx = df.grid.vx_mesh[None, :, :] - ux_b
            cy = df.grid.vy_mesh[None, :, :] - uy_b
            c_sq = cx**2 + cy**2
            term1 = d_rho_b / rho_b
            term2 = (cx * d_ux_b + cy * d_uy_b) / (self.R * T_b)
            term3 = (c_sq / (2.0 * self.R * T_b**2) - cv_fac / T_b) * d_T_b

        return g * (term1 + term2 + term3)

    def _compute_macroscopic_flux(self, df, f_hat):
        v_adv = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_adv_b = v_adv[None, ...]
        f_vx = f_hat * v_adv_b
        w = df.compute_moments(f_input=f_vx)
        if w.shape[0] != df.dim_v + 2:
            w = w.T
        return w

    def _update_macroscopics(self, w_inner, macro_flux_phys, dt, dx):
        return w_inner - (dt / dx) * (macro_flux_phys[:, 1:] - macro_flux_phys[:, :-1])

    def _final_microscopic_update(
        self, df, f_n, f_hat_plus, f_hat_minus, g_n, g_n1, dt, dx, tau_n, tau_n1
    ):
        v_adv = df.grid.v if df.dim_v == 1 else df.grid.vx_mesh
        v_adv_b = v_adv[None, ...]
        transport = -(dt / dx) * v_adv_b * (f_hat_plus - f_hat_minus)
        collision = (dt / 2.0) * ((g_n1 - f_n) / tau_n1 + (g_n - f_n) / tau_n)
        return f_n + (2.0 * tau_n1) / (dt + 2.0 * tau_n1) * (transport + collision)

    def _pad_with_ghosts(self, df: DistributionFunction, pad=2):
        f_inner = df.f
        Nx = f_inner.shape[0]
        v_shape = f_inner.shape[1:]
        f = np.zeros((Nx + 2 * pad,) + v_shape)
        f[pad:-pad, ...] = f_inner

        if self.bc_type == "periodic":
            f[:pad, ...] = f_inner[-pad:, ...]
            f[-pad:, ...] = f_inner[:pad, ...]

        elif self.bc_type == "specular":
            for k in range(pad):
                f[k, ...] = f_inner[pad - k - 1, ::-1, ...]
                f[-k - 1, ...] = f_inner[-(pad - k), ::-1, ...]

        elif self.bc_type == "diffusive":
            dim_v = df.dim_v
            TL = self.config.physics.T_L
            TR = self.config.physics.T_R
            VL = self.config.physics.u_L
            VR = self.config.physics.u_R
            v_adv = df.grid.v if dim_v == 1 else df.grid.vx_mesh

            if dim_v == 1:
                g_unit_L = (
                    1.0
                    / np.sqrt(2.0 * np.pi * self.R * TL)
                    * np.exp(-((df.grid.v - VL) ** 2) / (2.0 * self.R * TL))
                )
                g_unit_R = (
                    1.0
                    / np.sqrt(2.0 * np.pi * self.R * TR)
                    * np.exp(-((df.grid.v - VR) ** 2) / (2.0 * self.R * TR))
                )
            else:
                g_unit_L = (
                    1.0
                    / (2.0 * np.pi * self.R * TL)
                    * np.exp(
                        -(df.grid.vx_mesh**2 + (df.grid.vy_mesh - VL) ** 2)
                        / (2.0 * self.R * TL)
                    )
                )
                g_unit_R = (
                    1.0
                    / (2.0 * np.pi * self.R * TR)
                    * np.exp(
                        -(df.grid.vx_mesh**2 + (df.grid.vy_mesh - VR) ** 2)
                        / (2.0 * self.R * TR)
                    )
                )

            flux_in_L = df._integrate(np.where(v_adv < 0, v_adv * f_inner[0], 0.0))
            flux_out_unit_L = df._integrate(np.where(v_adv > 0, v_adv * g_unit_L, 0.0))
            rho_w_L = -flux_in_L / (flux_out_unit_L + 1e-15)

            flux_in_R = df._integrate(np.where(v_adv > 0, v_adv * f_inner[-1], 0.0))
            flux_out_unit_R = df._integrate(np.where(v_adv < 0, v_adv * g_unit_R, 0.0))
            rho_w_R = -flux_in_R / (flux_out_unit_R + 1e-15)

            for k in range(pad):
                f[k, ...] = rho_w_L * g_unit_L
                f[-k - 1, ...] = rho_w_R * g_unit_R

        return f


class LinearUGKSSolver(Solver):
    def __init__(self, config, u_bg: np.ndarray, T_bg: np.ndarray, rho_bg=1.0):
        self.config = config
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.bc_type = config.grid.bc_type
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)

        self.u_bg_inner = u_bg
        self.T_bg_inner = T_bg
        self.rho_bg_inner = rho_bg

        self._is_initialized = False
        self.reflectance_left = (
            float(config.physics.reflectance_left)
            if self.bc_type == "inflow/outflow"
            else None
        )

    def _init_background(self, df: DistributionFunction):
        if self._is_initialized:
            return

        pad = 2
        v = df.grid.v
        dx = df.grid.dx

        self.u_bg = self._pad_macro(self.u_bg_inner, pad)
        self.T_bg = self._pad_macro(self.T_bg_inner, pad)

        if isinstance(self.rho_bg_inner, np.ndarray):
            self.rho_bg = self._pad_macro(self.rho_bg_inner, pad)
        else:
            self.rho_bg = self.rho_bg_inner

        self.tau_cells = self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / self.rho_bg
        self.tau_cells = self.tau_cells.reshape(-1, 1)

        self.M_bg_cells = self._get_normalized_maxwellian(self.u_bg, self.T_bg, v)

        self.u_bg_face = 0.5 * (self.u_bg[:-1] + self.u_bg[1:])
        self.T_bg_face = 0.5 * (self.T_bg[:-1] + self.T_bg[1:])

        if isinstance(self.rho_bg, np.ndarray):
            self.rho_bg_face = 0.5 * (self.rho_bg[:-1] + self.rho_bg[1:])
        else:
            self.rho_bg_face = self.rho_bg

        self.tau_face = (
            self.tau_ref * (self.T_bg_face ** (self.omega - 1.0)) / self.rho_bg_face
        )
        self.tau_face = self.tau_face.reshape(-1, 1)

        self.M_bg_face = self._get_normalized_maxwellian(
            self.u_bg_face, self.T_bg_face, v
        )

        du_dx_face = (self.u_bg[1:] - self.u_bg[:-1]) / dx
        dT_dx_face = (self.T_bg[1:] - self.T_bg[:-1]) / dx

        c = v[None, :] - self.u_bg_face[:, None]
        RT = self.R * self.T_bg_face[:, None]

        dM_du = self.M_bg_face * (c / RT)
        dM_dT = self.M_bg_face * (
            c**2 / (2.0 * RT * self.T_bg_face[:, None]) - 0.5 / self.T_bg_face[:, None]
        )

        self.dMbg_dx_face = dM_du * du_dx_face[:, None] + dM_dT * dT_dx_face[:, None]
        self._is_initialized = True

    def _pad_macro(self, w_inner, pad):
        w = np.zeros(len(w_inner) + 2 * pad)
        w[pad:-pad] = w_inner
        if self.bc_type == "periodic":
            w[:pad] = w_inner[-pad:]
            w[-pad:] = w_inner[:pad]
        elif self.bc_type == "specular":
            for k in range(pad):
                w[k] = w_inner[pad - k - 1]
                w[-k - 1] = w_inner[-(pad - k)]
        elif self.bc_type == "inflow/outflow":
            # zero-gradient extrapolation
            w[:pad] = w_inner[0]
            w[-pad:] = w_inner[-1]
        return w

    def _get_normalized_maxwellian(self, u, T, v):
        pref = 1.0 / np.sqrt(2.0 * np.pi * self.R * T)
        v_shifted = v[None, :] - u[:, None]
        return pref[:, None] * np.exp(-(v_shifted**2) / (2.0 * self.R * T[:, None]))

    def step(
        self, df: DistributionFunction, dt: float, t_n=None, source_func=None
    ) -> DistributionFunction:
        self._init_background(df)
        dx = df.grid.dx
        v = df.grid.v
        dv = df.grid.dv
        pad = 2

        f_padded = self._pad_with_ghosts(df, pad=pad)

        f_face, df_face, f_L, f_R = self._compute_interface_reconstruction(
            f_padded, dx, df
        )

        rho_face = self._compute_interface_rho(f_L, f_R, v, dv)
        rho_cells = np.sum(f_padded, axis=1) * dv

        g_face = rho_face[:, None] * self.M_bg_face

        drho_dx_left, drho_dx_right = self._compute_drho_dx(rho_cells, rho_face, dx)
        drho_dt_face = self._compute_drho_dt(v, df_face, dv)

        v_pos = v > 0
        drho_dx_upwind = np.where(
            v_pos[None, :], drho_dx_left[:, None], drho_dx_right[:, None]
        )

        dgx_face = (
            self.M_bg_face * drho_dx_upwind + rho_face[:, None] * self.dMbg_dx_face
        )
        dgt_face = self.M_bg_face * drho_dt_face[:, None]

        f_hat = self._compute_time_averaged_f(
            f_face, df_face, g_face, dgx_face, dgt_face, v, dt, self.tau_face
        )
        if self.bc_type == "inflow/outflow":
            v_pos = v > 0
            v_neg = v < 0

            f_hat[pad - 1, v_pos] = f_L[pad - 1, v_pos]
            f_hat[pad - 1, v_neg] = f_R[pad - 1, v_neg]

            f_hat[-pad, v_pos] = f_L[-pad, v_pos]
            f_hat[-pad, v_neg] = f_R[-pad, v_neg]
        rho_flux = self._compute_rho_flux(f_hat, v, dv)
        rho_n1 = self._update_rho(rho_cells, rho_flux, dt, dx)

        g_n = rho_cells[:, None] * self.M_bg_cells
        g_n1 = rho_n1[:, None] * self.M_bg_cells

        f_new_inner = self._final_microscopic_update(
            f_n=f_padded[1:-1, :],
            f_hat_plus=f_hat[1:, :],
            f_hat_minus=f_hat[:-1, :],
            g_n=g_n[1:-1, :],
            g_n1=g_n1[1:-1, :],
            v=v,
            dt=dt,
            dx=dx,
            tau_n=self.tau_cells[1:-1, :],
            tau_n1=self.tau_cells[1:-1, :],
        )

        df.f = f_new_inner[pad - 1 : -pad + 1, :]

        return df

    def _compute_interface_rho(self, f_left, f_right, v, dv):
        v_pos, v_neg = v > 0, v < 0
        rho_pos = np.sum(f_left[:, v_pos], axis=-1) * dv
        rho_neg = np.sum(f_right[:, v_neg], axis=-1) * dv
        return rho_pos + rho_neg

    def _compute_drho_dx(self, rho_cells, rho_face, dx):
        drho_dx_left = 2.0 * (rho_face - rho_cells[:-1]) / dx
        drho_dx_right = 2.0 * (rho_cells[1:] - rho_face) / dx
        return drho_dx_left, drho_dx_right

    def _compute_drho_dt(self, v, df_face, dv):
        return -np.sum(v * df_face, axis=-1) * dv

    def _compute_rho_flux(self, f_hat, v, dv):
        return np.sum(v * f_hat, axis=-1) * dv

    def _update_rho(self, rho_n, rho_flux, dt, dx):
        rho_n1 = rho_n.copy()
        rho_n1[1:-1] = rho_n[1:-1] - (dt / dx) * (rho_flux[1:] - rho_flux[:-1])
        return rho_n1

    def _get_slopes(self, f: np.ndarray, dx: float) -> np.ndarray:
        s1 = (f[1:-1, :] - f[0:-2, :]) / dx
        s2 = (f[2:, :] - f[1:-1, :]) / dx
        eps = 1e-15
        slopes_inner = (
            (np.sign(s1) + np.sign(s2))
            * (np.abs(s1) * np.abs(s2))
            / (np.abs(s1) + np.abs(s2) + eps)
        )
        slopes = np.zeros_like(f)
        slopes[1:-1, :] = slopes_inner
        return slopes

    def _compute_interface_reconstruction(self, f, dx, df_obj):
        slopes = self._get_slopes(f, dx)

        if self.bc_type == "diffusive":
            pad = 2
            Nx = df_obj.f.shape[0]
            slopes[pad, ...] = (f[pad + 1, ...] - f[pad, ...]) / dx
            slopes[pad + Nx - 1, ...] = (
                f[pad + Nx - 1, ...] - f[pad + Nx - 2, ...]
            ) / dx

        f_left_limit = f[:-1, ...] + 0.5 * dx * slopes[:-1, ...]
        f_right_limit = f[1:, ...] - 0.5 * dx * slopes[1:, ...]
        vx = df_obj.grid.v if df_obj.dim_v == 1 else df_obj.grid.vx_mesh
        v_pos = vx[None, ...] > 0
        f_face = np.where(v_pos, f_left_limit, f_right_limit)
        df_face = np.where(v_pos, slopes[:-1, ...], slopes[1:, ...])
        return f_face, df_face, f_left_limit, f_right_limit

    def _get_time_averaged_coeffs(self, dt: float, tau):
        exp_at = np.exp(-dt / tau)
        C1 = (tau / dt) * (1.0 - exp_at)
        C2 = dt * ((tau / dt) * exp_at - (tau**2 / dt**2) * (1.0 - exp_at))
        C3 = 1.0 - (tau / dt) * (1.0 - exp_at)
        C4 = dt * (
            (2.0 * tau**2 / dt**2) * (1.0 - exp_at) - (tau / dt) - (tau / dt) * exp_at
        )
        C5 = dt * (0.5 - (tau / dt) + (tau**2 / dt**2) * (1.0 - exp_at))
        return C1, C2, C3, C4, C5

    def _compute_time_averaged_f(
        self, f_face, df_face, g_face, dgx_face, dgt_face, v, dt, tau_face
    ):
        C1, C2, C3, C4, C5 = self._get_time_averaged_coeffs(dt, tau_face)
        f_hat_fr = C1 * f_face + C2 * df_face * v
        f_hat_eq = C3 * g_face + C4 * dgx_face * v + C5 * dgt_face
        return f_hat_fr + f_hat_eq

    def _final_microscopic_update(
        self, f_n, f_hat_plus, f_hat_minus, g_n, g_n1, v, dt, dx, tau_n, tau_n1
    ):
        transport = -(dt / dx) * v * (f_hat_plus - f_hat_minus)
        collision = (dt / 2.0) * ((g_n1 - f_n) / tau_n1 + (g_n - f_n) / tau_n)
        return f_n + (2.0 * tau_n1) / (dt + 2.0 * tau_n1) * (transport + collision)

    def _pad_with_ghosts(self, df, pad=2):
        f_inner = df.f
        v = df.grid.v

        Nx, Nv = f_inner.shape
        f = np.zeros((Nx + 2 * pad, Nv))
        f[pad:-pad] = f_inner

        if self.bc_type == "periodic":
            f[:pad] = f_inner[-pad:]
            f[-pad:] = f_inner[:pad]

        elif self.bc_type == "specular":
            for k in range(pad):
                f[k] = f_inner[pad - k - 1, ::-1]
                f[-k - 1] = f_inner[-(pad - k), ::-1]

        elif self.bc_type == "inflow/outflow":
            v_pos = v > 0
            v_neg = v < 0
            R_n = self.reflectance_left

            f_reflected_L = f_inner[0, ::-1]
            if hasattr(df, "f_flow_left"):
                f[:pad, v_pos] = df.f_flow_left[v_pos] + R_n * f_reflected_L[v_pos]
            else:
                f[:pad, v_pos] = R_n * f_reflected_L[v_pos]
            f[:pad, v_neg] = f_inner[0, v_neg]

            f_reflected_R = f_inner[-1, ::-1]
            if hasattr(df, "f_flow_right"):
                f[-pad:, v_neg] = df.f_flow_right[v_neg] + R_n * f_reflected_R[v_neg]
            else:
                f[-pad:, v_neg] = 0.0  # vacuum
            f[-pad:, v_pos] = f_inner[-1, v_pos]

        return f
