import numpy as np
from bgk.core.distribution import DistributionFunction
from bgk.solvers.base import Solver


class ButcherTableau:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        self.A = A
        self.b = b
        self.c = c


class FVMSolver(Solver):
    def __init__(self, config):
        super().__init__(config)

        delta = 1.0 - np.sqrt(0.5)
        zeta = 1.0 - 1.0 / (2.0 * delta)

        A_exp = np.array(
            [
                [0.0, 0.0, 0.0],
                [delta, 0.0, 0.0],
                [zeta, 1.0 - zeta, 0.0],
            ]
        )
        b_exp = np.array([zeta, 1.0 - zeta, 0.0])
        c_exp = np.array([0.0, delta, 1.0])
        self.bt_exp = ButcherTableau(A_exp, b_exp, c_exp)

        A_imp = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, delta, 0.0],
                [0.0, 1.0 - delta, delta],
            ]
        )
        b_imp = np.array([0.0, 1.0 - delta, delta])
        c_imp = np.array([0.0, delta, 1.0])
        self.bt_imp = ButcherTableau(A_imp, b_imp, c_imp)

    def step(
        self, df: DistributionFunction, dt: float, **kwargs
    ) -> DistributionFunction:
        f_n = df.f.copy()

        F1 = f_n.copy()

        f1_padded = self._build_padded(df, F1)
        Kt1 = self._Kt_from_padded(df, F1, f1_padded)

        X2 = f_n + dt * self.bt_exp.A[1, 0] * Kt1

        macros2 = df.compute_macroscopics(f_input=X2)
        rho2 = macros2[0]
        T2 = macros2[-1]
        tau2 = self._compute_tau(rho2, T2)
        M2 = df.maxwellian(f_input=X2)

        F2 = (X2 + dt * self.bt_imp.A[1, 1] * M2 / tau2) / (
            1.0 + dt * self.bt_imp.A[1, 1] / tau2
        )

        f2_padded = self._build_padded(df, F2)
        Kt2 = self._Kt_from_padded(df, F2, f2_padded)
        Kc2 = (M2 - F2) / tau2

        X3_partial = (
            f_n + dt * self.bt_exp.A[2, 0] * Kt1 + dt * self.bt_exp.A[2, 1] * Kt2
        )

        macros3 = df.compute_macroscopics(f_input=X3_partial)
        rho3 = macros3[0]
        T3 = macros3[-1]
        tau3 = self._compute_tau(rho3, T3)
        M3 = df.maxwellian(f_input=X3_partial)

        X3_full = X3_partial + dt * self.bt_imp.A[2, 1] * Kc2

        F3 = (X3_full + dt * self.bt_imp.A[2, 2] * M3 / tau3) / (
            1.0 + dt * self.bt_imp.A[2, 2] / tau3
        )

        Kc3 = (M3 - F3) / tau3

        f_new = (
            f_n
            + dt * self.bt_exp.b[0] * Kt1
            + dt * self.bt_exp.b[1] * Kt2
            + dt * self.bt_imp.b[1] * Kc2
            + dt * self.bt_imp.b[2] * Kc3
        )

        df.f = f_new
        return df

    def _build_padded(self, df: DistributionFunction, f: np.ndarray) -> np.ndarray:
        Nx = f.shape[0]
        v_shape = f.shape[1:]
        pad = 2

        f_padded = np.zeros((Nx + 2 * pad,) + v_shape)
        f_padded[pad:-pad, ...] = f

        if self.bc_type == "periodic":
            f_padded[:pad, ...] = f[-pad:, ...]
            f_padded[-pad:, ...] = f[:pad, ...]

        elif self.bc_type == "specular":
            f_v_flip = f[:, ::-1, ...]
            f_padded[:pad, ...] = f_v_flip[:pad, ...][::-1, ...]
            f_padded[-pad:, ...] = f_v_flip[-pad:, ...][::-1, ...]

        elif self.bc_type == "inflow/outflow":
            v_pos = df.grid.v > 0
            v_neg = df.grid.v < 0
            R_n = self.config.physics.reflectance_left

            # left boundary
            f_reflected_L = f[0, ::-1]
            if hasattr(df, "f_flow_left"):
                f_padded[:pad, v_pos] = (
                    df.f_flow_left[v_pos] + R_n * f_reflected_L[v_pos]
                )
            else:
                f_padded[:pad, v_pos] = R_n * f_reflected_L[v_pos]
            f_padded[:pad, v_neg] = f[0, v_neg]

            # right boundary
            if hasattr(df, "f_flow_right"):
                f_padded[-pad:, v_neg] = df.f_flow_right[v_neg]
            else:
                f_padded[-pad:, v_neg] = 0.0
            f_padded[-pad:, v_pos] = f[-1, v_pos]

        elif self.bc_type == "diffusive":
            for k in range(pad):
                f_padded[k, ...] = f[0, ...]
                f_padded[-k - 1, ...] = f[-1, ...]

        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")

        return f_padded

    def _unit_maxwellian_1d(self, df, T_wall, V_wall):
        pref = 1.0 / np.sqrt(2.0 * np.pi * self.R * T_wall)
        return pref * np.exp(-((df.grid.v - V_wall) ** 2) / (2.0 * self.R * T_wall))

    def _unit_maxwellian_2d(self, df, T_wall, V_wall):
        pref = 1.0 / (2.0 * np.pi * self.R * T_wall)
        c_sq = df.grid.vx_mesh**2 + (df.grid.vy_mesh - V_wall) ** 2
        return pref * np.exp(-c_sq / (2.0 * self.R * T_wall))

    def _wall_face_flux(self, df, f_boundary, T_wall, V_wall, is_left):
        dim_v = df.dim_v
        v_adv = df.grid.v if dim_v == 1 else df.grid.vx_mesh

        if dim_v == 1:
            pref = 1.0 / np.sqrt(2.0 * np.pi * self.R * T_wall)
            g_unit = pref * np.exp(
                -((df.grid.v - V_wall) ** 2) / (2.0 * self.R * T_wall)
            )
        else:
            pref = 1.0 / (2.0 * np.pi * self.R * T_wall)
            g_unit = pref * np.exp(
                -(df.grid.vx_mesh**2 + (df.grid.vy_mesh - V_wall) ** 2)
                / (2.0 * self.R * T_wall)
            )

        if is_left:
            inc_mask = v_adv < 0
            out_mask = v_adv > 0
        else:
            inc_mask = v_adv > 0
            out_mask = v_adv < 0

        flux_in = df._integrate(np.where(inc_mask, v_adv * f_boundary, 0.0))
        flux_out_unit = df._integrate(np.where(out_mask, v_adv * g_unit, 0.0))
        rho_w = -flux_in / (flux_out_unit + 1e-15)

        return np.where(inc_mask, f_boundary, rho_w * g_unit)

    def _Kt_from_padded(
        self,
        df: DistributionFunction,
        f_stage: np.ndarray,
        f_n_padded: np.ndarray,
    ) -> np.ndarray:
        Nx = f_stage.shape[0]
        dim_v = df.dim_v
        dx = df.grid.dx
        pad = 2

        v_adv = df.grid.v if dim_v == 1 else df.grid.vx_mesh
        v_b = v_adv[None, ...]

        f_padded = f_n_padded.copy()
        f_padded[pad:-pad, ...] = f_stage

        s1 = (f_padded[1:-1, ...] - f_padded[:-2, ...]) / dx
        s2 = (f_padded[2:, ...] - f_padded[1:-1, ...]) / dx
        eps = 1e-15
        slope = (
            (np.sign(s1) + np.sign(s2))
            * np.abs(s1)
            * np.abs(s2)
            / (np.abs(s1) + np.abs(s2) + eps)
        )

        if self.bc_type == "diffusive":
            slope[pad - 1, ...] = (f_stage[1, ...] - f_stage[0, ...]) / dx
            slope[pad + Nx - 2, ...] = (f_stage[-1, ...] - f_stage[-2, ...]) / dx

        f_all = f_padded[1:-1, ...]
        f_L_all = f_all + 0.5 * dx * slope
        f_R_all = f_all - 0.5 * dx * slope
        face_pos = np.zeros((Nx + 1,) + f_stage.shape[1:])
        face_neg = np.zeros((Nx + 1,) + f_stage.shape[1:])

        for k in range(1, Nx):
            face_pos[k] = f_L_all[(k - 1) + pad - 1]
            face_neg[k] = f_R_all[k + pad - 1]

        if self.bc_type == "diffusive":
            TL, TR = self.config.physics.T_L, self.config.physics.T_R
            VL, VR = self.config.physics.u_L, self.config.physics.u_R

            f_incident_L = f_R_all[pad - 1]
            f_incident_R = f_L_all[pad + Nx - 2]

            f_face_L = self._wall_face_flux(df, f_incident_L, TL, VL, is_left=True)
            f_face_R = self._wall_face_flux(df, f_incident_R, TR, VR, is_left=False)

            face_pos[0], face_neg[0] = (
                np.where(v_adv > 0, f_face_L, 0.0),
                np.where(v_adv < 0, f_face_L, 0.0),
            )
            face_pos[Nx], face_neg[Nx] = (
                np.where(v_adv > 0, f_face_R, 0.0),
                np.where(v_adv < 0, f_face_R, 0.0),
            )

        else:
            face_pos[0] = f_L_all[pad - 2]
            face_neg[0] = f_R_all[pad - 1]
            face_pos[Nx] = f_L_all[Nx + pad - 2]
            face_neg[Nx] = f_R_all[Nx + pad - 1]

        F_right_pos = face_pos[1:, ...]
        F_right_neg = face_neg[1:, ...]
        F_left_pos = face_pos[:-1, ...]
        F_left_neg = face_neg[:-1, ...]

        Kt = (
            -(
                v_b
                * (
                    np.where(v_b > 0, F_right_pos - F_left_pos, 0.0)
                    + np.where(v_b < 0, F_right_neg - F_left_neg, 0.0)
                )
            )
            / dx
        )

        return Kt

    def Kt_flux(self, df: DistributionFunction) -> np.ndarray:
        f_padded = self._build_padded(df, df.f)
        return self._Kt_from_padded(df, df.f, f_padded)


class LinearFVMSolver(FVMSolver):
    def __init__(self, config, u_bg: np.ndarray, T_bg: np.ndarray, rho_bg: np.ndarray):
        super().__init__(config)
        self.u_bg = u_bg
        self.T_bg = T_bg

        self.tau = (self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / rho_bg).reshape(
            -1, 1
        )

        self._M_normalized = None

    def _init_background(self, df: DistributionFunction):
        if self._M_normalized is None:
            v = df.grid.v
            pref = 1.0 / np.sqrt(2.0 * np.pi * self.R * self.T_bg)
            v_shifted = v[None, :] - self.u_bg[:, None]
            self._M_normalized = pref[:, None] * np.exp(
                -(v_shifted**2) / (2.0 * self.R * self.T_bg[:, None])
            )

    def step(
        self, df: DistributionFunction, dt: float, t_n: float = 0.0, source_func=None
    ) -> DistributionFunction:
        self._init_background(df)
        dv = df.grid.dv
        f_n = df.f.copy()

        F1 = f_n.copy()
        f1_padded = self._build_padded(df, F1)
        Kt1 = self._Kt_from_padded(df, F1, f1_padded)

        X2 = f_n + dt * self.bt_exp.A[1, 0] * Kt1

        rho2 = (np.sum(X2, axis=1) * dv).reshape(-1, 1)
        M2 = rho2 * self._M_normalized

        F2 = (X2 + dt * self.bt_imp.A[1, 1] * M2 / self.tau) / (
            1.0 + dt * self.bt_imp.A[1, 1] / self.tau
        )

        f2_padded = self._build_padded(df, F2)
        Kt2 = self._Kt_from_padded(df, F2, f2_padded)
        Kc2 = (M2 - F2) / self.tau

        X3 = (
            f_n
            + dt * self.bt_exp.A[2, 0] * Kt1
            + dt * self.bt_exp.A[2, 1] * Kt2
            + dt * self.bt_imp.A[2, 1] * Kc2
        )

        rho3 = (np.sum(X3, axis=1) * dv).reshape(-1, 1)
        M3 = rho3 * self._M_normalized

        F3 = (X3 + dt * self.bt_imp.A[2, 2] * M3 / self.tau) / (
            1.0 + dt * self.bt_imp.A[2, 2] / self.tau
        )
        Kc3 = (M3 - F3) / self.tau

        f_new = (
            f_n
            + dt * self.bt_exp.b[0] * Kt1
            + dt * self.bt_exp.b[1] * Kt2
            + dt * self.bt_imp.b[1] * Kc2
            + dt * self.bt_imp.b[2] * Kc3
        )
        df.f = f_new
        return df
