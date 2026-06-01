import numpy as np
from bgk.core.distribution import DistributionFunction
from bgk.solvers.base import Solver


class HybridSolver(Solver):
    def __init__(self, config):
        super().__init__(config)
        alpha = 1.0 - np.sqrt(2.0) / 2.0
        self.a11 = alpha
        self.a21 = 1.0 - alpha
        self.a22 = alpha
        self.c1 = alpha

    def step(self, df: DistributionFunction, dt: float) -> DistributionFunction:
        if self.bc_type == "diffusive":
            bc_K = "open"
        else:
            bc_K = self.bc_type
        a11 = self.a11
        a21 = self.a21
        c1 = self.c1

        f_n = df.f.copy()

        macros_n = df.compute_macroscopics(f_input=f_n, R=self.R)
        g_n = df.maxwellian(macros=macros_n, R=self.R)

        f1 = df.advect(
            dt=c1 * dt,
            bc_type=self.bc_type,
            f_input=f_n,
            config=self.config,
        )
        macros1 = df.compute_macroscopics(f_input=f1, R=self.R)
        tau1 = self._compute_tau(macros1[0], macros1[-1])
        g1 = df.maxwellian(macros=macros1, R=self.R)

        lam1 = a11 * dt / tau1
        F1 = (f1 + lam1 * g1) / (1.0 + lam1)

        f2 = df.advect(
            dt=dt,
            bc_type=self.bc_type,
            f_input=f_n,
            config=self.config,
            mass_correction=True,
        )
        if self.bc_type == "diffusive":
            F1_star = df.advect(
                dt=(1.0 - c1) * dt,
                bc_type=self.bc_type,
                f_input=F1,
                config=self.config,
                mass_correction=True,
            )

            K1_star = (F1_star - f2) / (a11 * dt)
        else:
            K1 = (F1 - f1) / (a11 * dt)

            K1_star = df.advect(
                dt=(1.0 - c1) * dt,
                bc_type=bc_K,
                f_input=K1,
                config=self.config,
            )

        f_tilde = f2 + dt * a21 * K1_star

        macros2 = df.compute_macroscopics(f_input=f_tilde, R=self.R)
        g_n1 = df.maxwellian(macros=macros2, R=self.R)

        tau = self._compute_tau(macros2[0], macros2[-1])

        g_star = df.advect(
            dt=dt,
            bc_type=self.bc_type,
            f_input=g_n,
            config=self.config,
            mass_correction=True,
        )

        exp_t = np.exp(-dt / tau)
        tau_over_dt = tau / dt
        one_minus_e = 1.0 - exp_t

        C_n = tau_over_dt * one_minus_e - exp_t
        C_n1 = 1.0 - tau_over_dt * one_minus_e

        f_new = exp_t * f2 + C_n * g_star + C_n1 * g_n1

        df.f = f_new
        return df


class LinearHybridSolver(Solver):
    def __init__(self, config, u_bg: np.ndarray, T_bg: np.ndarray, rho_bg=1.0):
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)
        self.bc_type = config.grid.bc_type

        self.u_bg = u_bg
        self.T_bg = T_bg

        self.tau = self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / rho_bg
        self.tau = self.tau.reshape(-1, 1)

        alpha = 1.0 - np.sqrt(2.0) / 2.0
        self.a11 = alpha
        self.a21 = 1.0 - alpha
        self.a22 = alpha
        self.c1 = alpha

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
        self,
        df: DistributionFunction,
        dt: float,
    ):
        self._init_background(df)
        dv = df.grid.dv

        if hasattr(self, "f_flow_left"):
            df.f_flow_left = self.f_flow_left
        if hasattr(self, "f_flow_right"):
            df.f_flow_right = self.f_flow_right

        a11 = self.a11
        a21 = self.a21
        c1 = self.c1

        f_n = df.f.copy()

        rho_n = (np.sum(f_n, axis=1) * dv).reshape(-1, 1)
        g_n = rho_n * self._M_normalized

        f1 = df.advect(dt=c1 * dt, bc_type=self.bc_type, f_input=f_n)
        rho1 = (np.sum(f1, axis=1) * dv).reshape(-1, 1)
        g1 = rho1 * self._M_normalized

        lam1 = a11 * dt / self.tau
        F1 = (f1 + lam1 * g1) / (1.0 + lam1)
        K1 = (g1 - F1) / self.tau

        f2 = df.advect(dt=dt, bc_type=self.bc_type, f_input=f_n)

        K1_star = df.advect(dt=(1.0 - c1) * dt, bc_type=self.bc_type, f_input=K1)

        f_tilde = f2 + dt * a21 * K1_star

        rho2 = (np.sum(f_tilde, axis=1) * dv).reshape(-1, 1)
        g_n1 = rho2 * self._M_normalized

        g_star = df.advect(dt=dt, bc_type=self.bc_type, f_input=g_n)

        exp_t = np.exp(-dt / self.tau)
        tau_over_dt = self.tau / dt
        one_minus_e = 1.0 - exp_t

        C_n = tau_over_dt * one_minus_e - exp_t
        C_n1 = 1.0 - tau_over_dt * one_minus_e

        f_new = exp_t * f2 + C_n * g_star + C_n1 * g_n1

        df.f = f_new
        return df
