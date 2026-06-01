import numpy as np
from bgk.core.distribution import DistributionFunction
from bgk.solvers.base import Solver


class SLSolver(Solver):
    def __init__(self, config):
        super().__init__(config)

    def step(self, df: DistributionFunction, dt: float):
        if self.bc_type == "diffusive":
            bc_K = "open"
        else:
            bc_K = self.bc_type

        alpha = 1.0 - np.sqrt(2) / 2.0
        a11 = alpha
        a21 = 1.0 - alpha
        a22 = alpha
        c1 = alpha
        c2 = 1.0

        fn = df.f.copy()

        f1n = df.advect(
            dt=c1 * dt,
            bc_type=self.bc_type,
            f_input=fn,
            config=self.config,
            mass_correction=True,
        )

        macros1 = df.compute_macroscopics(f_input=f1n)
        rho1 = macros1[0]
        T1 = macros1[-1]
        tau11 = self._compute_tau(rho1, T1)
        M11 = df.maxwellian(f_input=f1n, macros=macros1)

        lam11 = (a11 * dt) / tau11
        F11 = (f1n + lam11 * M11) / (1.0 + lam11)

        f2n = df.advect(
            dt=dt,
            bc_type=self.bc_type,
            f_input=fn,
            config=self.config,
            mass_correction=True,
        )

        if self.bc_type == "diffusive":
            F21 = df.advect(
                dt=(c2 - c1) * dt,
                bc_type=self.bc_type,
                f_input=F11,
                config=self.config,
                lagrangian=False,
                mass_correction=True,
            )

            K21 = (F21 - f2n) / (a11 * dt)
        else:
            K11 = (F11 - f1n) / (a11 * dt)
            K21 = df.advect(
                dt=(c2 - c1) * dt,
                bc_type=bc_K,
                f_input=K11,
                config=self.config,
            )

        f_tilde = f2n + dt * a21 * K21

        macros2 = df.compute_macroscopics(f_input=f_tilde)
        rho2 = macros2[0]
        T2 = macros2[-1]
        tau22 = self._compute_tau(rho2, T2)
        M22 = df.maxwellian(f_input=f_tilde, macros=macros2)

        lam22 = (a22 * dt) / tau22
        f_new = (f_tilde + lam22 * M22) / (1.0 + lam22)

        df.f = f_new

        return df


class LinearSLSolver(Solver):
    def __init__(self, config, u_bg: np.ndarray, T_bg: np.ndarray, rho_bg=1.0):
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)
        self.bc_type = config.grid.bc_type

        self.u_bg = u_bg
        self.T_bg = T_bg
        self.config = config

        self.tau = self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / rho_bg
        self.tau = self.tau.reshape(-1, 1)

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

        alpha = 1.0 - np.sqrt(2) / 2.0
        a11 = alpha
        a21 = 1.0 - alpha
        a22 = alpha
        c1 = alpha
        c2 = 1.0

        fn = df.f.copy()

        f1n = df.advect(dt=c1 * dt, bc_type=self.bc_type, f_input=fn)

        rho1 = (np.sum(f1n, axis=1) * dv).reshape(-1, 1)
        M11 = rho1 * self._M_normalized

        lam11 = (a11 * dt) / self.tau
        F11 = (f1n + lam11 * M11) / (1.0 + lam11)

        K11 = (F11 - f1n) / (a11 * dt)

        f2n = df.advect(dt=dt, bc_type=self.bc_type, f_input=fn)

        K21 = df.advect(dt=(c2 - c1) * dt, bc_type=self.bc_type, f_input=K11)

        f_tilde = f2n + dt * a21 * K21

        rho2 = (np.sum(f_tilde, axis=1) * dv).reshape(-1, 1)
        M22 = rho2 * self._M_normalized

        lam22 = (a22 * dt) / self.tau
        f_new = (f_tilde + lam22 * M22) / (1.0 + lam22)

        df.f = f_new
        return df
