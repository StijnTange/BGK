import numpy as np
from bgk.core.distribution import DistributionFunction
from bgk.solvers.base import Solver


class StrangSolver(Solver):
    def __init__(self, config):
        super().__init__(config)

    def step(self, df: DistributionFunction, dt: float):
        macros = df.compute_macroscopics()
        rho = macros[0]
        T = macros[-1]
        tau_local = self._compute_tau(rho, T)

        M = df.maxwellian()
        df.f = (
            np.exp(-dt / 2.0 / tau_local) * df.f
            + (1.0 - np.exp(-dt / 2.0 / tau_local)) * M
        )

        df.advect(dt, self.bc_type, config=self.config, mass_correction=True)

        macros = df.compute_macroscopics()
        rho = macros[0]
        T = macros[-1]
        tau_local = self._compute_tau(rho, T)

        M = df.maxwellian()
        df.f = (
            np.exp(-dt / 2.0 / tau_local) * df.f
            + (1.0 - np.exp(-dt / 2.0 / tau_local)) * M
        )

        return df


class LinearStrangSolver(Solver):
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

        if hasattr(self, "f_flow_left"):
            df.f_flow_left = self.f_flow_left
        if hasattr(self, "f_flow_right"):
            df.f_flow_right = self.f_flow_right

        dv = df.grid.dv

        rho_neutral = (np.sum(df.f, axis=1) * dv).reshape(-1, 1)
        M = rho_neutral * self._M_normalized
        df.f = (
            np.exp(-dt / 2.0 / self.tau) * df.f
            + (1.0 - np.exp(-dt / 2.0 / self.tau)) * M
        )

        df.advect(dt, bc_type=self.bc_type, mass_correction=False)

        rho_neutral = (np.sum(df.f, axis=1) * dv).reshape(-1, 1)
        M = rho_neutral * self._M_normalized
        df.f = (
            np.exp(-dt / 2.0 / self.tau) * df.f
            + (1.0 - np.exp(-dt / 2.0 / self.tau)) * M
        )

        return df
