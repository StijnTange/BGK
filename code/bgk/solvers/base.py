from abc import ABC, abstractmethod

import numpy as np
from bgk.core.distribution import DistributionFunction


class Solver(ABC):
    def __init__(self, config):
        self.config = config
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.bc_type = config.grid.bc_type
        self.dim_v = config.grid.dim_v
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)

    @abstractmethod
    def step(self, df: DistributionFunction, dt: float) -> DistributionFunction:
        pass

    def _compute_tau(self, rho, T):
        if getattr(self.config.physics, "constant_tau", False):
            tau = np.full_like(rho, self.tau_ref)
            return tau.reshape((-1,) + (1,) * self.dim_v)
        T_s = np.maximum(T, 1e-15)
        rho_s = np.maximum(rho, 1e-15)
        tau = self.tau_ref * (T_s**self.omega) / (rho_s * self.R * T_s)
        return tau.reshape((-1,) + (1,) * self.dim_v)
