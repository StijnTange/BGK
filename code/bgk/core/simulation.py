import numpy as np


class Simulation:
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        f: np.ndarray,
        dt: float,
        Kn: float,
        omega: float,
        R: float,
        rho: np.ndarray,
        u: np.ndarray,
        T: np.ndarray,
        q: np.ndarray,
    ):
        self.x = x
        self.v = v
        self.t = t
        self.f = f
        self.dt = dt
        self.Kn = Kn
        self.omega = omega
        self.Kn = Kn
        self.omega = omega
        self.R = R
        self.rho = rho
        self.u = u
        self.T = T
        self.q = q
