from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class GridConfig:
    xL: float
    xR: float
    Nx: Optional[int]
    Nc: Optional[int]
    vmin: float
    vmax: float
    dim_v: int
    Nv: list[int]
    bc_type: Literal["periodic", "specular", "diffusive"]


@dataclass
class TimeConfig:
    t_final: float
    dt: Optional[float]
    CFL: Optional[float]


@dataclass
class PhysicsConfig:
    problem_name: str
    Kn: float
    omega: float = 0.5
    R: float = 1.0
    u_L: float = None
    u_R: float = None
    T_L: float = None
    T_R: float = None
    rho_in: float = None
    u_in: float = None
    T_in: float = None
    reflectance_left: Optional[float] = None
    constant_tau: bool = False
    tau: Optional[float] = None


@dataclass
class Config:
    grid: GridConfig
    time: TimeConfig
    physics: PhysicsConfig
