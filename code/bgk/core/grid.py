from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    def __init__(self, grid_config):
        self.xL = grid_config.xL
        self.xR = grid_config.xR
        self.Nx = grid_config.Nx
        self.Nc = grid_config.Nc
        self.dim_v = grid_config.dim_v
        self.Nv = grid_config.Nv

        raw_vmin = grid_config.vmin
        raw_vmax = grid_config.vmax

        if np.isscalar(raw_vmin):
            self.vmin = [raw_vmin] * self.dim_v
        else:
            self.vmin = list(raw_vmin)

        if np.isscalar(raw_vmax):
            self.vmax = [raw_vmax] * self.dim_v
        else:
            self.vmax = list(raw_vmax)

        if len(self.vmin) != self.dim_v or len(self.vmax) != self.dim_v:
            raise ValueError(
                f"vmin and vmax must each have length {self.dim_v} "
                f"for a {self.dim_v}D velocity space, "
                f"got vmin={self.vmin}, vmax={self.vmax}."
            )

        # ── Spatial grid ──────────────────────────────────────────────────────
        if self.Nc is not None and self.Nx is None:
            self.dx = (self.xR - self.xL) / self.Nc
            grid_points = np.linspace(self.xL, self.xR, self.Nc + 1)
            self.x = 0.5 * (grid_points[:-1] + grid_points[1:])
        elif self.Nx is not None and self.Nc is None:
            self.dx = (self.xR - self.xL) / (self.Nx - 1)
            self.x = np.linspace(self.xL, self.xR, self.Nx)
        else:
            raise ValueError("Exactly one of Nx or Nc must be provided.")

        # ── Velocity grid ─────────────────────────────────────────────────────
        if len(self.Nv) != self.dim_v:
            raise ValueError(
                f"Nv should have length {self.dim_v} "
                f"for a {self.dim_v}D velocity space."
            )

        if self.dim_v == 1:
            v0min, v0max = self.vmin[0], self.vmax[0]
            self.v = np.linspace(v0min, v0max, self.Nv[0])
            self.dv = (v0max - v0min) / (self.Nv[0] - 1)

        elif self.dim_v == 2:
            vxmin, vxmax = self.vmin[0], self.vmax[0]
            vymin, vymax = self.vmin[1], self.vmax[1]

            self.vx = np.linspace(vxmin, vxmax, self.Nv[0])
            self.vy = np.linspace(vymin, vymax, self.Nv[1])
            self.dvx = (vxmax - vxmin) / (self.Nv[0] - 1)
            self.dvy = (vymax - vymin) / (self.Nv[1] - 1)
            self.vx_mesh, self.vy_mesh = np.meshgrid(self.vx, self.vy, indexing="ij")

        else:
            raise ValueError("Only 1D and 2D velocity spaces are supported.")
