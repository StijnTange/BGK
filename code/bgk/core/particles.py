import numpy as np


class ParticleSystem:
    def __init__(self, config, x_init, v_init, vy_init, m_weight, N_inj=None):
        self.x = x_init.astype(float).copy()
        self.v = v_init.astype(float).copy()
        self.dim_v = config.grid.dim_v
        if self.dim_v == 2:
            self.vy = vy_init.astype(float).copy()
        else:
            self.vy = np.zeros(0)
        self.N_total = len(self.x)

        self.m_ref = m_weight
        self.m = np.full(self.N_total, m_weight, dtype=float)

        self.is_fresh = np.zeros(self.N_total, dtype=bool)

        self.config = config

        self.xL = config.grid.xL
        self.xR = config.grid.xR
        self.L = self.xR - self.xL
        self.n_cells = config.grid.Nc
        self.dx = self.L / self.n_cells
        self.bc_type = config.grid.bc_type

        self.R = config.physics.R

        self.T_L = getattr(config.physics, "T_L", 1.0)
        self.T_R = getattr(config.physics, "T_R", 1.0)

        self.u_L = getattr(config.physics, "u_L", 0.0)
        self.u_R = getattr(config.physics, "u_R", 0.0)

        self.reflectance_left = (
            float(config.physics.reflectance_left)
            if self.bc_type == "inflow/outflow"
            else None
        )

        self.N_inj = N_inj

    def _inject_particles(
        self,
        dt: float,
        rho: float,
        u: float,
        T: float,
        vmax: float,
    ):
        import math

        # incoming mass flux
        v_th = np.sqrt(2.0 * self.R * T)
        U = u / v_th
        flux = (
            rho
            * np.sqrt(self.R * T / (2.0 * np.pi))
            * (np.exp(-(U**2)) + np.sqrt(np.pi) * U * (1.0 + math.erf(U)))
        )

        total_injected_mass = flux * dt

        if total_injected_mass <= 1e-12:
            return

        N_expected = total_injected_mass / self.m_ref
        m_new_val = self.m_ref

        N_actual = int(np.floor(N_expected))
        if np.random.rand() < (N_expected % 1.0):
            N_actual += 1

        if N_actual == 0:
            return

        # acceptance-rejection for velocity flux distribution: v * f(v)
        v_peak = 0.5 * (u + np.sqrt(u**2 + 4.0 * self.R * T))
        pref = rho / np.sqrt(2.0 * np.pi * self.R * T)
        f_max = v_peak * pref * np.exp(-((v_peak - u) ** 2) / (2.0 * self.R * T))

        v_new = np.empty(N_actual)
        accepted = 0
        while accepted < N_actual:
            batch = N_actual - accepted
            v_prop = np.random.uniform(0, vmax, batch)
            y_test = np.random.uniform(0, f_max * 1.1, batch)
            f_prop = v_prop * pref * np.exp(-((v_prop - u) ** 2) / (2.0 * self.R * T))

            mask = y_test < f_prop
            take = np.sum(mask)
            if take > 0:
                v_new[accepted : accepted + take] = v_prop[mask]
                accepted += take

        # random starting position uniformly over the time step
        time_left = np.random.uniform(0, dt, N_actual)
        x_new = self.xL + v_new * time_left

        self.x = np.concatenate([self.x, x_new])
        self.v = np.concatenate([self.v, v_new])
        if self.dim_v == 2:
            vy_new = np.random.normal(0.0, np.sqrt(self.R * T), N_actual)
            self.vy = np.concatenate([self.vy, vy_new])

        self.m = np.concatenate([self.m, np.full(N_actual, m_new_val)])

        # tag the newly injected particles as fresh
        if hasattr(self, "is_fresh"):
            self.is_fresh = np.concatenate(
                [self.is_fresh, np.ones(N_actual, dtype=bool)]
            )

        self.N_total = len(self.x)

    def apply_boundary_conditions(self):
        keep_mask = np.ones(self.N_total, dtype=bool)

        if self.bc_type == "periodic":
            self.x = self.xL + (self.x - self.xL) % self.L

        elif self.bc_type == "specular":
            out_left = self.x < self.xL
            self.x[out_left] = 2.0 * self.xL - self.x[out_left]
            self.v[out_left] = np.abs(self.v[out_left])

            out_right = self.x > self.xR
            self.x[out_right] = 2.0 * self.xR - self.x[out_right]
            self.v[out_right] = -np.abs(self.v[out_right])

        elif self.bc_type == "inflow/outflow":
            keep_mask = self.x <= self.xR

            hit_left = self.x < self.xL
            R_n = self.reflectance_left

            reflect_mask = hit_left & (np.random.rand(self.N_total) < R_n)
            self.x[reflect_mask] = 2.0 * self.xL - self.x[reflect_mask]
            self.v[reflect_mask] = np.abs(self.v[reflect_mask])

            pumped_mask = hit_left & ~reflect_mask
            keep_mask = keep_mask & ~pumped_mask

            self.x = self.x[keep_mask]
            self.v = self.v[keep_mask]
            if self.dim_v == 2:
                self.vy = self.vy[keep_mask]
            self.m = self.m[keep_mask]

            if hasattr(self, "is_fresh") and len(self.is_fresh) == len(keep_mask):
                self.is_fresh = self.is_fresh[keep_mask]

            self.N_total = len(self.x)

        elif self.bc_type == "diffusive":
            self._apply_diffusive_bc()

        else:
            raise NotImplementedError

        return keep_mask

    def get_cell_indices(self):
        indices = np.floor((self.x - self.xL) / self.dx).astype(int)
        return np.clip(indices, 0, self.n_cells - 1)

    def get_particles_in_cell(self, cell_idx: int, cell_indices: np.ndarray = None):
        if cell_indices is None:
            cell_indices = self.get_cell_indices()
        return np.where(cell_indices == cell_idx)[0]

    def compute_cell_moments(self):
        cell_indices = self.get_cell_indices()

        mass_sum = np.bincount(cell_indices, weights=self.m, minlength=self.n_cells)
        density = mass_sum / self.dx

        mom_x = np.bincount(
            cell_indices, weights=self.m * self.v, minlength=self.n_cells
        )
        Ux = np.zeros(self.n_cells)
        valid = mass_sum > 0
        Ux[valid] = mom_x[valid] / mass_sum[valid]

        E_x = np.bincount(
            cell_indices, weights=self.m * self.v**2, minlength=self.n_cells
        )

        if self.dim_v == 2:
            mom_y = np.bincount(
                cell_indices, weights=self.m * self.vy, minlength=self.n_cells
            )
            Uy = np.zeros(self.n_cells)
            Uy[valid] = mom_y[valid] / mass_sum[valid]
            E_y = np.bincount(
                cell_indices, weights=self.m * self.vy**2, minlength=self.n_cells
            )
            T = np.zeros(self.n_cells)
            T[valid] = (
                (E_x[valid] - mass_sum[valid] * Ux[valid] ** 2)
                + (E_y[valid] - mass_sum[valid] * Uy[valid] ** 2)
            ) / (2.0 * mass_sum[valid] * self.R)
            return density, Ux, Uy, T
        else:
            T = np.zeros(self.n_cells)
            T[valid] = (E_x[valid] - mass_sum[valid] * Ux[valid] ** 2) / (
                mass_sum[valid] * self.R
            )
            return density, Ux, T

    def compute_heat_flux(self):
        cell_indices = self.get_cell_indices()

        if self.dim_v == 2:
            density, Ux, Uy, T = self.compute_cell_moments()
            Ux_mapped = Ux[cell_indices]
            Uy_mapped = Uy[cell_indices]
            cx = self.v - Ux_mapped
            cy = self.vy - Uy_mapped
            c2 = cx**2 + cy**2
        else:
            density, Ux, T = self.compute_cell_moments()
            Ux_mapped = Ux[cell_indices]
            cx = self.v - Ux_mapped
            c2 = cx**2

        q_weights = 0.5 * self.m * cx * c2
        q_sum = np.bincount(cell_indices, weights=q_weights, minlength=self.n_cells)

        valid = density > 0
        q = np.zeros(self.n_cells)
        q[valid] = q_sum[valid] / self.dx

        return q

    def advect(self, dt: float):
        if hasattr(self, "is_fresh") and np.any(self.is_fresh):
            non_fresh_mask = ~self.is_fresh
            self.x[non_fresh_mask] += self.v[non_fresh_mask] * dt
        else:
            self.x += self.v * dt

        self.apply_boundary_conditions()

    def get_cell_indices_from_positions(self, x_positions: np.ndarray):
        indices = np.floor((x_positions - self.xL) / self.dx).astype(int)
        return np.clip(indices, 0, self.n_cells - 1)

    def _apply_diffusive_bc(self):
        out_left = self.x < self.xL
        if np.any(out_left):
            n = np.sum(out_left)
            u_rand = np.random.uniform(0.0, 1.0, n)
            v_normal = np.sqrt(
                -2.0 * self.R * self.T_L * np.log(np.maximum(u_rand, 1e-15))
            )
            self.x[out_left] = 2.0 * self.xL - self.x[out_left]
            self.x[out_left] = np.maximum(self.x[out_left], self.xL)
            self.v[out_left] = v_normal
            if self.dim_v == 2:
                self.vy[out_left] = np.random.normal(
                    self.u_L, np.sqrt(self.R * self.T_L), n
                )

        out_right = self.x > self.xR
        if np.any(out_right):
            n = np.sum(out_right)
            u_rand = np.random.uniform(0.0, 1.0, n)
            v_normal = np.sqrt(
                -2.0 * self.R * self.T_R * np.log(np.maximum(u_rand, 1e-15))
            )
            self.x[out_right] = 2.0 * self.xR - self.x[out_right]
            self.x[out_right] = np.minimum(self.x[out_right], self.xR)
            self.v[out_right] = -v_normal
            if self.dim_v == 2:
                self.vy[out_right] = np.random.normal(
                    self.u_R, np.sqrt(self.R * self.T_R), n
                )

    def apply_bc_to_positions(self, x, v, vy=None):
        keep_mask = np.ones(len(x), dtype=bool)

        if self.bc_type == "periodic":
            x = (x - self.xL) % self.L + self.xL

        elif self.bc_type == "specular":
            out_left = x < self.xL
            x[out_left] = 2.0 * self.xL - x[out_left]
            v[out_left] = np.abs(v[out_left])

            out_right = x > self.xR
            x[out_right] = 2.0 * self.xR - x[out_right]
            v[out_right] = -np.abs(v[out_right])

        elif self.bc_type == "inflow/outflow":
            # right boundary
            keep_mask &= x <= self.xR

            # left boundary
            hit_left = x < self.xL
            R_n = self.reflectance_left

            reflect_mask = hit_left & (np.random.rand(len(x)) < R_n)

            # specular reflection
            x[reflect_mask] = 2.0 * self.xL - x[reflect_mask]
            v[reflect_mask] = np.abs(v[reflect_mask])

            pumped_mask = hit_left & ~reflect_mask
            keep_mask &= ~pumped_mask

        elif self.bc_type == "diffusive":
            out_left = x < self.xL
            if np.any(out_left):
                n = np.sum(out_left)
                u_rand = np.random.uniform(0.0, 1.0, n)
                v[out_left] = np.sqrt(
                    -2.0 * self.R * self.T_L * np.log(np.maximum(u_rand, 1e-15))
                )
                x[out_left] = np.maximum(2.0 * self.xL - x[out_left], self.xL)

                if vy is not None:
                    vy[out_left] = np.random.normal(
                        self.u_L, np.sqrt(self.R * self.T_L), n
                    )

            out_right = x > self.xR
            if np.any(out_right):
                n = np.sum(out_right)
                u_rand = np.random.uniform(0.0, 1.0, n)
                v[out_right] = -np.sqrt(
                    -2.0 * self.R * self.T_R * np.log(np.maximum(u_rand, 1e-15))
                )
                x[out_right] = np.minimum(2.0 * self.xR - x[out_right], self.xR)

                if vy is not None:
                    vy[out_right] = np.random.normal(
                        self.u_R, np.sqrt(self.R * self.T_R), n
                    )

        else:
            raise NotImplementedError

        if vy is not None:
            return x, v, vy, keep_mask
        return x, v, keep_mask
