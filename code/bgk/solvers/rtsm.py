import numpy as np
from bgk.solvers.base import Solver


class RTSMSolver(Solver):
    def __init__(self, config):
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.bc_type = config.grid.bc_type
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)

    def _compute_2d_cell_moments(self, particles, cell_indices):
        n_cells = particles.n_cells
        mass_sum = np.bincount(cell_indices, weights=particles.m, minlength=n_cells)
        valid = mass_sum > 0

        mom_x = np.bincount(
            cell_indices, weights=particles.m * particles.v, minlength=n_cells
        )
        Ux = np.zeros(n_cells)
        Ux[valid] = mom_x[valid] / mass_sum[valid]

        mom_y = np.bincount(
            cell_indices, weights=particles.m * particles.vy, minlength=n_cells
        )
        Uy = np.zeros(n_cells)
        Uy[valid] = mom_y[valid] / mass_sum[valid]

        E_x = np.bincount(
            cell_indices, weights=particles.m * particles.v**2, minlength=n_cells
        )
        E_y = np.bincount(
            cell_indices, weights=particles.m * particles.vy**2, minlength=n_cells
        )
        T = np.zeros(n_cells)
        T[valid] = (
            (E_x[valid] - mass_sum[valid] * Ux[valid] ** 2)
            + (E_y[valid] - mass_sum[valid] * Uy[valid] ** 2)
        ) / (2.0 * mass_sum[valid] * self.R)

        density = mass_sum / particles.dx
        return density, Ux, Uy, T

    def step(
        self,
        particles,
        dt: float,
    ):
        two_d = hasattr(particles, "vy") and particles.dim_v == 2

        particles.advect(dt)

        cell_indices = particles.get_cell_indices()

        if two_d:
            cell_density, cell_Ux, cell_Uy, cell_T = self._compute_2d_cell_moments(
                particles, cell_indices
            )
        else:
            cell_density, cell_Ux, cell_T = particles.compute_cell_moments()
            cell_Uy = None

        cell_T = np.maximum(cell_T, 1e-12)

        cell_p = cell_density * self.R * cell_T

        cell_p = np.maximum(cell_p, 1e-15)

        cell_mu = cell_T**self.omega
        cell_tau = self.tau_ref * cell_mu / cell_p
        tau_p = cell_tau[cell_indices]
        P_relax = 1.0 - np.exp(-dt / tau_p)

        valid_cells = cell_T > 0.0
        valid_particles = valid_cells[cell_indices]

        relax_mask = np.random.rand(particles.N_total) < P_relax
        update_mask = valid_particles & relax_mask

        if np.any(update_mask):
            c_idx_update = cell_indices[update_mask]
            old_vx = particles.v[update_mask]
            old_vy = particles.vy[update_mask] if two_d else None

            counts = np.bincount(c_idx_update, minlength=particles.n_cells)
            correctable_cells = counts > 1

            Ux_rem = np.copy(cell_Ux)
            Uy_rem = np.copy(cell_Uy) if two_d else np.zeros_like(cell_Ux)
            T_rem = np.copy(cell_T)

            sum_vx = np.bincount(
                c_idx_update, weights=old_vx, minlength=particles.n_cells
            )
            Ux_rem[correctable_cells] = (
                sum_vx[correctable_cells] / counts[correctable_cells]
            )
            sum_vx2 = np.bincount(
                c_idx_update, weights=old_vx**2, minlength=particles.n_cells
            )

            if two_d:
                sum_vy = np.bincount(
                    c_idx_update, weights=old_vy, minlength=particles.n_cells
                )
                Uy_rem[correctable_cells] = (
                    sum_vy[correctable_cells] / counts[correctable_cells]
                )
                sum_vy2 = np.bincount(
                    c_idx_update, weights=old_vy**2, minlength=particles.n_cells
                )

                cx2 = (
                    sum_vx2[correctable_cells]
                    - counts[correctable_cells] * Ux_rem[correctable_cells] ** 2
                )
                cy2 = (
                    sum_vy2[correctable_cells]
                    - counts[correctable_cells] * Uy_rem[correctable_cells] ** 2
                )
                raw_T = (cx2 + cy2) / (2.0 * counts[correctable_cells] * self.R)
            else:
                cx2 = (
                    sum_vx2[correctable_cells]
                    - counts[correctable_cells] * Ux_rem[correctable_cells] ** 2
                )
                raw_T = cx2 / (counts[correctable_cells] * self.R)

            T_rem[correctable_cells] = np.maximum(raw_T, 1e-12)

            Ux_target = Ux_rem[c_idx_update]
            Uy_target = Uy_rem[c_idx_update] if two_d else None
            T_target = T_rem[c_idx_update]
            std_dev = np.sqrt(self.R * T_target)

            new_vx = np.random.normal(loc=Ux_target, scale=std_dev)
            new_vy = np.random.normal(loc=Uy_target, scale=std_dev) if two_d else None

            Ux_samp = np.zeros(particles.n_cells)
            T_samp = np.zeros(particles.n_cells)

            sum_new_vx = np.bincount(
                c_idx_update, weights=new_vx, minlength=particles.n_cells
            )
            sum_new_vx2 = np.bincount(
                c_idx_update, weights=new_vx**2, minlength=particles.n_cells
            )
            Ux_samp[correctable_cells] = (
                sum_new_vx[correctable_cells] / counts[correctable_cells]
            )

            if two_d:
                Uy_samp = np.zeros(particles.n_cells)
                sum_new_vy = np.bincount(
                    c_idx_update, weights=new_vy, minlength=particles.n_cells
                )
                sum_new_vy2 = np.bincount(
                    c_idx_update, weights=new_vy**2, minlength=particles.n_cells
                )
                Uy_samp[correctable_cells] = (
                    sum_new_vy[correctable_cells] / counts[correctable_cells]
                )
                cx2_samp = (
                    sum_new_vx2[correctable_cells]
                    - counts[correctable_cells] * Ux_samp[correctable_cells] ** 2
                )
                cy2_samp = (
                    sum_new_vy2[correctable_cells]
                    - counts[correctable_cells] * Uy_samp[correctable_cells] ** 2
                )
                raw_T_samp = (cx2_samp + cy2_samp) / (
                    2.0 * counts[correctable_cells] * self.R
                )
            else:
                cx2_samp = (
                    sum_new_vx2[correctable_cells]
                    - counts[correctable_cells] * Ux_samp[correctable_cells] ** 2
                )
                raw_T_samp = cx2_samp / (counts[correctable_cells] * self.R)
                Uy_samp = None

            T_samp[correctable_cells] = np.maximum(raw_T_samp, 1e-12)

            Ux_samp_p = Ux_samp[c_idx_update]
            T_samp_p = T_samp[c_idx_update]
            can_correct = correctable_cells[c_idx_update] & (T_samp_p > 1e-12)

            scale = np.where(
                can_correct,
                np.sqrt(T_target / np.where(T_samp_p > 1e-12, T_samp_p, 1.0)),
                1.0,
            )

            new_vx[can_correct] = (
                new_vx[can_correct] - Ux_samp_p[can_correct]
            ) * scale[can_correct] + Ux_target[can_correct]
            particles.v[update_mask] = new_vx

            if two_d:
                Uy_samp_p = Uy_samp[c_idx_update]
                new_vy[can_correct] = (
                    new_vy[can_correct] - Uy_samp_p[can_correct]
                ) * scale[can_correct] + Uy_target[can_correct]
                particles.vy[update_mask] = new_vy

        return particles


class LinearRTSMSolver(Solver):
    def __init__(
        self,
        config,
        u_bg: np.ndarray,
        T_bg: np.ndarray,
        rho_bg=1.0,
        target_N_total=None,
        target_ppc=None,
    ):
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.bc_type = config.grid.bc_type
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)

        self.u_bg = u_bg
        self.T_bg = T_bg
        self.rho_bg = rho_bg

        self.target_ppc = target_ppc

        self.tau_cells = self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / self.rho_bg

    def step(
        self,
        particles,
        dt: float,
    ):
        if self.bc_type in ("inflow/outflow", "open") and hasattr(
            particles, "flow_left"
        ):
            for inflow_component in particles.flow_left:
                particles._inject_particles(
                    dt=dt,
                    rho=inflow_component["rho"],
                    u=inflow_component["u"],
                    T=inflow_component["T"],
                    vmax=inflow_component["vmax"],
                    C1_scale=1.0,
                )

        particles.advect(dt)
        if hasattr(particles, "is_fresh"):
            particles.is_fresh[:] = False

        cell_indices = particles.get_cell_indices()
        tau_p = self.tau_cells[cell_indices]
        P_relax = 1.0 - np.exp(-dt / tau_p)

        relax_mask = np.random.rand(particles.N_total) < P_relax

        if np.any(relax_mask):
            c_idx_update = cell_indices[relax_mask]

            U_target = self.u_bg[c_idx_update]
            T_target = self.T_bg[c_idx_update]
            std_dev = np.sqrt(self.R * T_target)
            new_v = np.random.normal(loc=U_target, scale=std_dev)

            particles.v[relax_mask] = new_v

        return particles
