import numpy as np
from bgk.solvers.base import Solver
from scipy.special import erf


class UGKPSolver(Solver):
    def __init__(self, config):
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.bc_type = config.grid.bc_type
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)
        self.config = config

    def step(self, particles, macro_grid, dt: float, t_n=None, source_func=None):
        n_cells = particles.n_cells
        dx = particles.dx
        dim_v = particles.dim_v

        cell_indices = particles.get_cell_indices()
        tau_cells = self.compute_tau(macro_grid.W)
        tau_p = tau_cells[cell_indices]

        eta = np.random.rand(particles.N_total)
        t_c = -tau_p * np.log(eta)

        collisionless_mask = t_c >= dt
        collisional_mask = ~collisionless_mask

        x_free = particles.x[collisionless_mask]
        v_free = particles.v[collisionless_mask]

        x_coll = particles.x[collisional_mask]
        v_coll = particles.v[collisional_mask]
        m_coll = particles.m[collisional_mask]
        t_c_coll = t_c[collisional_mask]

        if dim_v == 2:
            vy_free = particles.vy[collisionless_mask]
            vy_coll = particles.vy[collisional_mask]
            x_coll_new = x_coll + v_coll * t_c_coll
            x_coll_new, v_coll, vy_coll, keep_mask = particles.apply_bc_to_positions(
                x_coll_new, v_coll, vy=vy_coll
            )
        else:
            x_coll_new = x_coll + v_coll * t_c_coll
            x_coll_new, v_coll, keep_mask = particles.apply_bc_to_positions(
                x_coll_new, v_coll
            )
            vy_coll = None

        m_coll_active = np.where(keep_mask, m_coll, 0.0)
        cell_indices_coll = particles.get_cell_indices_from_positions(x_coll_new)

        w_coll = np.zeros((n_cells, dim_v + 2))
        if len(x_coll_new) > 0:
            w_coll[:, 0] = (
                np.bincount(cell_indices_coll, weights=m_coll_active, minlength=n_cells)
                / dx
            )
            w_coll[:, 1] = (
                np.bincount(
                    cell_indices_coll, weights=m_coll_active * v_coll, minlength=n_cells
                )
                / dx
            )
            if dim_v == 2:
                w_coll[:, 2] = (
                    np.bincount(
                        cell_indices_coll,
                        weights=m_coll_active * vy_coll,
                        minlength=n_cells,
                    )
                    / dx
                )
                w_coll[:, 3] = (
                    np.bincount(
                        cell_indices_coll,
                        weights=m_coll_active * 0.5 * (v_coll**2 + vy_coll**2),
                        minlength=n_cells,
                    )
                    / dx
                )
            else:
                w_coll[:, 2] = (
                    np.bincount(
                        cell_indices_coll,
                        weights=m_coll_active * 0.5 * (v_coll**2),
                        minlength=n_cells,
                    )
                    / dx
                )

        particles.x = x_free
        particles.v = v_free
        if dim_v == 2:
            particles.vy = vy_free
        particles.m = particles.m[collisionless_mask]
        particles.N_total = len(x_free)
        particles.advect(dt)

        new_cell_indices = particles.get_cell_indices()
        w_Pf = np.zeros((n_cells, dim_v + 2))

        if particles.N_total > 0:
            w_Pf[:, 0] = (
                np.bincount(new_cell_indices, weights=particles.m, minlength=n_cells)
                / dx
            )
            w_Pf[:, 1] = (
                np.bincount(
                    new_cell_indices,
                    weights=particles.m * particles.v,
                    minlength=n_cells,
                )
                / dx
            )
            if dim_v == 2:
                w_Pf[:, 2] = (
                    np.bincount(
                        new_cell_indices,
                        weights=particles.m * particles.vy,
                        minlength=n_cells,
                    )
                    / dx
                )
                w_Pf[:, 3] = (
                    np.bincount(
                        new_cell_indices,
                        weights=particles.m * 0.5 * (particles.v**2 + particles.vy**2),
                        minlength=n_cells,
                    )
                    / dx
                )
            else:
                w_Pf[:, 2] = (
                    np.bincount(
                        new_cell_indices,
                        weights=particles.m * 0.5 * (particles.v**2),
                        minlength=n_cells,
                    )
                    / dx
                )

        w_total_particles = w_Pf + w_coll

        F_eq = self.compute_equilibrium_flux(
            macro_grid.W,
            dt,
            tau_cells,
            dx,
            particles.bc_type,
            config=getattr(self, "config", None),
        )
        macro_grid.W = w_total_particles - (dt / dx) * (F_eq[1:] - F_eq[:-1])

        rho_min, T_min = 1e-12, 1e-10
        rho = np.maximum(macro_grid.W[:, 0], rho_min)
        valid = macro_grid.W[:, 0] > rho_min

        ux = np.zeros_like(rho)
        ux[valid] = macro_grid.W[valid, 1] / rho[valid]

        macro_grid.W[:, 0] = rho
        macro_grid.W[:, 1] = np.where(valid, macro_grid.W[:, 1], 0.0)

        if dim_v == 2:
            uy = np.zeros_like(rho)
            uy[valid] = macro_grid.W[valid, 2] / rho[valid]
            E = np.maximum(
                macro_grid.W[:, 3], 0.5 * rho * (ux**2 + uy**2) + rho * self.R * T_min
            )
            macro_grid.W[:, 2] = np.where(valid, macro_grid.W[:, 2], 0.0)
            macro_grid.W[:, 3] = E
        else:
            E = np.maximum(
                macro_grid.W[:, 2], 0.5 * rho * ux**2 + 0.5 * rho * self.R * T_min
            )
            macro_grid.W[:, 2] = E

        N_to_spawn = np.sum(keep_mask)

        w_h = macro_grid.W - w_Pf
        rho_h, target_Ux, target_Uy, target_T = self._w_to_primitive(w_h)
        rho_h = np.maximum(rho_h, 0.0)

        cells_to_spawn = np.zeros(n_cells, dtype=int)
        valid_h_cells = rho_h > 1e-12
        mass_h = np.where(valid_h_cells, rho_h * dx, 0.0)
        total_mass_h = np.sum(mass_h)

        if N_to_spawn > 0 and total_mass_h > 1e-14:
            exact_counts = N_to_spawn * (mass_h / total_mass_h)
            cells_to_spawn = np.floor(exact_counts).astype(int)

            shortfall = N_to_spawn - np.sum(cells_to_spawn)
            if shortfall > 0:
                remainders = exact_counts - cells_to_spawn
                largest_rem_idx = np.argsort(remainders)[-shortfall:]
                cells_to_spawn[largest_rem_idx] += 1

        total_new = np.sum(cells_to_spawn)

        if total_new > 0:
            new_cell_indices = np.repeat(np.arange(n_cells), cells_to_spawn)

            safe_spawn = np.maximum(cells_to_spawn, 1)
            exact_mass_per_cell = (rho_h * dx) / safe_spawn
            new_m = exact_mass_per_cell[new_cell_indices]

            u_pos = np.random.uniform(0.0, 1.0, size=total_new)
            cell_left_edges = particles.xL + new_cell_indices * dx
            new_x = cell_left_edges + u_pos * dx

            Ux_mapped = target_Ux[new_cell_indices]
            T_mapped = target_T[new_cell_indices]
            std_dev_mapped = np.sqrt(self.R * T_mapped)

            raw_v = np.random.normal(loc=Ux_mapped, scale=std_dev_mapped)

            sum_v = np.bincount(new_cell_indices, weights=raw_v, minlength=n_cells)
            count_v = np.bincount(new_cell_indices, minlength=n_cells)
            safe_count = np.maximum(count_v, 1)
            mean_v_mapped = (sum_v / safe_count)[new_cell_indices]

            var_v = (
                np.bincount(
                    new_cell_indices,
                    weights=(raw_v - mean_v_mapped) ** 2,
                    minlength=n_cells,
                )
                / safe_count
            )
            safe_var = np.where(
                var_v[new_cell_indices] > 1e-14, var_v[new_cell_indices], 1.0
            )

            new_v = np.copy(raw_v)
            multi_mask = count_v[new_cell_indices] > 1

            new_v[multi_mask] = (
                raw_v[multi_mask] - mean_v_mapped[multi_mask]
            ) * np.sqrt(
                (self.R * T_mapped[multi_mask]) / safe_var[multi_mask]
            ) + Ux_mapped[multi_mask]

            vmax = getattr(self.config.grid, "vmax", 20.0)
            new_v = np.clip(new_v, -vmax, vmax)

            particles.x = np.concatenate([particles.x, new_x])
            particles.v = np.concatenate([particles.v, new_v])

            if particles.dim_v == 2:
                Uy_mapped = target_Uy[new_cell_indices]
                raw_vy = np.random.normal(loc=Uy_mapped, scale=std_dev_mapped)

                sum_vy = np.bincount(
                    new_cell_indices, weights=raw_vy, minlength=n_cells
                )
                mean_vy_mapped = (sum_vy / safe_count)[new_cell_indices]
                var_vy = (
                    np.bincount(
                        new_cell_indices,
                        weights=(raw_vy - mean_vy_mapped) ** 2,
                        minlength=n_cells,
                    )
                    / safe_count
                )
                safe_var_y = np.where(
                    var_vy[new_cell_indices] > 1e-14, var_vy[new_cell_indices], 1.0
                )

                new_vy = np.copy(raw_vy)
                new_vy[multi_mask] = (
                    raw_vy[multi_mask] - mean_vy_mapped[multi_mask]
                ) * np.sqrt(
                    (self.R * T_mapped[multi_mask]) / safe_var_y[multi_mask]
                ) + Uy_mapped[multi_mask]

                new_vy = np.clip(new_vy, -vmax, vmax)
                particles.vy = np.concatenate([particles.vy, new_vy])

            particles.m = np.concatenate([particles.m, new_m])

        particles.N_total = len(particles.x)
        return particles, macro_grid

    def _w_to_primitive(self, W):
        rho = np.maximum(W[:, 0], 1e-12)
        valid = W[:, 0] > 1e-12

        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        T = np.ones_like(rho) * 1e-8

        vmax = getattr(self.config.grid, "vmax", 20.0)

        if W.shape[1] == 4:
            ux[valid] = W[valid, 1] / rho[valid]
            uy[valid] = W[valid, 2] / rho[valid]

            ux = np.clip(ux, -vmax, vmax)
            uy = np.clip(uy, -vmax, vmax)

            E = np.maximum(W[:, 3], 0.0)
            raw_T = (1.0 / self.R) * (
                (E[valid] / rho[valid]) - 0.5 * (ux[valid] ** 2 + uy[valid] ** 2)
            )
        else:
            ux[valid] = W[valid, 1] / rho[valid]

            ux = np.clip(ux, -vmax, vmax)

            E = np.maximum(W[:, 2], 0.0)
            raw_T = (2.0 / self.R) * ((E[valid] / rho[valid]) - 0.5 * ux[valid] ** 2)

        T[valid] = np.maximum(raw_T, 1e-8)
        return rho, ux, uy, T

    def _compute_analytical_moments(self, rho, u, T, max_k=5):
        N = len(rho)
        G_plus = np.zeros((max_k + 1, N))
        G_minus = np.zeros((max_k + 1, N))
        sqrt_2RT = np.sqrt(2.0 * self.R * T)

        G_0_plus = (rho / 2.0) * (1.0 + erf(u / sqrt_2RT))
        G_0_minus = (rho / 2.0) * (1.0 - erf(u / sqrt_2RT))

        G_plus[0], G_minus[0] = G_0_plus, G_0_minus

        if max_k >= 1:
            exp_term = (
                rho
                * np.sqrt(self.R * T / (2.0 * np.pi))
                * np.exp(-(u**2) / (2.0 * self.R * T))
            )
            G_1_plus = u * G_0_plus + exp_term
            G_1_minus = u * G_0_minus - exp_term
            G_plus[1], G_minus[1] = G_1_plus, G_1_minus

        for k in range(max_k - 1):
            G_plus[k + 2] = u * G_plus[k + 1] + (k + 1) * self.R * T * G_plus[k]
            G_minus[k + 2] = u * G_minus[k + 1] + (k + 1) * self.R * T * G_minus[k]

        return G_plus, G_minus

    def _get_time_averaged_coeffs(self, dt, tau):
        exp_at = np.exp(-dt / tau)
        C1 = (tau / dt) * (1.0 - exp_at)
        C2 = dt * ((tau / dt) * exp_at - (tau**2 / dt**2) * (1.0 - exp_at))
        C3 = 1.0 - (tau / dt) * (1.0 - exp_at)
        C4 = dt * (
            (2.0 * tau**2 / dt**2) * (1.0 - exp_at) - (tau / dt) - (tau / dt) * exp_at
        )
        C5 = dt * (0.5 - (tau / dt) + (tau**2 / dt**2) * (1.0 - exp_at))
        return C1, C2, C3, C4, C5

    def _get_slopes(self, W, dx):
        s1 = (W[1:-1, :] - W[0:-2, :]) / dx
        s2 = (W[2:, :] - W[1:-1, :]) / dx
        eps = 1e-15
        slopes_inner = (
            (np.sign(s1) + np.sign(s2))
            * (np.abs(s1) * np.abs(s2))
            / (np.abs(s1) + np.abs(s2) + eps)
        )
        slopes = np.zeros_like(W)
        slopes[1:-1, :] = slopes_inner
        return slopes

    def _pad_with_ghosts(self, W_inner, bc_type="periodic", pad=2, config=None):
        Nx, Nv = W_inner.shape
        W = np.zeros((Nx + 2 * pad, Nv))
        W[pad:-pad] = W_inner

        if bc_type == "periodic":
            W[:pad] = W_inner[-pad:]
            W[-pad:] = W_inner[:pad]
        elif bc_type == "specular":
            for k in range(pad):
                W[k] = W_inner[pad - k - 1]
                W[k, 1] *= -1.0
                W[-k - 1] = W_inner[-(pad - k)]
                W[-k - 1, 1] *= -1.0
        elif bc_type in ("inflow/outflow", "open"):
            if config is not None and hasattr(config.physics, "rho_L"):
                rho_in = config.physics.rho_L
                u_in = config.physics.u_L
                T_in = config.physics.T_L

                if W_inner.shape[1] == 4:
                    W_ghost_L = np.array(
                        [
                            rho_in,
                            rho_in * u_in,
                            0.0,
                            0.5 * rho_in * u_in**2 + rho_in * self.R * T_in,
                        ]
                    )
                else:
                    W_ghost_L = np.array(
                        [
                            rho_in,
                            rho_in * u_in,
                            0.5 * rho_in * u_in**2 + 0.5 * rho_in * self.R * T_in,
                        ]
                    )

                for k in range(pad):
                    W[k] = W_ghost_L
                    W[-k - 1] = W_inner[-1]
            else:
                for k in range(pad):
                    W[k] = W_inner[0]
                    W[-k - 1] = W_inner[-1]

        elif bc_type == "diffusive":
            if config is not None:
                TL = getattr(config.physics, "T_L", 1.0)
                TR = getattr(config.physics, "T_R", 1.0)

                uyL = getattr(config.physics, "u_L", 0.0)
                uyR = getattr(config.physics, "u_R", 0.0)
            else:
                TL = TR = 1.0
                uyL = uyR = 0.0

            rho_L = W_inner[0, 0]
            rho_R = W_inner[-1, 0]

            if W_inner.shape[1] == 4:
                W_ghost_L = np.array(
                    [
                        rho_L,
                        0.0,
                        rho_L * uyL,
                        0.5 * rho_L * (uyL**2) + rho_L * self.R * TL,
                    ]
                )
                W_ghost_R = np.array(
                    [
                        rho_R,
                        0.0,
                        rho_R * uyR,
                        0.5 * rho_R * (uyR**2) + rho_R * self.R * TR,
                    ]
                )
            else:
                W_ghost_L = np.array([rho_L, 0.0, 0.5 * rho_L * self.R * TL])
                W_ghost_R = np.array([rho_R, 0.0, 0.5 * rho_R * self.R * TR])

            for k in range(pad):
                W[k] = W_ghost_L
                W[-k - 1] = W_ghost_R
        else:
            for k in range(pad):
                W[k] = W_inner[0]
                W[-k - 1] = W_inner[-1]
        return W

    def compute_equilibrium_flux(
        self, W_cells, dt, tau_cells, dx, bc_type, config=None
    ):
        dim_w = W_cells.shape[1]

        W_pad = self._pad_with_ghosts(W_cells, bc_type=bc_type, pad=2, config=config)
        slopes = self._get_slopes(W_pad, dx)

        W_L = W_pad[1:-2] + 0.5 * dx * slopes[1:-2]
        W_R = W_pad[2:-1] - 0.5 * dx * slopes[2:-1]

        rho_L, ux_L, uy_L, T_L = self._w_to_primitive(W_L)
        rho_R, ux_R, uy_R, T_R = self._w_to_primitive(W_R)

        G_plus_L, _ = self._compute_analytical_moments(rho_L, ux_L, T_L, max_k=3)
        _, G_minus_R = self._compute_analytical_moments(rho_R, ux_R, T_R, max_k=3)

        W_face = np.zeros_like(W_L)
        W_face[:, 0] = G_plus_L[0] + G_minus_R[0]
        W_face[:, 1] = G_plus_L[1] + G_minus_R[1]

        if dim_w == 4:
            W_face[:, 2] = uy_L * G_plus_L[0] + uy_R * G_minus_R[0]
            E1D_L, E1D_R = 0.5 * G_plus_L[2], 0.5 * G_minus_R[2]
            Hy_L = 0.5 * (uy_L**2 + self.R * T_L)
            Hy_R = 0.5 * (uy_R**2 + self.R * T_R)
            W_face[:, 3] = (E1D_L + Hy_L * G_plus_L[0]) + (E1D_R + Hy_R * G_minus_R[0])
        else:
            W_face[:, 2] = 0.5 * (G_plus_L[2] + G_minus_R[2])

        rho_f, u_f, uy_f, T_f = self._w_to_primitive(W_face)

        G_plus_f, G_minus_f = self._compute_analytical_moments(rho_f, u_f, T_f, max_k=6)
        Gp = G_plus_f
        Gm = G_minus_f
        G = Gp + Gm

        half_dx = 0.5 * dx
        W_i = W_pad[1:-2]
        W_ip1 = W_pad[2:-1]

        drho_L = (W_face[:, 0] - W_i[:, 0]) / half_dx
        drho_R = (W_ip1[:, 0] - W_face[:, 0]) / half_dx
        dm_L = (W_face[:, 1] - W_i[:, 1]) / half_dx
        dm_R = (W_ip1[:, 1] - W_face[:, 1]) / half_dx

        if dim_w == 4:
            rho_p, ux_p, uy_p, T_p = self._w_to_primitive(W_pad)
            E1D_pad = 0.5 * rho_p * ux_p**2 + 0.5 * rho_p * self.R * T_p
            E1D_i = E1D_pad[1:-2]
            E1D_ip1 = E1D_pad[2:-1]
            E1D_face = 0.5 * G_plus_L[2] + 0.5 * G_minus_R[2]
            dE1D_L = (E1D_face - E1D_i) / half_dx
            dE1D_R = (E1D_ip1 - E1D_face) / half_dx
        else:
            E1D_face = W_face[:, 2]
            dE1D_L = (E1D_face - W_i[:, 2]) / half_dx
            dE1D_R = (W_ip1[:, 2] - E1D_face) / half_dx

        u2, u3, u4 = u_f**2, u_f**3, u_f**4
        RT = self.R * T_f
        RT2 = RT**2
        rho_safe = np.maximum(rho_f, 1e-15)

        def _compute_a(drho, dm, dE):
            inv_rho = np.where(rho_safe > 1e-8, 1.0 / rho_safe, 0.0)

            a0 = (
                (1.5 + u4 / (2 * RT2)) * inv_rho * drho
                - (u3 / RT2) * inv_rho * dm
                + (u2 / RT - 1) * (inv_rho / RT) * dE
            )
            a1 = (
                -(u3 / RT2) * inv_rho * drho
                + (1 + 2 * u2 / RT) * (inv_rho / RT) * dm
                - (2 * u_f / RT2) * inv_rho * dE
            )
            a2 = (
                (u2 / RT2 - 1 / RT) * inv_rho * drho
                - (2 * u_f / RT2) * inv_rho * dm
                + (2 / RT2) * inv_rho * dE
            )
            return a0, a1, a2

        a0_L, a1_L, a2_L = _compute_a(drho_L, dm_L, dE1D_L)
        a0_R, a1_R, a2_R = _compute_a(drho_R, dm_R, dE1D_R)

        S1_L = a0_L * Gp[1] + a1_L * Gp[2] + 0.5 * a2_L * Gp[3]
        S2_L = a0_L * Gp[2] + a1_L * Gp[3] + 0.5 * a2_L * Gp[4]
        S3_L = a0_L * 0.5 * Gp[3] + a1_L * 0.5 * Gp[4] + 0.5 * a2_L * 0.5 * Gp[5]

        S1_R = a0_R * Gm[1] + a1_R * Gm[2] + 0.5 * a2_R * Gm[3]
        S2_R = a0_R * Gm[2] + a1_R * Gm[3] + 0.5 * a2_R * Gm[4]
        S3_R = a0_R * 0.5 * Gm[3] + a1_R * 0.5 * Gm[4] + 0.5 * a2_R * 0.5 * Gm[5]

        dt_rho = -(S1_L + S1_R)
        dt_m = -(S2_L + S2_R)
        dt_E1D = -(S3_L + S3_R)

        a0_prime = (
            (1.5 + u4 / (2 * RT2)) * (1 / rho_safe) * dt_rho
            - (u3 / (rho_safe * RT2)) * dt_m
            + (u2 / RT - 1) * (1 / (rho_safe * RT)) * dt_E1D
        )
        a1_prime = (
            -(u3 / RT2) * (1 / rho_safe) * dt_rho
            + (1 + 2 * u2 / RT) * (1 / (rho_safe * RT)) * dt_m
            - (2 * u_f) / (rho_safe * RT2) * dt_E1D
        )
        a2_prime = (
            (u2 / RT2 - 1 / RT) * (1 / rho_safe) * dt_rho
            - 2 * u_f / (rho_safe * RT2) * dt_m
            + 2 / (rho_safe * RT2) * dt_E1D
        )

        if isinstance(tau_cells, np.ndarray) and len(tau_cells) > 1:
            tau_pad = np.pad(tau_cells, (1, 1), mode="edge")
            tau_face = 0.5 * (tau_pad[:-1] + tau_pad[1:])
        else:
            tau_face = tau_cells

        _, _, C3, C4, C5 = self._get_time_averaged_coeffs(dt, tau_face)

        F_eq = np.zeros_like(W_face)

        F_eq[:, 0] = (
            C3 * G[1]
            + C4 * (S1_L + S1_R)
            + C5 * (a0_prime * G[1] + a1_prime * G[2] + 0.5 * a2_prime * G[3])
        )

        F_eq[:, 1] = (
            C3 * G[2]
            + C4 * (S2_L + S2_R)
            + C5 * (a0_prime * G[2] + a1_prime * G[3] + 0.5 * a2_prime * G[4])
        )

        F_E1D_part = (
            C3 * 0.5 * G[3]
            + C4 * (S3_L + S3_R)
            + C5 * 0.5 * (a0_prime * G[3] + a1_prime * G[4] + 0.5 * a2_prime * G[5])
        )

        if dim_w == 4:
            F_eq[:, 2] = (
                C3 * (uy_f * G[1])
                + C4 * uy_f * (S1_L + S1_R)
                + C5
                * uy_f
                * (a0_prime * G[1] + a1_prime * G[2] + 0.5 * a2_prime * G[3])
            )

            Hy_f = 0.5 * (uy_f**2 + self.R * T_f)
            F_Hy_part = (
                C3 * (Hy_f * G[1])
                + C4 * Hy_f * (S1_L + S1_R)
                + C5
                * Hy_f
                * (a0_prime * G[1] + a1_prime * G[2] + 0.5 * a2_prime * G[3])
            )

            F_eq[:, 3] = F_E1D_part + F_Hy_part
        else:
            F_eq[:, 2] = F_E1D_part

        return F_eq

    def compute_tau(self, W):
        prims = self._w_to_primitive(W)
        rho, T = prims[0], prims[-1]
        p = rho * self.R * T
        mu = T**self.omega
        tau = self.tau_ref * mu / p
        return tau


class LinearUGKPSolver(UGKPSolver):
    def __init__(
        self,
        config,
        u_bg: np.ndarray,
        T_bg: np.ndarray,
        rho_bg=1.0,
        uy_bg=None,
        target_ppc=None,
        rho_in: float = None,
        u_in: float = None,
        T_in: float = None,
    ):
        super().__init__(config)
        self.u_bg = u_bg
        self.T_bg = T_bg
        self.rho_bg = rho_bg
        self.uy_bg = uy_bg if uy_bg is not None else np.zeros_like(u_bg)
        self.tau_cells = self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / self.rho_bg
        self.target_ppc = target_ppc

        self.rho_in = rho_in
        self.u_in = u_in
        self.T_in = T_in
        self.reflectance_left = float(config.physics.reflectance_left)

    def step(self, particles, macro_grid, dt: float):
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
                )

        n_cells = particles.n_cells
        dx = particles.dx

        cell_indices = particles.get_cell_indices()
        tau_p = self.tau_cells[cell_indices]

        eta = np.random.rand(particles.N_total)
        t_c = -tau_p * np.log(eta)

        collisionless_mask = t_c >= dt
        if hasattr(particles, "is_fresh"):
            collisionless_mask = collisionless_mask | particles.is_fresh
        collisional_mask = ~collisionless_mask

        x_coll = particles.x[collisional_mask]
        v_coll = particles.v[collisional_mask]
        m_coll = particles.m[collisional_mask]
        t_c_coll = t_c[collisional_mask]

        x_coll_new = x_coll + v_coll * t_c_coll
        x_coll_new, v_coll_new, keep_mask = particles.apply_bc_to_positions(
            x_coll_new, v_coll
        )
        m_coll_active = np.where(keep_mask, m_coll, 0.0)
        cell_indices_coll = particles.get_cell_indices_from_positions(x_coll_new)

        rho_coll = np.zeros(n_cells)
        if len(x_coll_new) > 0:
            rho_coll = (
                np.bincount(cell_indices_coll, weights=m_coll_active, minlength=n_cells)
                / dx
            )

        x_free = particles.x[collisionless_mask]
        v_free = particles.v[collisionless_mask]
        m_free = particles.m[collisionless_mask]
        if particles.dim_v == 2:
            vy_free = particles.vy[collisionless_mask]

        is_fresh_free = particles.is_fresh[collisionless_mask]

        particles.x = x_free
        particles.v = v_free
        if particles.dim_v == 2:
            particles.vy = vy_free
        particles.m = m_free

        particles.is_fresh = is_fresh_free
        particles.N_total = len(x_free)

        particles.advect(dt)

        particles.is_fresh[:] = False

        new_cell_indices = particles.get_cell_indices()
        rho_Pf = np.zeros(n_cells)
        if particles.N_total > 0:
            rho_Pf = (
                np.bincount(new_cell_indices, weights=particles.m, minlength=n_cells)
                / dx
            )

        rho_total_particles = rho_Pf + rho_coll

        rho_macro = np.maximum(macro_grid.W[:, 0] - rho_Pf, 0.0)

        W_cells = np.zeros((n_cells, 4))
        W_cells[:, 0] = rho_macro
        W_cells[:, 1] = rho_macro * self.u_bg
        W_cells[:, 2] = rho_macro * self.uy_bg
        W_cells[:, 3] = rho_macro * (
            0.5 * (self.u_bg**2 + self.uy_bg**2) + self.R * self.T_bg
        )

        F_eq_mass = self._compute_linear_equilibrium_mass_flux(
            W_cells, dt, self.tau_cells, dx
        )

        rho_n1 = rho_total_particles - (dt / dx) * (F_eq_mass[1:] - F_eq_mass[:-1])
        rho_n1 = np.maximum(rho_n1, 1e-12)

        macro_grid.W[:, 0] = rho_n1
        rho_h = np.maximum(rho_n1 - rho_Pf, 0.0)

        N_to_spawn = np.sum(keep_mask)

        cells_to_spawn = np.zeros(n_cells, dtype=int)
        valid_h_cells = rho_h > 1e-8
        mass_h = np.where(valid_h_cells, rho_h * dx, 0.0)
        total_mass_h = np.sum(mass_h)

        if total_mass_h > 1e-14:
            cells_to_spawn[valid_h_cells] = 1

            remaining_budget = max(0, N_to_spawn - np.sum(cells_to_spawn))
            if remaining_budget > 0:
                exact_counts = remaining_budget * (mass_h / total_mass_h)
                extra_spawn = np.floor(exact_counts).astype(int)
                cells_to_spawn += extra_spawn

                shortfall = remaining_budget - np.sum(extra_spawn)
                if shortfall > 0:
                    remainders = exact_counts - extra_spawn
                    largest_rem_idx = np.argsort(remainders)[-shortfall:]
                    cells_to_spawn[largest_rem_idx] += 1

        total_new = int(np.sum(cells_to_spawn))

        if total_new > 0:
            new_cell_indices = np.repeat(np.arange(n_cells), cells_to_spawn)

            safe_spawn = np.maximum(cells_to_spawn, 1)
            exact_mass_per_cell = (rho_h * dx) / safe_spawn
            new_m = exact_mass_per_cell[new_cell_indices]

            u_pos = np.random.uniform(0.0, 1.0, size=total_new)
            cell_left_edges = particles.xL + new_cell_indices * dx
            new_x = cell_left_edges + u_pos * dx

            Ux_mapped = self.u_bg[new_cell_indices]
            T_mapped = self.T_bg[new_cell_indices]
            std_dev_mapped = np.sqrt(self.R * T_mapped)
            new_v = np.random.normal(loc=Ux_mapped, scale=std_dev_mapped)

            vmax_grid = self.config.grid.vmax
            new_v = np.clip(new_v, -vmax_grid, vmax_grid)

            particles.x = np.concatenate([particles.x, new_x])
            particles.v = np.concatenate([particles.v, new_v])

            if particles.dim_v == 2:
                Uy_mapped = self.uy_bg[new_cell_indices]
                new_vy = np.random.normal(loc=Uy_mapped, scale=std_dev_mapped)
                new_vy = np.clip(new_vy, -vmax_grid, vmax_grid)
                particles.vy = np.concatenate([particles.vy, new_vy])

            particles.m = np.concatenate([particles.m, new_m])
            particles.is_fresh = np.concatenate(
                [particles.is_fresh, np.zeros(total_new, dtype=bool)]
            )

        particles.N_total = len(particles.x)
        return particles, macro_grid

    def _compute_linear_equilibrium_mass_flux(self, W_cells, dt, tau_cells, dx):
        rho_cells = W_cells[:, 0]
        rho_pad = np.zeros(len(rho_cells) + 4)
        rho_pad[2:-2] = rho_cells

        u_pad = np.zeros(len(self.u_bg) + 4)
        u_pad[2:-2] = self.u_bg

        T_pad = np.zeros(len(self.T_bg) + 4)
        T_pad[2:-2] = self.T_bg

        bc_type = getattr(self.config.grid, "bc_type", "open")
        R_n = self.reflectance_left

        if bc_type in ("inflow/outflow", "open") and self.rho_in is not None:
            rho_0, u0, T0 = rho_cells[0], self.u_bg[0], self.T_bg[0]
            _, G_m = self._compute_analytical_moments(
                np.atleast_1d(rho_0), np.atleast_1d(u0), np.atleast_1d(T0), max_k=1
            )

            rho_pad[:2] = R_n * rho_cells[0]
            u_pad[:2] = self.u_bg[0]
            T_pad[:2] = self.T_bg[0]
            rho_pad[-2:] = rho_cells[-1]
            u_pad[-2:] = self.u_bg[-1]
            T_pad[-2:] = self.T_bg[-1]

        else:
            rho_pad[:2], rho_pad[-2:] = rho_cells[0], rho_cells[-1]
            u_pad[:2], u_pad[-2:] = self.u_bg[0], self.u_bg[-1]
            T_pad[:2], T_pad[-2:] = self.T_bg[0], self.T_bg[-1]

        s1 = (rho_pad[1:-1] - rho_pad[0:-2]) / dx
        s2 = (rho_pad[2:] - rho_pad[1:-1]) / dx
        slopes_rho = np.zeros_like(rho_pad)
        slopes_rho[1:-1] = (
            (np.sign(s1) + np.sign(s2))
            * (np.abs(s1) * np.abs(s2))
            / (np.abs(s1) + np.abs(s2) + 1e-15)
        )

        rho_L = rho_pad[1:-2] + 0.5 * dx * slopes_rho[1:-2]
        rho_R = rho_pad[2:-1] - 0.5 * dx * slopes_rho[2:-1]
        u_face = 0.5 * (u_pad[1:-2] + u_pad[2:-1])
        T_face = 0.5 * (T_pad[1:-2] + T_pad[2:-1])

        G_plus_L, _ = self._compute_analytical_moments(rho_L, u_face, T_face, max_k=2)
        _, G_minus_R = self._compute_analytical_moments(rho_R, u_face, T_face, max_k=2)
        rho_face = G_plus_L[0] + G_minus_R[0]

        G_plus, G_minus = self._compute_analytical_moments(
            rho_face, u_face, T_face, max_k=5
        )
        G = G_plus + G_minus

        half_dx = 0.5 * dx
        drho_L = (rho_face - rho_pad[1:-2]) / half_dx
        drho_R = (rho_pad[2:-1] - rho_face) / half_dx

        du_dx = (u_pad[2:-1] - u_pad[1:-2]) / dx
        dT_dx = (T_pad[2:-1] - T_pad[1:-2]) / dx

        dm_L = drho_L * u_face + rho_face * du_dx
        dm_R = drho_R * u_face + rho_face * du_dx

        dE1D_L = 0.5 * (
            drho_L * u_face**2
            + 2 * rho_face * u_face * du_dx
            + drho_L * self.R * T_face
            + rho_face * self.R * dT_dx
        )
        dE1D_R = 0.5 * (
            drho_R * u_face**2
            + 2 * rho_face * u_face * du_dx
            + drho_R * self.R * T_face
            + rho_face * self.R * dT_dx
        )

        u_f, T_f = u_face, T_face
        u2, u3, u4 = u_f**2, u_f**3, u_f**4
        RT, RT2 = self.R * T_f, (self.R * T_f) ** 2
        rho_safe = np.maximum(rho_face, 1e-15)

        def _compute_a(drho, dm, dE):
            inv_rho = np.where(rho_safe > 1e-8, 1.0 / rho_safe, 0.0)

            a0 = (
                (1.5 + u4 / (2 * RT2)) * inv_rho * drho
                - (u3 / RT2) * inv_rho * dm
                + (u2 / RT - 1) * (inv_rho / RT) * dE
            )
            a1 = (
                -(u3 / RT2) * inv_rho * drho
                + (1 + 2 * u2 / RT) * (inv_rho / RT) * dm
                - (2 * u_f / RT2) * inv_rho * dE
            )
            a2 = (
                (u2 / RT2 - 1 / RT) * inv_rho * drho
                - (2 * u_f / RT2) * inv_rho * dm
                + (2 / RT2) * inv_rho * dE
            )
            return a0, a1, a2

        a0_L, a1_L, a2_L = _compute_a(drho_L, dm_L, dE1D_L)
        a0_R, a1_R, a2_R = _compute_a(drho_R, dm_R, dE1D_R)

        S1_L = a0_L * G_plus[1] + a1_L * G_plus[2] + 0.5 * a2_L * G_plus[3]
        S1_R = a0_R * G_minus[1] + a1_R * G_minus[2] + 0.5 * a2_R * G_minus[3]

        S2_L = a0_L * G_plus[2] + a1_L * G_plus[3] + 0.5 * a2_L * G_plus[4]
        S2_R = a0_R * G_minus[2] + a1_R * G_minus[3] + 0.5 * a2_R * G_minus[4]

        S3_L = (
            a0_L * 0.5 * G_plus[3]
            + a1_L * 0.5 * G_plus[4]
            + 0.5 * a2_L * 0.5 * G_plus[5]
        )
        S3_R = (
            a0_R * 0.5 * G_minus[3]
            + a1_R * 0.5 * G_minus[4]
            + 0.5 * a2_R * 0.5 * G_minus[5]
        )

        dt_rho = -(S1_L + S1_R)
        dt_m = -(S2_L + S2_R)
        dt_E1D = -(S3_L + S3_R)

        a0_prime = (
            (1.5 + u4 / (2 * RT2)) * (1 / rho_safe) * dt_rho
            - (u3 / (rho_safe * RT2)) * dt_m
            + (u2 / RT - 1) * (1 / (rho_safe * RT)) * dt_E1D
        )
        a1_prime = (
            -(u3 / RT2) * (1 / rho_safe) * dt_rho
            + (1 + 2 * u2 / RT) * (1 / (rho_safe * RT)) * dt_m
            - (2 * u_f) / (rho_safe * RT2) * dt_E1D
        )
        a2_prime = (
            (u2 / RT2 - 1 / RT) * (1 / rho_safe) * dt_rho
            - 2 * u_f / (rho_safe * RT2) * dt_m
            + 2 / (rho_safe * RT2) * dt_E1D
        )

        if isinstance(tau_cells, np.ndarray) and len(tau_cells) > 1:
            tau_pad = np.pad(tau_cells, (1, 1), mode="edge")
            tau_face = 0.5 * (tau_pad[:-1] + tau_pad[1:])
        else:
            tau_face = tau_cells

        _, _, C3, C4, C5 = self._get_time_averaged_coeffs(dt, tau_face)

        F_eq_mass = (
            C3 * G[1]
            + C4 * (S1_L + S1_R)
            + C5 * (a0_prime * G[1] + a1_prime * G[2] + 0.5 * a2_prime * G[3])
        )

        rho_0, u0, T0 = rho_cells[0], self.u_bg[0], self.T_bg[0]
        G_p_0, G_m_0 = self._compute_analytical_moments(
            np.atleast_1d(rho_0), np.atleast_1d(u0), np.atleast_1d(T0), max_k=1
        )
        F_out = G_m_0[1, 0]  # negative (mass leaving leftward)
        F_eq_mass[0] = (1.0 - R_n) * F_out

        return F_eq_mass
