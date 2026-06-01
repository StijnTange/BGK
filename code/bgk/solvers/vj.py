import numpy as np
from bgk.solvers.base import Solver


def _advance_ray_tracing(
    particles,
    active_idx,
    t_rem,
    eta,
    cell_nu,
    nu_max_unused,
    on_collision,
):
    a_idx = active_idx
    x_act = particles.x[a_idx]
    v_act = particles.v[a_idx]
    t_act = t_rem[a_idx]
    eta_act = eta[a_idx]

    c_idx = particles.get_cell_indices_from_positions(x_act)
    nu_act = cell_nu[c_idx]

    v_safe = np.where(v_act == 0, 1e-14, v_act)
    x_left = particles.xL + c_idx * particles.dx
    x_right = x_left + particles.dx
    dist_to_face = np.where(v_safe > 0, x_right - x_act, x_act - x_left)
    dt_face = np.abs(dist_to_face / v_safe)

    dt_step = np.minimum(dt_face, t_act)
    d_eta = nu_act * dt_step

    collide_mask = eta_act < d_eta

    dt_actual = np.copy(dt_step)
    nu_safe = np.where(nu_act == 0, 1e-14, nu_act)
    dt_actual[collide_mask] = eta_act[collide_mask] / nu_safe[collide_mask]

    x_new = x_act + v_act * dt_actual
    v_new = v_act.copy()

    hit_face_mask = (~collide_mask) & (dt_face <= t_act)
    x_new[hit_face_mask] += np.sign(v_safe[hit_face_mask]) * 1e-10

    x_new, v_new, keep_mask = particles.apply_bc_to_positions(x_new, v_new)

    outflow_mask = ~keep_mask
    dt_actual[outflow_mask] = t_act[outflow_mask]
    collide_mask = collide_mask & keep_mask

    if np.any(collide_mask):
        c_idx_coll = c_idx[collide_mask]
        a_idx_coll = a_idx[collide_mask]
        old_v_coll = v_act[collide_mask]

        new_v_coll = on_collision(a_idx_coll, c_idx_coll, old_v_coll)
        v_new[collide_mask] = new_v_coll

        eta_act[collide_mask] = -np.log(np.random.rand(int(np.sum(collide_mask))))

    no_collide_mask = (~collide_mask) & keep_mask
    eta_act[no_collide_mask] -= d_eta[no_collide_mask]

    particles.x[a_idx] = x_new
    particles.v[a_idx] = v_new
    t_rem[a_idx] -= dt_actual
    eta[a_idx] = eta_act

    return t_rem, eta


def _advance_null_collision(
    particles,
    active_idx,
    t_rem,
    eta_unused,
    cell_nu,
    nu_max,
    on_collision,
):
    a_idx = active_idx
    x_act = particles.x[a_idx]
    v_act = particles.v[a_idx]
    t_act = t_rem[a_idx]
    N = len(a_idx)

    U1 = np.random.rand(N)
    U1 = np.maximum(U1, 1e-300)
    dt_prov = -np.log(U1) / nu_max

    past_end = dt_prov >= t_act
    dt_actual = np.where(past_end, t_act, dt_prov)

    x_new = x_act + v_act * dt_actual
    v_new = v_act.copy()

    x_new, v_new, keep_mask = particles.apply_bc_to_positions(x_new, v_new)
    outflow_mask = ~keep_mask
    dt_actual[outflow_mask] = t_act[outflow_mask]

    provisional_mask = keep_mask & (~past_end)

    if np.any(provisional_mask):
        c_idx_prov = particles.get_cell_indices_from_positions(x_new[provisional_mask])
        p_acc = cell_nu[c_idx_prov] / nu_max
        U2 = np.random.rand(int(np.sum(provisional_mask)))
        local_accept = U2 < p_acc

        if np.any(local_accept):
            idx_back = np.where(provisional_mask)[0][local_accept]
            a_idx_coll = a_idx[idx_back]
            c_idx_coll = c_idx_prov[local_accept]
            old_v_coll = v_act[idx_back]

            new_v_coll = on_collision(a_idx_coll, c_idx_coll, old_v_coll)
            v_new[idx_back] = new_v_coll

    particles.x[a_idx] = x_new
    particles.v[a_idx] = v_new
    t_rem[a_idx] -= dt_actual

    return t_rem, eta_unused


class LinearVelocityJumpSolver(Solver):
    def __init__(
        self,
        config,
        u_bg: np.ndarray,
        T_bg: np.ndarray,
        rho_bg: np.ndarray,
        target_N_total=None,
        use_null_collision: bool = False,
    ):
        self.config = config
        self.R = config.physics.R
        self.Kn = config.physics.Kn
        self.omega = config.physics.omega
        self.bc_type = config.grid.bc_type
        self.tau_ref = self.Kn * np.sqrt(2.0 / np.pi)

        self.u_bg = u_bg
        self.T_bg = T_bg
        self.rho_bg = rho_bg

        self.use_null_collision = use_null_collision

        cell_tau = self.tau_ref * (self.T_bg ** (self.omega - 1.0)) / self.rho_bg

        valid_cells = self.T_bg > 0.0
        self.cell_nu = np.zeros_like(cell_tau)
        self.cell_nu[valid_cells] = 1.0 / cell_tau[valid_cells]
        self.nu_max = float(np.max(self.cell_nu)) if np.any(valid_cells) else 0.0

        if target_N_total is not None and target_N_total > 0:
            self.target_ppc = int(target_N_total / config.grid.Nc)
        else:
            self.target_ppc = None

    def step(self, particles, dt: float, t_n=None, source_func=None):
        N_total_old = particles.N_total

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
        N_total_current = particles.N_total
        if N_total_current == 0:
            return particles

        def on_collision(a_idx_coll, c_idx_coll, old_v_coll):
            U_targ = self.u_bg[c_idx_coll]
            T_targ = self.T_bg[c_idx_coll]
            std_dev = np.sqrt(self.R * T_targ)
            return U_targ + std_dev * np.random.randn(len(c_idx_coll))

        advance = (
            _advance_null_collision
            if getattr(self, "use_null_collision", False)
            else _advance_ray_tracing
        )

        t_rem = np.full(N_total_current, dt)
        eta = -np.log(np.random.rand(N_total_current))

        N_injected = N_total_current - N_total_old
        if N_injected > 0:
            fresh_idx = np.arange(N_total_old, N_total_current)

            particles.x[fresh_idx] = particles.xL

            particles.v[fresh_idx] = np.abs(particles.v[fresh_idx])

            t_rem[fresh_idx] = dt * np.random.rand(N_injected)
            particles.x[fresh_idx] = particles.xL

        active = t_rem > 1e-13

        while np.any(active):
            active_idx = np.where(active)[0]

            x_act = particles.x[active_idx]
            v_act = particles.v[active_idx]
            t_rem_act = t_rem[active_idx].copy()

            t_hit = np.full(len(active_idx), np.inf)

            moving_left = v_act < -1e-12
            t_hit[moving_left] = (particles.xL - x_act[moving_left]) / v_act[
                moving_left
            ]

            moving_right = v_act > 1e-12
            t_hit[moving_right] = (particles.xR - x_act[moving_right]) / v_act[
                moving_right
            ]

            t_hit = np.maximum(t_hit + 1e-13, 0.0)

            t_step = np.minimum(t_rem_act, t_hit)

            t_rem[active_idx] = t_step

            t_rem_out, eta_out = advance(
                particles,
                active_idx,
                t_rem,
                eta,
                self.cell_nu,
                self.nu_max,
                on_collision,
            )

            consumed = t_step - t_rem_out[active_idx]
            t_rem[active_idx] = np.maximum(t_rem_act - consumed, 0.0)
            eta[active_idx] = eta_out[active_idx]

            inside = (particles.x >= particles.xL) & (particles.x <= particles.xR)
            if not np.all(inside):
                particles.x = particles.x[inside]
                particles.v = particles.v[inside]
                if particles.dim_v == 2:
                    particles.vy = particles.vy[inside]
                particles.m = particles.m[inside]
                if hasattr(particles, "is_fresh") and len(particles.is_fresh) == len(
                    inside
                ):
                    particles.is_fresh = particles.is_fresh[inside]
                particles.N_total = int(np.sum(inside))

                if len(inside) == len(t_rem):
                    t_rem = t_rem[inside]
                    eta = eta[inside]

            active = t_rem > 1e-13

        return particles
