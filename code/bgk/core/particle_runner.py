import numpy as np
from bgk.core.grid import Grid
from bgk.core.particles import ParticleSystem
from bgk.core.simulation import Simulation
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class ParticleRunner:
    def __init__(self, config, solver, problem, Np, m_ref=None, N_inj=None):
        self.config = config
        self.solver = solver
        self.problem = problem

        self.grid = Grid(config.grid)

        # initialize particles with vy_init for 1D2V
        x_init, v_init, vy_init, m_weight = self._initialize_particles(
            Np=Np, explicit_m_ref=m_ref
        )
        self.particles = ParticleSystem(
            config, x_init, v_init, vy_init, m_weight, N_inj=N_inj
        )

        self.t = 0.0
        self.dt = config.time.dt
        self.hooks = []

        self.history = {"t": [], "rho": [], "u": [], "uy": [], "T": [], "q": []}

    def _initialize_particles(self, Np, explicit_m_ref=None):
        xL, xR = self.config.grid.xL, self.config.grid.xR
        vmin, vmax = self.config.grid.vmin, self.config.grid.vmax

        # handle Np = None
        if Np is None:
            m_weight = explicit_m_ref or getattr(self, "m_ref", None)

            if m_weight is None:
                raise ValueError(
                    "If Np is None (Vacuum), you MUST explicitly pass "
                    "m_ref into this function or set self.m_ref!"
                )

            print(f"Np is None. Initializing vacuum with m_weight = {m_weight:.4e}")
            return np.array([]), np.array([]), np.array([]), m_weight

        # integrate f0_func to find the total initial mass
        x_grid = np.linspace(xL, xR, 1000)
        v_grid = np.linspace(vmin, vmax, 1000)
        X, V = np.meshgrid(x_grid, v_grid)

        F0 = self.problem.f0_func(X, V)

        dx_int = x_grid[1] - x_grid[0]
        dv_int = v_grid[1] - v_grid[0]
        M_total = np.sum(F0) * dx_int * dv_int

        if M_total <= 1e-10:
            print("Warning: Initial mass is near zero. Starting with 0 particles.")
            nominal_mass = 1.0 * (xR - xL)
            m_weight = nominal_mass / Np
            return np.array([]), np.array([]), np.array([]), m_weight

        # general case
        m_weight = M_total / Np

        x_init = np.empty(Np)
        v_init = np.empty(Np)
        f_max = np.max(F0) * 1.1

        particles_accepted = 0
        batch_size = Np

        # acceptance-rejection to spawn exactly N particles
        while particles_accepted < Np:
            x_prop = np.random.uniform(xL, xR, batch_size)
            v_prop = np.random.uniform(vmin, vmax, batch_size)

            f_prop = self.problem.f0_func(x_prop, v_prop)
            y_test = np.random.uniform(0, f_max, batch_size)

            accepted_mask = y_test < f_prop

            x_acc = x_prop[accepted_mask]
            v_acc = v_prop[accepted_mask]

            take = min(len(x_acc), Np - particles_accepted)

            x_init[particles_accepted : particles_accepted + take] = x_acc[:take]
            v_init[particles_accepted : particles_accepted + take] = v_acc[:take]

            particles_accepted += take

        # handle 2D velocity (vy)
        if self.config.grid.dim_v == 2:
            if hasattr(self.problem, "T_bg_func"):
                T_init = self.problem.T_bg_func(x_init)
            else:
                T_init = np.ones_like(x_init)

            R = self.config.physics.R
            vy_init = np.random.normal(loc=0.0, scale=np.sqrt(R * T_init))
        else:
            vy_init = np.zeros(0)

        return x_init, v_init, vy_init, m_weight

    def add_hook(self, hook):
        self.hooks.append(hook)

    def run(self):
        t_final = self.config.time.t_final
        self._record_state()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Running particle simulation...", total=t_final
            )

            while self.t < t_final:
                current_dt = min(self.dt, t_final - self.t)

                self.particles = self.solver.step(
                    particles=self.particles,
                    dt=current_dt,
                )
                self.t += current_dt

                progress.update(task, completed=self.t)
                self._record_state()

                for hook in self.hooks:
                    hook(self)

        for hook in self.hooks:
            if hasattr(hook, "finalize"):
                hook.finalize()

        return self._create_simulation_object()

    def _record_state(self):
        if self.config.grid.dim_v == 2:
            rho, ux, uy, T = self.particles.compute_cell_moments()
        else:
            rho, ux, T = self.particles.compute_cell_moments()
            uy = np.zeros_like(rho)
        q = self.particles.compute_heat_flux()

        self.history["t"].append(self.t)
        self.history["rho"].append(rho)
        self.history["u"].append(ux)
        self.history["uy"].append(uy)
        self.history["T"].append(T)
        self.history["q"].append(q)

    def _create_simulation_object(self):
        kwargs = {
            "x": self.grid.x,
            "v": self.grid.v,
            "f": np.zeros((self.grid.Nc, self.grid.Nv[0])),
            "t": np.array(self.history["t"]),
            "dt": self.dt,
            "Kn": self.config.physics.Kn,
            "omega": self.config.physics.omega,
            "R": self.config.physics.R,
            "rho": np.array(self.history["rho"]),
            "u": np.array(self.history["u"]),
            "T": np.array(self.history["T"]),
            "q": np.array(self.history["q"]),
        }
        try:
            return Simulation(**kwargs, uy=np.array(self.history["uy"]))
        except TypeError:
            return Simulation(**kwargs)
