import numpy as np
from bgk.core.distribution import DistributionFunction
from bgk.core.grid import Grid
from bgk.core.simulation import Simulation
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class Runner:
    def __init__(self, config, solver, problem):
        self.config = config
        self.solver = solver
        self.problem = problem
        self.dim_v = config.grid.dim_v

        # create grid
        self.grid = Grid(config.grid)

        # initialize distribution function
        self.df = DistributionFunction(self.grid, config)
        self.df.initialize(problem.f0_func)

        # simulation state
        self.t = 0.0
        self.dt = config.time.dt
        self.hooks = []

        # storage
        self.history = {
            "t": [],
            "rho": [],
            "u": [],
            "T": [],
            "q": [],
        }

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
            task = progress.add_task(f"Solving {self.problem.name}...", total=t_final)

            while self.t < t_final - 1e-12:
                current_dt = self.dt
                if self.t + current_dt > t_final:
                    current_dt = t_final - self.t

                self.df = self.solver.step(df=self.df, dt=current_dt)
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
        macros = self.df.compute_macroscopics(R=self.config.physics.R)
        rho = macros[:1]
        u = macros[1 : 1 + self.dim_v]
        T = macros[-1:]
        q = self.df.compute_heat_flux(u=u)

        self.history["t"].append(self.t)
        self.history["rho"].append(rho)
        self.history["u"].append(u)
        self.history["T"].append(T)
        self.history["q"].append(q)

    def _create_simulation_object(self):
        if self.dim_v == 1:
            v = self.grid.v
        else:
            v = [self.grid.vx, self.grid.vy]

        return Simulation(
            x=self.grid.x,
            v=v,
            f=self.df.f.copy(),
            t=np.array(self.history["t"]),
            dt=self.dt,
            Kn=self.config.physics.Kn,
            omega=self.config.physics.omega,
            R=self.config.physics.R,
            rho=np.array(self.history["rho"]),
            u=np.array(self.history["u"]),
            T=np.array(self.history["T"]),
            q=np.array(self.history["q"]),
        )
