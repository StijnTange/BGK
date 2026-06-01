import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver


class ConservationTrackerHook:
    """
    Custom hook to calculate and store macroscopic variables at each step.
    """

    def __init__(self, dx, R, is_node_based):
        self.dx = dx
        self.R = R
        self.is_node_based = is_node_based
        self.times = []
        self.mass = []
        self.momentum = []
        self.energy = []

    def __call__(self, runner):
        # Extract macroscopics directly from the runner's distribution function
        rho, p, E = runner.df.compute_moments()

        if self.is_node_based:
            # Omit the last point for periodic grid-based methods to prevent double-counting
            # since x[0] and x[-1] represent the exact same physical location
            mass_tot = np.sum(rho[:-1]) * self.dx
            mom_tot = np.sum(p[:-1]) * self.dx
            ener_tot = np.sum(E[:-1]) * self.dx
        else:
            # Cell-based methods do not overlap, sum everything
            mass_tot = np.sum(rho) * self.dx
            mom_tot = np.sum(p) * self.dx
            ener_tot = np.sum(E) * self.dx

        self.times.append(runner.t)
        self.mass.append(mass_tot)
        self.momentum.append(mom_tot)
        self.energy.append(ener_tot)


def main():
    # 1. Setup Configuration
    problem_name = "gaussian"
    problem = get_problem(problem_name)

    SAVE_PLOTS = True

    CFL = 0.9
    target_dt = 1e-3
    vmax = 10.0
    t_final = 5.0
    Kn = 1e-3
    Nv = 200
    R = 1.0

    L = problem.x_bounds[1] - problem.x_bounds[0]

    # Calculate a base resolution based on the target CFL and dt
    target_dx = target_dt * vmax / CFL
    N_res = int(np.ceil(L / target_dx))

    # 2. Define Solvers to Test
    solvers = {
        "Strang (Lagrangian)": StrangSolver,
        "SL (Lagrangian)": SLSolver,
        "FVM": FVMSolver,
        "UGKS": UGKSSolver,
        "Hybrid (Lagrangian)": HybridSolver,
    }

    results = {}

    physics_conf = PhysicsConfig(
        Kn=Kn, problem_name=problem_name, R=R, constant_tau=True
    )

    # 3. Run Experiments
    for name, SolverClass in solvers.items():
        print(f"\n--- Running {name} solver ---")

        # --- Grid switching logic ---
        if name in ["Strang (Lagrangian)", "SL (Lagrangian)", "Hybrid (Lagrangian)"]:
            # Grid-based methods
            Nx = N_res
            Nc = None
            dx = L / (Nx - 1)
            is_node_based = True
        elif name in ["FVM", "UGKS"]:
            # Cell-based methods
            Nx = None
            Nc = N_res
            dx = L / Nc
            is_node_based = False
        else:
            raise ValueError(f"Unknown method: {name}")

        # Ensure dt is consistent with the calculated dx
        dt = CFL * dx / vmax

        print(f"Setting up grid with Nx={Nx}, Nc={Nc}, dx={dx:.4f}")

        grid_conf = GridConfig(
            xL=problem.x_bounds[0],
            xR=problem.x_bounds[1],
            Nx=Nx,
            Nc=Nc,
            Nv=[Nv],
            dim_v=1,
            vmax=vmax,
            vmin=-vmax,
            bc_type="periodic",
        )

        time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
        config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

        solver = SolverClass(config)
        runner = Runner(config=config, solver=solver, problem=problem)

        # Add our custom tracking hook
        tracker = ConservationTrackerHook(dx=dx, R=R, is_node_based=is_node_based)
        runner.add_hook(tracker)

        runner.run()
        results[name] = tracker

    # 4. Plot Relative Errors (1x3 Horizontal Layout)

    # Get full page width. Set a smaller height (e.g., 2.5 or 2.8 inches)
    # so the 3 plots remain roughly square-shaped and don't dominate the page.
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(width, 2.8), constrained_layout=True)

    variables = ["mass", "momentum", "energy"]

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
    ]  # Circle, Square, Diamond, Triangle, Inverted Triangle

    for idx, var in enumerate(variables):
        ax = axs[idx]
        for name, tracker in results.items():
            data = np.array(getattr(tracker, var.lower()))

            # Calculate relative error
            initial_val = data[0] if abs(data[0]) > 1e-14 else 1.0
            rel_error = np.abs(data - data[0]) / np.abs(initial_val)

            # Use simple solid lines since there are 5000 data points
            ax.plot(
                tracker.times,
                rel_error,
                label=name,
                color=colors[list(results.keys()).index(name)],
            )

        ax.set_ylabel(rf"Relative {var} error")
        ax.set_xlabel(r"Time $t$")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

    # We only need one legend. Placing it on the last plot (Energy) is standard.
    # It will automatically inherit the white background and black border from thesis_plots.
    axs[0].legend()

    # Save as a single, perfectly scaled PDF
    filename = "latex/thesis/figures/ch4/conservation/comparison_lagrangian.pdf"
    if SAVE_PLOTS:
        tp.save_plot(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
