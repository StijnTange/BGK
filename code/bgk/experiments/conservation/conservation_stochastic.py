import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.problems.problems import get_problem
from bgk.solvers.rtsm import RTSMSolver
from bgk.solvers.ugkp import UGKPSolver
from bgk.solvers.vj import VelocityJumpSolver


class ParticleConservationTrackerHook:
    """
    Calculates and stores macroscopic conserved variables for particle-based solvers.
    Adapted from the deterministic ConservationTrackerHook.
    """

    def __init__(self, dx, R):
        self.dx = dx
        self.R = R
        self.times = []
        self.mass = []
        self.momentum = []
        self.energy = []

    def __call__(self, runner):
        # 1. Fetch conserved variables W [rho, rho*u, E]
        if hasattr(runner, "grid") and hasattr(runner.grid, "W"):
            # UGKP: Use the deterministic macroscopic finite volume grid
            W = runner.grid.W
        elif hasattr(runner, "particles"):
            # RTSM: Compute moments from particles
            rho, u, T = runner.particles.compute_cell_moments()
            # Construct conserved W matching your 1D physics config
            W = np.column_stack(
                [rho, rho * u, 0.5 * rho * u**2 + 0.5 * rho * self.R * T]
            )
        else:
            return

        # 2. Integrate over the domain
        self.times.append(runner.t)
        self.mass.append(np.sum(W[:, 0]) * self.dx)
        self.momentum.append(np.sum(W[:, 1]) * self.dx)
        self.energy.append(np.sum(W[:, 2]) * self.dx)


def main():
    problem_name = "gaussian"  # Good test for flux-based conservation
    problem = get_problem(problem_name)

    SAVE_PLOTS = True

    # Use periodic BCs for a closed system to test numerical conservation
    bc_type = "periodic"
    vmax = 10.0
    dt = 1e-3
    CFL = 0.9
    dx = vmax * dt / CFL
    Nx = int(np.ceil((problem.x_bounds[1] - problem.x_bounds[0]) / dx))
    R = 1.0
    print(f"Calculated Nx={Nx} based on CFL condition with dx={dx:.4f}")

    config = Config(
        grid=GridConfig(
            xL=problem.x_bounds[0],
            xR=problem.x_bounds[1],
            Nx=Nx,
            vmin=-vmax,
            vmax=vmax,
            Nv=0,
            bc_type=bc_type,
        ),
        time=TimeConfig(t_final=0.1, dt=dt),
        physics=PhysicsConfig(Kn=1e-3, problem_name=problem_name, R=R),
    )

    # Define particle solvers to test
    particle_solvers = {
        "RTSM": (
            RTSMSolver,
            ParticleRunner,
        ),  # RTSM uses the base ParticleRunner with no special solver
        "VJ": (VelocityJumpSolver, ParticleRunner),
        "UGKP": (
            UGKPSolver,
            UGKPRunner,
        ),  # UGKP also uses the base ParticleRunner but with a different solver
    }

    results = {}
    N_particles = 10**6

    for name, (SolverClass, RunnerClass) in particle_solvers.items():
        print(f"\n--- Running {name} conservation test ---")
        solver = SolverClass(config)
        runner = RunnerClass(config, solver, problem, N_particles)

        tracker = ParticleConservationTrackerHook(dx=dx, R=R)
        runner.add_hook(tracker)
        runner.run()
        results[name] = tracker

    # =========================================================================
    # 4. Plot Relative Errors (Thesis Format - 1x3 Horizontal Layout)
    # =========================================================================
    # Get full page width, but keep the height short so plots remain square-ish
    plt.close("all")
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(width, 2.8), constrained_layout=True)

    variables = ["mass", "momentum", "energy"]

    for idx, var in enumerate(variables):
        ax = axs[idx]
        for name, tracker in results.items():
            data = np.array(getattr(tracker, var.lower()))
            initial_val = data[0] if abs(data[0]) > 1e-14 else 1.0

            # Relative error: |M(t) - M(0)| / |M(0)|
            rel_error = np.abs(data - data[0]) / np.abs(initial_val)

            # Removed manual linewidth=2 so it inherits your thesis_plots.py styling
            ax.plot(tracker.times, rel_error, label=name)

        ax.set_ylabel(rf"Relative {var} error")
        ax.set_xlabel(r"Time $t$")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4, linewidth=0.5)

    # Place a single legend in the final subplot (Energy)
    axs[2].legend()

    # Save as a perfectly scaled PDF instead of using plt.show()
    filename = "latex/thesis/figures/stochastic/conservation/comparison.pdf"
    if SAVE_PLOTS:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        tp.save_plot(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
