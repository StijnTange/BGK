import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.problems.problems import get_problem
from bgk.solvers.rtsm import RTSMSolver
from bgk.solvers.ugkp import UGKPSolver
from bgk.solvers.ugks import UGKSSolver
from bgk.solvers.vj import VelocityJumpSolver


def get_final_density(runner):
    """Helper to safely extract the final density array from different runner types."""
    if hasattr(runner, "df"):
        rho, _, _ = runner.df.compute_macroscopics()
    elif hasattr(runner, "grid") and hasattr(runner.grid, "W"):
        rho = runner.grid.W[:, 0]
    elif hasattr(runner, "particles"):
        rho, _, _ = runner.particles.compute_cell_moments()
    else:
        raise AttributeError("Runner does not have a recognizable fluid state.")
    return rho


def run_particle_sim(SolverClass, RunnerClass, Nc, Kn, problem, cfl, N_particles):
    """Runs a particle simulation and returns final density, execution time, and dx."""
    vmax = 10.0
    t_final = 0.2

    L = problem.x_bounds[1] - problem.x_bounds[0]
    dx = L / Nc
    dt = cfl * dx / vmax

    config = Config(
        grid=GridConfig(
            xL=problem.x_bounds[0],
            xR=problem.x_bounds[1],
            Nx=None,
            Nc=Nc,
            vmin=-vmax,
            vmax=vmax,
            Nv=[0],
            dim_v=1,
            bc_type=problem.bc_type,
        ),
        time=TimeConfig(t_final=t_final, dt=dt, CFL=cfl),
        physics=PhysicsConfig(Kn=Kn, problem_name=problem.name, R=1.0),
    )

    solver = SolverClass(config)
    runner = RunnerClass(config, solver, problem, N_particles)

    sim = runner.run()

    rho_final = get_final_density(runner)
    x_centers = sim.x

    return rho_final, x_centers


def main():
    problem_name = "gaussian"
    problem = get_problem(problem_name)
    SAVE_PLOTS = False

    # =========================================================
    # 1. Configuration Setup
    # =========================================================
    Kns = [1e-3]
    methods = ["UGKP", "RTSM", "VJ"]

    n_cols = len(Kns)
    n_rows = len(methods)

    # We use a coarse grid to clearly see CFL diffusion, and a high
    # particle count to minimize noise burying the actual profile shape.
    Nc_test = 20
    N_particles = 100000

    # Define which Knudsen numbers each method is allowed to run on
    ALLOWED_KNS = {
        "UGKP": [1e-6, 1e-2, 1e2],
        "RTSM": [1e-6, 1e-3, 1e2],
        "VJ": [1e-3, 1e2],  # Excluded from highly collisional regimes
    }

    # CFLs to plot for each method
    CFL_SETTINGS = {
        "UGKP": [0.5, 0.9],
        "RTSM": [0.9, 5.0, 10.0, 20.0],
        "VJ": [0.9, 2.0, 5.0, 10.0],
    }

    particle_solvers = {
        "UGKP": (UGKPSolver, UGKPRunner),
        "RTSM": (RTSMSolver, ParticleRunner),
        "VJ": (VelocityJumpSolver, ParticleRunner),
    }

    # Color gradients (fading from bright to light for higher CFLs)
    color_maps = {
        "UGKP": plt.cm.Reds(np.linspace(1.0, 0.4, max(len(CFL_SETTINGS["UGKP"]), 1))),
        "RTSM": plt.cm.Blues(np.linspace(1.0, 0.4, max(len(CFL_SETTINGS["RTSM"]), 1))),
        "VJ": plt.cm.Greens(np.linspace(1.0, 0.4, max(len(CFL_SETTINGS["VJ"]), 1))),
    }

    markers = ["o", "^", "s", "D", "v"]
    line_styles = ["-", "--", ":", "-."]

    # =========================================================
    # 2. Setup Figure
    # =========================================================
    width, _ = tp.get_figsize(fraction=1.0)
    fig_height = width * 0.3 * n_rows

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    Nc_ref = 1000
    vmax = 10.0

    for col, Kn in enumerate(Kns):
        print(f"\n{'=' * 60}")
        print(f" GENERATING TRUTH SOLUTION FOR Kn = {Kn}")
        print(f"{'=' * 60}")

        L = problem.x_bounds[1] - problem.x_bounds[0]
        dx_ref = L / Nc_ref
        dt_ref = 0.5 * dx_ref / vmax

        # Deterministic UGKS reference solution
        ref_config = Config(
            grid=GridConfig(
                xL=problem.x_bounds[0],
                xR=problem.x_bounds[1],
                Nx=None,
                Nc=Nc_ref,
                vmin=-vmax,
                vmax=vmax,
                Nv=[200],
                dim_v=1,
                bc_type=problem.bc_type,
            ),
            time=TimeConfig(t_final=0.2, dt=dt_ref, CFL=0.5),
            physics=PhysicsConfig(Kn=Kn, problem_name=problem_name, R=1.0),
        )

        ugks_solver = UGKSSolver(ref_config)
        ugks_runner = Runner(ref_config, ugks_solver, problem)
        ugks_sim = ugks_runner.run()

        truth_rho = get_final_density(ugks_runner)
        truth_x = ugks_sim.x

        for row, method_name in enumerate(methods):
            ax = axs[row, col]

            # --- FILTER LOGIC ---
            if Kn not in ALLOWED_KNS.get(method_name, Kns):
                ax.axis("off")  # Hide the subplot entirely if not allowed
                continue

            print(f"\n--- Testing {method_name} at Kn = {Kn} ---")

            # 1. Plot the Reference Solution (solid black line)
            ax.plot(
                truth_x,
                truth_rho,
                color="black",
                linestyle="-",
                linewidth=1.5,
                label="Reference",
                zorder=10,
            )

            SolverClass, RunnerClass = particle_solvers[method_name]

            # 2. Plot the stochastic runs
            for cfl_idx, cfl in enumerate(CFL_SETTINGS[method_name]):
                print(f"Running {method_name} (CFL={cfl})...")

                rho_final, coarse_x = run_particle_sim(
                    SolverClass, RunnerClass, Nc_test, Kn, problem, cfl, N_particles
                )

                color = color_maps[method_name][cfl_idx]
                marker = markers[cfl_idx % len(markers)]
                ls = line_styles[cfl_idx % len(line_styles)]

                ax.plot(
                    coarse_x,
                    rho_final,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markevery=10,  # Space out the markers
                    label=f"CFL={cfl}",
                    markersize=4,
                    linewidth=1.2,
                )

            # Formatting specific to this subplot
            ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

            # Left column gets Y-labels
            if col == 0 or (col == 1 and method_name == "VJ"):
                # Ensure VJ gets a label even if its first column is skipped
                ax.set_ylabel(f"{method_name}\nDensity $\\rho$")
            elif col > 0 and not (col == 1 and method_name == "VJ"):
                ax.set_yticklabels([])

            # Top row gets titles
            if row == 0:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$")
            # If VJ is the top row of a column (e.g. if we skip UGKP/RTSM), give it a title
            elif row > 0 and axs[row - 1, col].axis() == False:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$")

            # Bottom row gets X-labels
            if row == n_rows - 1:
                ax.set_xlabel("Position $x$")
            else:
                ax.set_xticklabels([])

            # Put a legend in the leftmost active plot of each row
            if col == 0 or (method_name == "VJ" and col == 1):
                ax.legend(fontsize="x-small", loc="best", framealpha=0.9)

    filename = (
        "latex/thesis/figures/stochastic/profiles/density_stochastic_profiles.pdf"
    )

    if SAVE_PLOTS:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tp.save_plot(filename)
        print(f"\n✅ Saved stochastic profile grid plot to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
