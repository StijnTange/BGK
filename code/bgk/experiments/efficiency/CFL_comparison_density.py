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


def run_simulation(method_name, SolverClass, N_res, Kn, problem, cfl):
    """Runs a simulation and returns the final simulation object."""
    vmax = 10.0
    t_final = 0.2
    Nv = 40
    R = 1.0

    L = problem.x_bounds[1] - problem.x_bounds[0]

    # --- Grid switching logic ---
    if method_name in ["Strang", "SL", "Hybrid"]:
        # Grid-based methods
        Nx = N_res
        Nc = None
        dx = L / (Nx - 1)
    elif method_name in ["FVM", "UGKS"]:
        # Cell-based methods
        Nx = None
        Nc = N_res
        dx = L / Nc
    else:
        raise ValueError(f"Unknown method: {method_name}")

    dt = cfl * dx / vmax

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
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=cfl)
    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name, R=R)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    solver = SolverClass(config)
    runner = Runner(config=config, solver=solver, problem=problem)

    sim = runner.run()
    return sim


def main():
    problem = get_problem("gaussian")
    SAVE_PLOTS = True

    # ---------------------------------------------------------
    # CONFIGURATION BLOCK
    # ---------------------------------------------------------
    Kns = [1e-6, 1e-2, 1e2]
    methods = ["Strang", "SL", "Hybrid"]

    # Selected CFLs to show the degradation without cluttering the profile plot
    CFL_LIST = [10.0, 20.0, 50.0, 100.0]

    N_test = 200  # Coarse grid to clearly see numerical artifacts/diffusion
    N_ref = 800  # High resolution for the reference solution

    solver_classes = {
        "Strang": StrangSolver,
        "SL": SLSolver,
        "FVM": FVMSolver,
        "UGKS": UGKSSolver,
        "Hybrid": HybridSolver,
    }

    # Define color maps for each method.
    # Linspace goes backwards (1.0 down to 0.35) so low CFLs get the most
    # saturated/brightest colors and high CFLs fade out.
    color_maps = {
        "Strang": plt.cm.Blues(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "SL": plt.cm.Reds(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "Hybrid": plt.cm.RdPu(np.linspace(1.0, 0.35, len(CFL_LIST))),
    }

    markers = ["o", "^", "s", "D", "v"]
    line_styles = ["-", "--", ":", "-.", "--"]
    # ---------------------------------------------------------

    # Setup the 3x3 plot figure
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(3, 3, figsize=(width, width * 0.8), constrained_layout=True)

    for col, Kn in enumerate(Kns):
        print(f"\n{'=' * 60}")
        print(f" GENERATING TRUTH SOLUTION FOR Kn = {Kn}")
        print(f"{'=' * 60}")

        # Generate Truth solution once per Knudsen number using FVM
        truth_sim = run_simulation("FVM", FVMSolver, N_ref, Kn, problem, cfl=0.9)
        truth_x = truth_sim.x.flatten()
        truth_rho = truth_sim.rho[-1].flatten()

        for row, method_name in enumerate(methods):
            SolverClass = solver_classes[method_name]
            ax = axs[row, col]

            print(f"\n--- Testing {method_name} at Kn = {Kn} ---")

            # 1. Plot the Reference Solution (solid black line)
            ax.plot(
                truth_x,
                truth_rho,
                color="black",
                linestyle="-",
                linewidth=1,
                label="Reference",
                zorder=10,  # Ensure reference stays on top
            )

            # 2. Plot the different CFLs
            for idx, cfl in enumerate(CFL_LIST):
                run_name = f"CFL={cfl}"

                sim = run_simulation(method_name, SolverClass, N_test, Kn, problem, cfl)

                coarse_x = sim.x.flatten()
                coarse_rho = sim.rho[-1].flatten()

                color = color_maps[method_name][idx]
                marker = markers[idx % len(markers)]
                ls = line_styles[idx % len(line_styles)]

                ax.plot(
                    coarse_x,
                    coarse_rho,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    label=run_name,
                    markersize=2,
                    linewidth=1,
                    markevery=15,  # Spaces out the markers so the line is visible
                )

            # Formatting specific to this subplot
            ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

            # Formatting Rows (Y-labels)
            if col == 0:
                ax.set_ylabel(f"{method_name}\nDensity $\\rho$")
            else:
                ax.set_yticklabels(
                    []
                )  # Hide y-axis labels for inner plots to save space

            # Formatting Columns (Titles and X-labels)
            if row == 0:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$")
                # set title size
                ax.title.set_fontsize(10)

            if row == 2:
                ax.set_xlabel("Position $x$")
            else:
                ax.set_xticklabels([])  # Hide x-axis labels for top rows

            # Put a small legend in the first column to avoid clutter
            if col == 0:
                ax.legend(loc="best")

    # Save the full 3x3 grid
    filename = "latex/thesis/figures/ch4/efficiency/density_profile_grid_comparison.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved 3x3 density profile grid to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
