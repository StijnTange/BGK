import os
import sys
import time

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
    """Runs a simulation and returns the final simulation object, execution time, and dx."""
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

    start_time = time.perf_counter()
    sim = runner.run()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return sim, execution_time, dx


def main():
    problem = get_problem("gaussian")
    SAVE_PLOTS = True

    # ---------------------------------------------------------
    # CONFIGURATION BLOCK
    # ---------------------------------------------------------
    Kns = [1e-8, 1e-3, 1e2]
    methods = ["Strang", "SL", "Hybrid"]

    CFL_LIST = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    N_grids = [100, 200, 400, 800]

    solver_classes = {
        "Strang": StrangSolver,
        "SL": SLSolver,
        "FVM": FVMSolver,
        "UGKS": UGKSSolver,
        "Hybrid": HybridSolver,
    }

    # Define color maps for each method.
    # Linspace goes backwards (1.0 down to 0.35) so low CFLs get the most
    # saturated/brightest colors and high CFLs fade out to lighter colors.
    # We use RdPu (Red-Purple) for Hybrid to make it warmer and distinct from Blues.
    color_maps = {
        "Strang": plt.cm.Blues(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "SL": plt.cm.Reds(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "Hybrid": plt.cm.RdPu(np.linspace(1.0, 0.35, len(CFL_LIST))),
    }

    markers = ["o", "^", "s", "D", "v", "p", "*", "X"]
    line_styles = ["-", "--", ":", "-."]
    # ---------------------------------------------------------

    # Setup the 3x3 plot figure
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(3, 3, figsize=(width, width * 0.8), constrained_layout=True)

    # Resolution for the Truth solution
    N_res_ref = N_grids[-1] * 16
    cfl_ref = 0.9

    for col, Kn in enumerate(Kns):
        print(f"\n{'=' * 60}")
        print(f" GENERATING TRUTH SOLUTION FOR Kn = {Kn}")
        print(f"{'=' * 60}")

        # Generate Truth solution once per Knudsen number
        truth_sim, _, _ = run_simulation(
            "FVM", FVMSolver, N_res_ref, Kn, problem, cfl=cfl_ref
        )
        truth_x = truth_sim.x.flatten()
        truth_rho = truth_sim.rho[-1].flatten()

        for row, method_name in enumerate(methods):
            SolverClass = solver_classes[method_name]
            ax = axs[row, col]

            print(f"\n--- Testing {method_name} at Kn = {Kn} ---")

            for idx, cfl in enumerate(CFL_LIST):
                run_name = f"CFL={cfl}"

                times = []
                errors = []

                for N_res in N_grids:
                    sim, exec_time, dx = run_simulation(
                        method_name, SolverClass, N_res, Kn, problem, cfl
                    )

                    # Interpolate the Truth solution onto this specific coarse grid
                    coarse_x = sim.x.flatten()
                    coarse_rho = sim.rho[-1].flatten()
                    truth_projected = np.interp(coarse_x, truth_x, truth_rho)

                    # Calculate RELATIVE L1 Error based on True Integral
                    if method_name in ["Strang", "SL", "Hybrid"]:
                        # Node-based: Drop the overlapping periodic boundary point
                        abs_error = (
                            np.sum(np.abs(coarse_rho[:-1] - truth_projected[:-1])) * dx
                        )
                        truth_norm = np.sum(np.abs(truth_projected[:-1])) * dx
                    else:
                        # Cell-based: Sum all points
                        abs_error = np.sum(np.abs(coarse_rho - truth_projected)) * dx
                        truth_norm = np.sum(np.abs(truth_projected)) * dx

                    rel_error = abs_error / truth_norm

                    times.append(exec_time)
                    errors.append(rel_error)

                # Plot the line for this CFL
                color = color_maps[method_name][idx]
                marker = markers[idx % len(markers)]
                ls = line_styles[idx % len(line_styles)]

                ax.loglog(
                    times,
                    errors,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    label=f"CFL={cfl}",
                    markersize=4,
                    linewidth=1.2,
                )

            # Formatting specific to this subplot
            ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

            # Formatting Rows (Y-labels)
            if col == 0:
                ax.set_ylabel(f"{method_name}\nRelative $L_1$ error")
            else:
                ax.set_yticklabels(
                    []
                )  # Hide y-axis labels for inner plots to save space

            # Formatting Columns (Titles and X-labels)
            if row == 0:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$")
                ax.title.set_fontsize(10)

            if row == 2:
                ax.set_xlabel("Execution time (s)")
            else:
                ax.set_xticklabels([])  # Hide x-axis labels for top rows

            # Put a small legend in each plot, or just the first column
            if col == 0:
                ax.legend(loc="lower left")

    # Save the full 3x3 grid
    filename = "latex/thesis/figures/ch4/efficiency/work_precision_grid_comparison.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved 3x3 grid plot to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
