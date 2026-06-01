import os
import sys
import time

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


def get_final_temperature(runner):
    """Helper to safely extract the final temperature array from different runner types."""
    if hasattr(runner, "df"):
        macros = runner.df.compute_macroscopics()
        T = macros[-1]

    elif hasattr(runner, "grid") and hasattr(runner.grid, "W"):
        rho = runner.grid.W[:, 0]
        momentum = runner.grid.W[:, 1]
        E = runner.grid.W[:, 2]

        rho_safe = np.maximum(rho, 1e-12)
        u = momentum / rho_safe
        e_internal = (E / rho_safe) - 0.5 * u**2

        T_min = 1e-8
        T = np.maximum(2.0 * e_internal, T_min)

    elif hasattr(runner, "particles"):
        macros = runner.particles.compute_cell_moments()
        T = macros[-1]

    else:
        raise AttributeError("Runner does not have a recognizable fluid state.")

    return T


def run_particle_sim(SolverClass, RunnerClass, Nc, Kn, problem, cfl, N_particles):
    """Runs a particle simulation and returns final density, execution time, and dx."""
    vmax = 10.0
    t_final = 0.3

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

    start_time = time.perf_counter()
    sim = runner.run()
    end_time = time.perf_counter()

    cpu_time = end_time - start_time
    rho_final = get_final_density(runner)

    x_centers = sim.x

    return rho_final, x_centers, cpu_time, dx


def main():
    problem_name = "gaussian"
    problem = get_problem(problem_name)
    SAVE_PLOTS = False

    # =========================================================
    # 1. Configuration Setup
    # =========================================================
    # You can freely add/remove items from these lists
    Kns = [1e-3]
    N_grids = [20]

    n_cols = len(Kns)
    n_rows = len(N_grids)

    # Define which Knudsen numbers each method is allowed to run on.
    # This prevents VJ from choking on Kn=1e-6 without breaking 1-column layouts.
    ALLOWED_KNS = {
        "UGKP": [1e-6, 1e-2, 1e2],
        "RTSM": [1e-6, 1e-2, 1e2],
        "VJ": [1e-3, 1e2],  # Excluded from highly collisional regimes
    }

    CFL_SETTINGS = {
        "UGKP": [0.9],
        "RTSM": [0.9, 2.0, 5.0, 10.0],
        "VJ": [0.9, 2.0, 5.0, 10.0],
    }

    N_particles_list = [1000, 10000, 100000]

    particle_solvers = {
        "UGKP": (UGKPSolver, UGKPRunner),
        "RTSM": (RTSMSolver, ParticleRunner),
        "VJ": (VelocityJumpSolver, ParticleRunner),
    }

    colors = {"UGKP": "red", "RTSM": "blue", "VJ": "green"}
    markers = {"UGKP": "^", "RTSM": "o", "VJ": "s"}
    line_styles = ["-", "--", ":", "-."]

    # =========================================================
    # 2. Setup Figure
    # =========================================================
    width, _ = tp.get_figsize(fraction=1.0)

    # Scale height dynamically based on the number of rows
    fig_height = width * 0.3 * n_rows

    # squeeze=False ensures axs is ALWAYS a 2D array, even if n_rows=1 or n_cols=1
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

        for row, Nc in enumerate(N_grids):
            ax = axs[row, col]
            print(f"\n--- Testing Nc = {Nc} at Kn = {Kn} ---")

            for method_name, (SolverClass, RunnerClass) in particle_solvers.items():
                # --- FILTER LOGIC ---
                # Skip this solver if the current Kn is not in its allowed list
                allowed_kns_for_method = ALLOWED_KNS.get(method_name, Kns)
                if Kn not in allowed_kns_for_method:
                    continue

                for cfl_idx, cfl in enumerate(CFL_SETTINGS[method_name]):
                    times = []
                    errors = []

                    for N in N_particles_list:
                        dt = cfl * (L / Nc) / vmax
                        n_steps = int(np.ceil(0.2 / dt))
                        print(
                            f"Running {method_name} (CFL={cfl}, dt={dt}, n_steps={n_steps}, N={N})..."
                        )

                        rho_final, coarse_x, exec_time, dx = run_particle_sim(
                            SolverClass, RunnerClass, Nc, Kn, problem, cfl, N
                        )

                        truth_projected = np.interp(coarse_x, truth_x, truth_rho)

                        abs_error = np.sum(np.abs(rho_final - truth_projected)) * dx
                        truth_norm = np.sum(np.abs(truth_projected)) * dx
                        rel_error = abs_error / truth_norm

                        times.append(exec_time)
                        errors.append(rel_error)

                    ax.loglog(
                        times,
                        errors,
                        color=colors[method_name],
                        linestyle=line_styles[cfl_idx % len(line_styles)],
                        marker=markers[method_name],
                        label=f"{method_name} (CFL={cfl})",
                        markersize=4,
                        linewidth=1.2,
                    )

                    if method_name == "UGKP" and cfl == 0.9:
                        for i, N in enumerate(N_particles_list):
                            ax.annotate(
                                f"{N // 1000}k",
                                (times[i], errors[i]),
                                textcoords="offset points",
                                xytext=(0, 8),
                                ha="center",
                                fontsize=6,
                                alpha=0.7,
                            )

            # Formatting specific to this subplot
            ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

            # Left column gets Y-labels
            if col == 0:
                ax.set_ylabel(f"$N_c = {Nc}$\nRelative $L_1$ error")
            else:
                ax.set_yticklabels([])

            # Top row gets titles
            if row == 0:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$")

            # Bottom row gets X-labels
            if row == n_rows - 1:
                ax.set_xlabel("Execution time (s)")
            else:
                ax.set_xticklabels([])

            # Put a legend in the top plot of EVERY column.
            # This ensures that whatever methods are active in this column are labeled.
            if row == 0:
                ax.legend(fontsize="x-small", loc="best", framealpha=0.9)

    filename = (
        "latex/thesis/figures/stochastic/efficiency/work_precision_stochastic_grid.pdf"
    )

    if SAVE_PLOTS:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tp.save_plot(filename)
        print(f"\n✅ Saved dynamic stochastic grid plot to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
