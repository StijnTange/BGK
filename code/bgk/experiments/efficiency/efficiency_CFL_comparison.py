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


def run_simulation(SolverClass, Nx, Kn, problem, cfl):
    """Runs a simulation and returns the final simulation object and execution time."""
    vmax = 10.0
    t_final = 0.2
    Nv = 40
    R = 1.0

    dx = (problem.x_bounds[1] - problem.x_bounds[0]) / Nx
    dt = cfl * dx / vmax

    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=Nx,
        Nc=None,
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

    # Start the timer!
    print(f"Running {SolverClass.__name__} with Nx={Nx}, dt={dt:.2e} (CFL={cfl})...")
    start_time = time.perf_counter()
    sim = runner.run()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return sim, execution_time


def main():
    problem = get_problem("gaussian")
    Kn = 1e2  # Transition regime
    SAVE_PLOTS = True

    # ---------------------------------------------------------
    # CONFIGURATION BLOCK FOR SINGLE SOLVER
    # ---------------------------------------------------------
    TARGET_SOLVER_NAME = "Strang"  # Change this to "Strang", "FVM", or "UGKS"
    CFL_LIST = [
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ]  # Add more CFL values as needed
    N_grids = [
        100,
        200,
        400,
        800,
    ]  # Geometric progression for Nx (e.g., 100, 300, 900, ...)

    solver_classes = {
        "Strang": StrangSolver,
        "SL": SLSolver,
        "FVM": FVMSolver,
        "UGKS": UGKSSolver,
        "Hybrid": HybridSolver,
    }

    # Style lists to differentiate the CFLs
    # define colors using a gradient for better visual distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(CFL_LIST)))
    markers = ["o", "^", "s", "D", "v", "p"]
    line_styles = ["-", "--", ":", "-."]
    # ---------------------------------------------------------

    TargetSolverClass = solver_classes[TARGET_SOLVER_NAME]

    # 1. Generate a high-resolution "Truth" Solution using UGKS at CFL 0.9
    Nx_ref = N_grids[-1] * 8
    cfl_ref = 0.9
    print(
        f"Generating High-Resolution Truth Solution (UGKS, Nx={Nx_ref}, CFL={cfl_ref})"
    )

    truth_sim, _ = run_simulation(UGKSSolver, Nx_ref, Kn, problem, cfl=cfl_ref)
    truth_x = truth_sim.x.flatten()
    truth_rho = truth_sim.rho[-1].flatten()

    results_time = {}
    results_error = {}

    # 2. Run Experiments for the target solver across different CFLs
    for idx, cfl in enumerate(CFL_LIST):
        run_name = f"CFL = {cfl}"
        print("\n========================================")
        print(f" Testing {TARGET_SOLVER_NAME} at {run_name}")
        print("========================================")

        results_time[run_name] = []
        results_error[run_name] = []

        for Nx in N_grids:
            sim, exec_time = run_simulation(TargetSolverClass, Nx, Kn, problem, cfl)

            # Interpolate the Truth solution onto this specific coarse grid
            coarse_x = sim.x.flatten()
            coarse_rho = sim.rho[-1].flatten()
            truth_projected = np.interp(coarse_x, truth_x, truth_rho)

            # Calculate RELATIVE L1 Error
            abs_error = np.sum(np.abs(coarse_rho - truth_projected))
            truth_norm = np.sum(np.abs(truth_projected))
            rel_error = abs_error / truth_norm

            print(
                f"  Nx={Nx:3d} | Time: {exec_time:.3f} s | Rel L1 Error: {rel_error:.2e}"
            )

            results_time[run_name].append(exec_time)
            results_error[run_name].append(rel_error)

    # 3. Plot Work-Precision Diagram using thesis_plots
    width, _ = tp.get_figsize(fraction=0.6)  # Adjust fraction if it's too wide
    fig, ax = plt.subplots(figsize=(width, width * 0.75), constrained_layout=True)

    for idx, cfl in enumerate(CFL_LIST):
        run_name = f"CFL = {cfl}"

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ls = line_styles[idx % len(line_styles)]

        ax.loglog(
            results_time[run_name],
            results_error[run_name],
            color=color,
            marker=marker,
            linestyle=ls,
            label=run_name,
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xlabel("Execution time (seconds)")
    ax.set_ylabel(r"Density relative $L_1$ error")
    ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

    # Place legend cleanly inside the plot
    ax.legend()

    # Save as a single, perfectly scaled PDF dynamically named by the solver
    filename = f"latex/thesis/figures/ch4/efficiency/work_precision_{TARGET_SOLVER_NAME}_CFL.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved {TARGET_SOLVER_NAME} CFL comparison plot to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
