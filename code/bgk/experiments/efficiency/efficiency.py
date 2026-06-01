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
    Nv = 100
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

    # Start the timer!
    start_time = time.perf_counter()
    sim = runner.run()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return sim, execution_time, dx


def main():
    problem = get_problem("gaussian")
    Kn = 1e6
    SAVE_PLOTS = False

    # ---------------------------------------------------------
    # CONFIGURATION BLOCK
    # ---------------------------------------------------------
    cfl_settings = {
        "Strang": [],
        "SL": [],
        "FVM": [0.9],
        "UGKS": [0.9],
        "Hybrid": [],
    }

    N_grids = [50, 100, 200, 400, 800]

    solver_classes = {
        "Strang": StrangSolver,
        "SL": SLSolver,
        "FVM": FVMSolver,
        "UGKS": UGKSSolver,
        "Hybrid": HybridSolver,
    }

    base_styles = {
        "Strang": {"color": "tab:blue", "marker": "o"},
        "SL": {"color": "tab:red", "marker": "^"},
        "FVM": {"color": "tab:green", "marker": "s"},
        "UGKS": {"color": "tab:orange", "marker": "D"},
        "Hybrid": {"color": "tab:purple", "marker": "v"},
    }
    line_styles = ["-", "--", ":", "-."]
    # ---------------------------------------------------------

    # 1. Generate a high-resolution "Truth" Solution using FVM
    N_res_ref = N_grids[-1] * 4
    cfl_ref = 0.9
    print(
        f"Generating High-Resolution Truth Solution (FVM, Nc={N_res_ref}, CFL={cfl_ref})"
    )

    truth_sim, _, _ = run_simulation(
        "FVM", FVMSolver, N_res_ref, Kn, problem, cfl=cfl_ref
    )
    truth_x = truth_sim.x.flatten()
    truth_rho = truth_sim.rho[-1].flatten()  # Final time step density profile

    results_time = {}
    results_error = {}

    # 2. Run Experiments dynamically based on cfl_settings
    for name, cfl_list in cfl_settings.items():
        if not cfl_list:
            continue

        SolverClass = solver_classes[name]

        for cfl_idx, cfl in enumerate(cfl_list):
            run_name = f"{name} (CFL={cfl})"
            print("\n========================================")
            print(f" Testing {run_name}")
            print("========================================")

            results_time[run_name] = []
            results_error[run_name] = []

            for N_res in N_grids:
                sim, exec_time, dx = run_simulation(
                    name, SolverClass, N_res, Kn, problem, cfl
                )

                # Interpolate the Truth solution onto this specific coarse grid
                coarse_x = sim.x.flatten()
                coarse_rho = sim.rho[-1].flatten()

                # linear interpolation
                truth_projected = np.interp(coarse_x, truth_x, truth_rho)

                # Calculate RELATIVE L1 Error based on True Integral
                if name in ["Strang", "SL", "Hybrid"]:
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

                print(
                    f"  N_res={N_res:3d} | Time: {exec_time:.3f} s | Rel L1 Error: {rel_error:.2e}"
                )

                results_time[run_name].append(exec_time)
                results_error[run_name].append(rel_error)

    # 3. Plot Work-Precision Diagram using thesis_plots
    width, _ = tp.get_figsize(fraction=1.0)  # Adjust fraction if it's too wide
    fig, ax = plt.subplots(figsize=(width, width * 0.75), constrained_layout=True)

    for name, cfl_list in cfl_settings.items():
        for cfl_idx, cfl in enumerate(cfl_list):
            run_name = f"{name} (CFL={cfl})"
            if run_name in results_time:
                color = base_styles[name]["color"]
                marker = base_styles[name]["marker"]
                ls = line_styles[cfl_idx % len(line_styles)]

                label = f"{name} (CFL={cfl})"
                ax.loglog(
                    results_time[run_name],
                    results_error[run_name],
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    label=label,
                    markersize=4,
                    linewidth=1.5,
                )

    ax.set_xlabel("Execution time (seconds)")
    ax.set_ylabel(r"Density relative $L_1$ error")
    ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

    # Place legend cleanly inside the plot (or adjust loc as needed)
    ax.legend(fontsize="small", loc="best")

    # Save as a single, perfectly scaled PDF
    filename = (
        f"latex/thesis/figures/ch4/efficiency/work_precision_CFL_comparison_Kn_{Kn}.pdf"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved combined plot using thesis_plots to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
