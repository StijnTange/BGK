import itertools
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


def run_simulation(SolverClass, N_val, CFL, Kn, problem, is_cell_centered):
    """Helper to configure and run a simulation for a specific solver, fixed Nx, varying CFL, and Kn."""
    vmax = 10.0
    t_final = 0.10
    Nv = 40
    R = 1.0

    # Calculate dx and dt based on the fixed N_val and the requested CFL
    dx = (problem.x_bounds[1] - problem.x_bounds[0]) / N_val
    dt = CFL * dx / vmax

    # Dynamically assign Nc (cell centers) or Nx (grid points)
    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=None if is_cell_centered else N_val,
        Nc=N_val if is_cell_centered else None,
        Nv=[Nv],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type="periodic",
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name, R=R)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    solver = SolverClass(config)
    runner = Runner(config=config, solver=solver, problem=problem)
    sim = runner.run()

    return sim.rho[-1], dt


def main():
    SAVE_PLOTS = False
    problem = get_problem("gaussian")

    # =========================================================================
    # CONFIGURATION: Choose methods and temporal parameters
    # =========================================================================
    selected_methods = [
        "Hybrid",
    ]

    # 1. Define the parameter space
    Kns = np.logspace(-8, 4, 10)  # 40 points from 10^-8 to 10^4

    # Define fixed spatial grid and temporal refinement
    fixed_N = 200  # High resolution fixed grid to minimize baseline spatial error
    test_CFLs = np.logspace(
        np.log10(10.0), np.log10(0.02), 5
    )  # 5 CFL values from 0.9 to 0.005
    ref_CFL = test_CFLs[-1] / 4
    solvers = {}
    for method in selected_methods:
        if method == "FVM":
            solvers[method] = FVMSolver
        elif method == "SL":
            solvers[method] = SLSolver
        elif method == "Strang":
            solvers[method] = StrangSolver
        elif method == "UGKS":
            solvers[method] = UGKSSolver
        elif method == "Hybrid":
            solvers[method] = HybridSolver
        else:
            raise ValueError(f"Unknown method: {method}")

    print("Running simulations to isolate temporal error. This may take a while...")

    # ---------------------------------------------------------
    # GLOBAL PLOT SETUP
    # ---------------------------------------------------------
    width, height = tp.get_figsize(fraction=1.0)
    row_height_multiplier = 0.48

    fig, axes = plt.subplots(
        nrows=len(solvers),
        ncols=2,
        figsize=(width, height * row_height_multiplier * len(solvers)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    # 2. Iterate over solvers
    for row_idx, (solver_name, SolverClass) in enumerate(solvers.items()):
        print("\n========================================")
        print(f" Testing {solver_name} Solver (Temporal Isolation)")
        print("========================================")

        ax_err = axes[row_idx, 0]
        ax_ord = axes[row_idx, 1]

        is_cell_centered = solver_name in ["FVM", "UGKS"]

        print(
            f"Fixed Spatial Grid: {fixed_N} {'Cell Centers (Nc)' if is_cell_centered else 'Grid Points (Nx)'}"
        )
        print(f"Test CFLs: {test_CFLs}")
        print(f"Reference CFL: {ref_CFL}")

        errors = {cfl: [] for cfl in test_CFLs}
        order_pairs = [
            (test_CFLs[i], test_CFLs[i + 1]) for i in range(len(test_CFLs) - 1)
        ]
        orders = {pair: [] for pair in order_pairs}

        # Store exact dt values for the legend once per solver
        dt_values = {}

        # Iterate over Knudsen numbers
        for iter_count, Kn in enumerate(Kns):
            print(f"  [{iter_count + 1}/{len(Kns)}] Knudsen number: {Kn:.2e}")
            rhos = {}

            # 1. Run the reference solution (very small time step)
            rho_ref, _ = run_simulation(
                SolverClass, fixed_N, ref_CFL, Kn, problem, is_cell_centered
            )
            rho_ref_1d = np.squeeze(rho_ref)

            # 2. Run the test solutions
            for CFL in test_CFLs:
                rho_test, dt = run_simulation(
                    SolverClass, fixed_N, CFL, Kn, problem, is_cell_centered
                )
                dt_values[CFL] = dt  # Save for legend

                rho_test_1d = np.squeeze(rho_test)

                # Compute Relative L1 Error.
                # Since the grids are identical, we subtract directly!
                abs_diff = np.sum(np.abs(rho_test_1d - rho_ref_1d))
                norm_ref = np.sum(np.abs(rho_ref_1d))
                rel_err = abs_diff / norm_ref

                errors[CFL].append(rel_err)

            # 3. Compute Order of Accuracy dynamically based on dt ratios
            for cfl_coarse, cfl_fine in order_pairs:
                err_coarse = errors[cfl_coarse][-1]
                err_fine = errors[cfl_fine][-1]

                dt_coarse = dt_values[cfl_coarse]
                dt_fine = dt_values[cfl_fine]

                # Calculate temporal order: log(E1/E2) / log(dt1/dt2)
                order_val = np.log(err_coarse / err_fine) / np.log(dt_coarse / dt_fine)
                orders[(cfl_coarse, cfl_fine)].append(order_val)

        # ---------------------------------------------------------
        # ROW SUBTITLE
        # ---------------------------------------------------------
        ax_err.set_title(
            f"{solver_name} (Fixed Spatial Grid)",
            loc="left",
            fontweight="bold",
            fontsize=10,
        )

        # ---------------------------------------------------------
        # PLOTTING ERROR FOR THIS SOLVER
        # ---------------------------------------------------------
        markers_err = itertools.cycle(["bo", "rs", "g^", "cd", "m^"])
        for CFL, marker in zip(test_CFLs, markers_err):
            dt_str = f"{dt_values[CFL]:.2e}"
            ax_err.loglog(
                Kns,
                errors[CFL],
                marker,
                label=rf"$\Delta t = {dt_str}$ (CFL={CFL:.2f})",
            )

        ax_err.set_ylabel("rel. error")
        ax_err.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
        ax_err.legend(fontsize=9)

        # ---------------------------------------------------------
        # PLOTTING ORDER FOR THIS SOLVER
        # ---------------------------------------------------------
        markers_ord = itertools.cycle(["cd", "m+", "g*", "r^", "bd"])
        for pair, marker in zip(order_pairs, markers_ord):
            dt_coarse_str = f"{dt_values[pair[0]]:.2e}"
            dt_fine_str = f"{dt_values[pair[1]]:.2e}"
            ax_ord.semilogx(
                Kns,
                orders[pair],
                marker,
                label=rf"$\Delta t: {dt_coarse_str} \to {dt_fine_str}$",
            )

        ax_ord.set_ylabel("temporal order")
        ax_ord.set_ylim([0, 4.5])
        ax_ord.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
        ax_ord.legend(fontsize=9)

    # Add x-labels only to the bottom row
    axes[-1, 0].set_xlabel(r"Kn")
    axes[-1, 1].set_xlabel(r"Kn")

    # ---------------------------------------------------------
    # SAVE OR SHOW THE SINGLE BIG PLOT
    # ---------------------------------------------------------
    filename_base = "latex/thesis/figures/ch4/order/"
    filename = (
        filename_base
        + f"knudsen_temporal_analysis_{'_'.join(selected_methods).lower()}"
    )
    if SAVE_PLOTS:
        tp.save_plot(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
