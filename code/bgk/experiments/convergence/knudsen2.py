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


def run_simulation(SolverClass, N_val, Kn, problem, is_cell_centered):
    """Helper to configure and run a simulation for a specific solver, Nx, and Kn."""
    CFL = 0.5
    vmax = 10.0
    t_final = 0.30
    Nv = 40
    R = 1.0

    # Calculate dx and dt based on N_val
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

    return sim.rho[-1], dx


def main():
    SAVE_PLOTS = False
    problem = get_problem("gaussian")

    # =========================================================================
    # CONFIGURATION: Choose which methods to plot by editing this list
    # =========================================================================
    selected_methods = [
        "Hybrid",
    ]

    # 1. Define the parameter space
    Kns = np.logspace(-8, 4, 6)  # 40 points from 10^-10 to 10^4

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

    print("Running simulations. This may take a while depending on the grid sizes...")

    # ---------------------------------------------------------
    # GLOBAL PLOT SETUP
    # ---------------------------------------------------------
    # Get base width, and scale the height dynamically per row
    width, height = tp.get_figsize(fraction=1.0)
    row_height_multiplier = 0.48  # Scales cleanly based on original design

    # squeeze=False ensures axes is ALWAYS a 2D array, even if len(solvers) == 1
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
        print(f" Testing {solver_name} Solver")
        print("========================================")

        # Assign the correct subplots for this solver
        ax_err = axes[row_idx, 0]
        ax_ord = axes[row_idx, 1]

        # ---------------------------------------------------------
        # Set grids based on solver type (Factor 6 Reference Grid)
        # ---------------------------------------------------------
        is_cell_centered = solver_name in ["FVM", "UGKS"]
        grid_sym = "N_c" if is_cell_centered else "N_x"

        base_Nx = 50
        num_test_grids = 3  # The coarse grids used for error/order calculation
        test_grids = []

        if is_cell_centered:
            # Cell centers subdivide volumes cleanly
            refinement_factor = 3
            test_grids = [
                base_Nx * (refinement_factor**i) for i in range(num_test_grids)
            ]

            # The reference grid has a factor 3 more cells than the last coarse grid.
            # Factor 3 (odd) ensures the middle fine cell aligns exactly with the
            # coarse cell center, so point-sampling introduces no shift error.
            ref_factor = 3
            N_ref = test_grids[-1] * ref_factor
        else:
            # Nodal points need the 2N - 1 logic to align intervals
            refinement_factor = 2
            current_N = base_Nx
            for _ in range(num_test_grids):
                test_grids.append(current_N)
                current_N = current_N * 2 - 1

            # The reference grid has a factor 4 more CELLS (intervals).
            # Number of cells = (Nx - 1). Therefore, Ref Nx = (cells * 4) + 1
            ref_factor = 4
            N_ref = (test_grids[-1] - 1) * ref_factor + 1

        N_grids = test_grids + [N_ref]

        print(f"Using test refinement factor: {refinement_factor}")
        print(
            f"Grid type: {'Cell Centers (Nc)' if is_cell_centered else 'Grid Points (Nx)'}"
        )
        print(f"Test grids: {test_grids}")
        print(f"Reference grid: {N_ref}")

        errors = {Nx: [] for Nx in test_grids}
        order_pairs = [
            (test_grids[i], test_grids[i + 1]) for i in range(len(test_grids) - 1)
        ]
        orders = {pair: [] for pair in order_pairs}

        # Iterate over Knudsen numbers
        iter_count = 0
        for Kn in Kns:
            iter_count += 1
            print(f"  [{iter_count}/{len(Kns)}] Knudsen number: {Kn:.2e}")
            rhos = {}
            dxs = {}

            # Run all grid resolutions for this Kn
            for N_val in N_grids:
                rho_final, dx = run_simulation(
                    SolverClass, N_val, Kn, problem, is_cell_centered
                )
                rhos[N_val] = rho_final
                dxs[N_val] = dx

            # Compute Relative L1 Errors against the finest reference grid
            for N_c in test_grids:
                if is_cell_centered:
                    # Cell centers scale purely by total cells N
                    factor = N_ref // N_c
                    center_offset = factor // 2
                else:
                    # Grid points scale strictly by intervals (N - 1)
                    factor = (N_ref - 1) // (N_c - 1)
                    center_offset = 0

                rho_coarse_1d = np.squeeze(rhos[N_c])
                rho_ref_1d = np.squeeze(rhos[N_ref])

                # Dynamic slicing handles the factor 6 perfectly
                proj_ref = rho_ref_1d[center_offset::factor]

                abs_diff = np.sum(np.abs(rho_coarse_1d - proj_ref))
                norm_ref = np.sum(np.abs(proj_ref))
                rel_err = abs_diff / norm_ref

                errors[N_c].append(rel_err)

            # Compute Order of Accuracy dynamically
            for N_c, N_f in order_pairs:
                err_coarse = errors[N_c][-1]
                err_fine = errors[N_f][-1]

                # The log base uses the test_grid refinement_factor (2 or 3), not 6!
                order_val = np.log(err_coarse / err_fine) / np.log(refinement_factor)
                orders[(N_c, N_f)].append(order_val)

        # ---------------------------------------------------------
        # ROW SUBTITLE (METHOD NAME)
        # ---------------------------------------------------------
        ax_err.set_title(f"{solver_name}", loc="left", fontweight="bold", fontsize=10)

        # ---------------------------------------------------------
        # PLOTTING ERROR FOR THIS SOLVER
        # ---------------------------------------------------------
        markers_err = itertools.cycle(["bo", "rs", "g^", "cd", "m^"])
        for N_val, marker in zip(test_grids, markers_err):
            ax_err.loglog(Kns, errors[N_val], marker, label=rf"${grid_sym}={N_val}$")

        ax_err.set_ylabel("rel. error")
        ax_err.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
        ax_err.legend(fontsize=9)

        # ---------------------------------------------------------
        # PLOTTING ORDER FOR THIS SOLVER
        # ---------------------------------------------------------
        markers_ord = itertools.cycle(["cd", "m+", "g*", "r^", "bd"])
        for pair, marker in zip(order_pairs, markers_ord):
            ax_ord.semilogx(
                Kns,
                orders[pair],
                marker,
                label=rf"${grid_sym}={pair[0]} \to {pair[1]}$",
            )

        ax_ord.set_ylabel("order")
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
        + f"knudsen_combined_analysis_{'_'.join(selected_methods).lower()}"
    )
    if SAVE_PLOTS:
        # Check if file already exists
        if os.path.exists(f"{filename}.pdf"):
            counter = 1
            base_filename = filename
            while os.path.exists(f"{filename}.pdf"):
                filename = f"{base_filename}_{counter}"
                counter += 1
        tp.save_plot(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
