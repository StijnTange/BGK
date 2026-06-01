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
    CFL = 5.0
    vmax = 10.0
    t_final = 0.2
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
    SAVE_PLOTS = True

    problem = get_problem("gaussian")

    # 1. Define the parameter space
    Kns = np.logspace(-10, 4, 6)  # 6 points from 10^-10 to 10^4

    # Add the solvers you want to test to this list
    methods = ["Strang", "SL", "Hybrid"]  # Add "Hybrid" if implemented

    solvers = {}
    for method in methods:
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

    # 2. Iterate over solvers
    for row_idx, (solver_name, SolverClass) in enumerate(solvers.items()):
        print("\n========================================")
        print(f" Testing {solver_name} Solver")
        print("========================================")

        # ---------------------------------------------------------
        # Set factor and grids based on solver type
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # NEW LOGIC: Define factor and grids based on solver type
        # ---------------------------------------------------------
        is_cell_centered = solver_name in ["FVM", "UGKS"]

        base_Nx = 200
        num_grids = 2
        N_grids = []

        if is_cell_centered:
            # Cell centers subdivide volumes cleanly: 200 -> 600 -> 1800
            refinement_factor = 3
            N_grids = [base_Nx * (refinement_factor**i) for i in range(num_grids)]
        else:
            # Nodal points need the 2N - 1 logic to align: 200 -> 399 -> 797
            refinement_factor = 2  # The intervals are exactly halved
            current_N = base_Nx
            for _ in range(num_grids):
                N_grids.append(current_N)
                current_N = current_N * 2 - 1

        N_ref = N_grids[-1]
        test_grids = N_grids[:-1]

        print(f"Using refinement factor: {refinement_factor}")
        print(
            f"Grid type: {'Cell Centers (Nc)' if is_cell_centered else 'Grid Points (Nx)'}"
        )
        print(f"Running grids: {N_grids}")

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
                # Pass the is_cell_centered boolean to handle Nx/Nc properly
                rho_final, dx = run_simulation(
                    SolverClass, N_val, Kn, problem, is_cell_centered
                )
                rhos[N_val] = rho_final
                dxs[N_val] = dx

            # Compute Relative L1 Errors against the finest reference grid
            # Compute Relative L1 Errors against the finest reference grid
            for N_c in test_grids:
                # Calculate the slicing factor based on grid type
                if is_cell_centered:
                    # Cell centers scale purely by N
                    factor = N_ref // N_c
                    center_offset = factor // 2
                else:
                    # Grid points (nodes) scale by intervals (N - 1)
                    factor = (N_ref - 1) // (N_c - 1)
                    center_offset = 0

                # Squeeze the arrays to remove the dummy (1, Nx) dimension
                rho_coarse_1d = np.squeeze(rhos[N_c])
                rho_ref_1d = np.squeeze(rhos[N_ref])

                # Slice the 1D reference array using the dynamic offset
                proj_ref = rho_ref_1d[center_offset::factor]

                # Calculate the relative L1 error
                abs_diff = np.sum(np.abs(rho_coarse_1d - proj_ref))
                norm_ref = np.sum(np.abs(proj_ref))
                rel_err = abs_diff / norm_ref

                errors[N_c].append(rel_err)

            # Compute Order of Accuracy dynamically based on the solver's factor
            for N_c, N_f in order_pairs:
                err_coarse = errors[N_c][-1]
                err_fine = errors[N_f][-1]

                # Use the dynamic refinement factor for the log base (2 or 3)
                order_val = np.log(err_coarse / err_fine) / np.log(refinement_factor)
                orders[(N_c, N_f)].append(order_val)

        # ---------------------------------------------------------
        # PLOT 1: ERROR ANALYSIS
        # ---------------------------------------------------------
        # Use fraction=0.48 so they can sit side-by-side in LaTeX
        width, height = tp.get_figsize(fraction=0.48)
        fig, ax_err = plt.subplots(figsize=(width, height), constrained_layout=True)

        markers_err = itertools.cycle(["bo", "rs", "g^", "cd", "m^"])
        for Nx, marker in zip(test_grids, markers_err):
            ax_err.loglog(Kns, errors[Nx], marker, label=rf"$N_x={Nx}$")

        ax_err.set_xlabel(r"Kn")
        ax_err.set_ylabel(r"Density rel. $L_1$ error")
        ax_err.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
        ax_err.legend()

        filename_base = "latex/thesis/figures/ch4/order/"
        if SAVE_PLOTS:
            tp.save_plot(filename_base + f"knudsen_error_{solver_name.lower()}")

        # ---------------------------------------------------------
        # PLOT 2: ORDER OF ACCURACY
        # ---------------------------------------------------------
        fig, ax_ord = plt.subplots(figsize=(width, height), constrained_layout=True)

        markers_ord = itertools.cycle(["cd", "m+", "g*", "c^", "md"])
        for pair, marker in zip(order_pairs, markers_ord):
            ax_ord.semilogx(
                Kns, orders[pair], marker, label=rf"$N_x={pair[0]} \to {pair[1]}$"
            )

        ax_ord.set_xlabel(r"Kn")
        ax_ord.set_ylabel(r"Density order")
        ax_ord.set_ylim([0, 4.5])
        ax_ord.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
        ax_ord.legend()

        if SAVE_PLOTS:
            tp.save_plot(filename_base + f"knudsen_order_{solver_name.lower()}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
