import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import bgk.thesis_plots as tp
from bgk.io.io_hdf5 import load_result


def main():
    SAVE_PLOTS = False
    RESULTS_DIR = (
        "code/bgk/experiments/convergence/sims"  # Make sure this matches run_sims.py
    )
    problem_name = "gaussian"

    selected_methods = ["Hybrid"]
    Kns = np.logspace(-8, 4, 6)

    # Fixed parameters used in simulations
    t_final = 0.20
    Nv = 40
    CFL = 0.9

    width, height = tp.get_figsize(fraction=1.0)
    row_height_multiplier = 0.48

    fig, axes = plt.subplots(
        nrows=len(selected_methods),
        ncols=2,
        figsize=(width, height * row_height_multiplier * len(selected_methods)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, solver_name in enumerate(selected_methods):
        print(f"Plotting {solver_name} Solver...")
        ax_err = axes[row_idx, 0]
        ax_ord = axes[row_idx, 1]

        is_cell_centered = solver_name in ["FVM", "UGKS"]
        grid_sym = "N_c" if is_cell_centered else "N_x"

        base_Nx = 50
        num_test_grids = 3
        test_grids = []

        if is_cell_centered:
            refinement_factor = 3
            test_grids = [
                base_Nx * (refinement_factor**i) for i in range(num_test_grids)
            ]
            N_ref = test_grids[-1] * 9
        else:
            refinement_factor = 2
            current_N = base_Nx
            for _ in range(num_test_grids):
                test_grids.append(current_N)
                current_N = current_N * refinement_factor - 1
            N_ref = (test_grids[-1] - 1) * 8 + 1

        N_grids = test_grids + [N_ref]

        errors = {Nx: [] for Nx in test_grids}
        order_pairs = [
            (test_grids[i], test_grids[i + 1]) for i in range(len(test_grids) - 1)
        ]
        orders = {pair: [] for pair in order_pairs}

        for Kn in Kns:
            rhos = {}
            # Load all grids for this Kn
            for N_val in N_grids:
                res = load_result(
                    solver=solver_name,
                    Nc=N_val,
                    Nvx=Nv,
                    Nvy=None,
                    Kn=Kn,
                    t_final=t_final,
                    problem=problem_name,
                    results_dir=RESULTS_DIR,
                    CFL=CFL,
                )

                if res is None:
                    raise FileNotFoundError(
                        f"Data missing for {solver_name}, N={N_val}, Kn={Kn}. "
                        "Did you run run_sims.py first?"
                    )
                rhos[N_val] = res["rho"]

            # Compute L1 Errors
            for N_c in test_grids:
                if is_cell_centered:
                    factor = N_ref // N_c
                    center_offset = factor // 2
                else:
                    factor = (N_ref - 1) // (N_c - 1)
                    center_offset = 0

                rho_coarse_1d = np.squeeze(rhos[N_c])
                rho_ref_1d = np.squeeze(rhos[N_ref])
                proj_ref = rho_ref_1d[center_offset::factor]

                abs_diff = np.sum(np.abs(rho_coarse_1d - proj_ref))
                norm_ref = np.sum(np.abs(proj_ref))
                errors[N_c].append(abs_diff / norm_ref)

            # Compute Order of Accuracy
            for N_c, N_f in order_pairs:
                err_coarse = errors[N_c][-1]
                err_fine = errors[N_f][-1]
                order_val = np.log(err_coarse / err_fine) / np.log(refinement_factor)
                orders[(N_c, N_f)].append(order_val)

        # Apply plotting aesthetics
        ax_err.set_title(f"{solver_name}", loc="left", fontweight="bold", fontsize=10)

        markers_err = itertools.cycle(["bo", "rs", "g^", "cd", "m^"])
        for N_val, marker in zip(test_grids, markers_err):
            ax_err.loglog(Kns, errors[N_val], marker, label=rf"${grid_sym}={N_val}$")
        ax_err.set_ylabel("rel. error")
        ax_err.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
        ax_err.legend(fontsize=9)

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

    axes[-1, 0].set_xlabel(r"Kn")
    axes[-1, 1].set_xlabel(r"Kn")

    if SAVE_PLOTS:
        filename_base = "latex/thesis/figures/ch4/order/"
        filename = (
            filename_base
            + f"knudsen_combined_analysis_{'_'.join(selected_methods).lower()}"
        )
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
