import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import bgk.io.io_hdf5 as io_hdf5
import bgk.thesis_plots as tp

RESULTS_DIR = "simulation_results"


def load_sim(method_name, N_res, Kn, cfl, Nv=40, t_final=0.2):
    res = io_hdf5.load_result(
        method_name, N_res, Nv, None, Kn, t_final, "gaussian", RESULTS_DIR, cfl
    )
    if res is None:
        raise FileNotFoundError(
            f"Missing simulation: {method_name}, N={N_res}, Kn={Kn}, CFL={cfl}"
        )
    return res


def main():
    SAVE_PLOTS = True

    # =========================================================
    # ⚙️ PLOTTING CONFIGURATION
    # =========================================================
    KNS_TO_PLOT = [1e-6, 1e-2, 1e2]  # Which Knudsen numbers to plot
    SUBPLOT_LAYOUT = (3, 1)  # (rows, columns)
    # =========================================================

    N_grids = [50, 100, 200, 400, 800, 3200]
    N_res_ref = N_grids[-1] * 4
    cfl_settings = {
        "Strang": [0.9],
        "SL": [0.9],
        "FVM": [0.9],
        "UGKS": [0.9],
        "Hybrid": [0.9],
    }

    base_styles = {
        "Strang": {"color": "tab:blue", "marker": "o"},
        "SL": {"color": "tab:red", "marker": "^"},
        "FVM": {"color": "tab:green", "marker": "s"},
        "UGKS": {"color": "tab:orange", "marker": "D"},
        "Hybrid": {"color": "tab:purple", "marker": "v"},
    }
    line_styles = ["-", "--", ":", "-."]

    # Initialize the figure based on the requested layout
    rows, cols = SUBPLOT_LAYOUT
    if rows * cols < len(KNS_TO_PLOT):
        raise ValueError(
            f"Layout {SUBPLOT_LAYOUT} only has {rows * cols} slots, but you asked to plot {len(KNS_TO_PLOT)} Kn values."
        )

    width, _ = tp.get_figsize(fraction=0.8)
    # Adjust height dynamically based on the number of rows
    fig, axs = plt.subplots(
        rows, cols, figsize=(width, width * 0.6 * rows), constrained_layout=True
    )

    # Flatten the axes array to easily iterate over it, even if it's 1D or 2D
    axs_flat = np.atleast_1d(axs).flatten()

    for idx, Kn in enumerate(KNS_TO_PLOT):
        print(f"\nProcessing Kn = {Kn}...")
        ax = axs_flat[idx]

        # Load truth for this specific Kn
        truth_res = load_sim("FVM", N_res_ref, Kn, cfl=0.9, Nv=40)
        truth_x, truth_rho = truth_res["x"], truth_res["rho"]

        results_time = {}
        results_error = {}

        for name, cfl_list in cfl_settings.items():
            if not cfl_list:
                continue

            for cfl_idx, cfl in enumerate(cfl_list):
                run_name = f"{name} (CFL={cfl})"
                results_time[run_name] = []
                results_error[run_name] = []

                for N_res in N_grids:
                    res = load_sim(name, N_res, Kn, cfl, Nv=40)
                    coarse_x, coarse_rho = res["x"], res["rho"]
                    exec_time, dx = res["execution_time"], res["dx"]

                    truth_projected = np.interp(coarse_x, truth_x, truth_rho)

                    if name in ["Strang", "SL", "Hybrid"]:
                        abs_error = (
                            np.sum(np.abs(coarse_rho[:-1] - truth_projected[:-1])) * dx
                        )
                        truth_norm = np.sum(np.abs(truth_projected[:-1])) * dx
                    else:
                        abs_error = np.sum(np.abs(coarse_rho - truth_projected)) * dx
                        truth_norm = np.sum(np.abs(truth_projected)) * dx

                    results_time[run_name].append(exec_time)
                    results_error[run_name].append(abs_error / truth_norm)

                # Plot onto the specific subplot
                if run_name in results_time:
                    color, marker = (
                        base_styles[name]["color"],
                        base_styles[name]["marker"],
                    )
                    ls = line_styles[cfl_idx % len(line_styles)]

                    ax.loglog(
                        results_time[run_name],
                        results_error[run_name],
                        color=color,
                        marker=marker,
                        linestyle=ls,
                        label=run_name,
                        markersize=4,
                        linewidth=1.5,
                    )

        # Formatting the subplot
        ax.set_xlabel("Execution time (seconds)")
        if idx % cols == 0:  # Only label the Y axis on the left-most plots
            ax.set_ylabel(r"Density relative $L_1$ error")

        ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

        kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn not in [1, 1.0] else "1"
        ax.set_title(rf"Kn = ${kn_formatted}$", fontsize=10)

        # Add legend only to the first subplot to avoid clutter
        if idx == 0:
            ax.legend(fontsize="small", loc="best")

    # Hide any unused subplots (if your layout is larger than your Kn list)
    for idx in range(len(KNS_TO_PLOT), len(axs_flat)):
        axs_flat[idx].set_visible(False)

    filename = "latex/thesis/figures/ch4/efficiency/work_precision_combined.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved combined plot using thesis_plots to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
