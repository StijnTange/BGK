import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.io.io_hdf5 import load_result

RESULTS_DIR = "simulation_results"


def calculate_l1_error(rho_coarse, x_coarse, rho_ref, x_ref):
    """Computes the discrete L1 relative error by interpolating the reference."""
    rho_ref_interp = np.interp(x_coarse, x_ref, rho_ref)
    l1_diff = np.sum(np.abs(rho_coarse - rho_ref_interp))
    l1_ref = np.sum(np.abs(rho_ref_interp))
    return l1_diff / l1_ref


def plot_panel(
    ax,
    solvers_to_plot,
    x_ref,
    rho_ref,
    Kn,
    t_final,
    problem,
    NP_LIST,
    solver_colors,
    cfl_index_map,
):
    LINE_STYLES = ["-", "--", ":", "-."]
    MARKERS = ["o", "^", "s", "D", "v"]

    L_domain = x_ref[-1] - x_ref[0]
    vmax = 20.0

    for solver, solver_conf in solvers_to_plot.items():
        base_color = solver_colors.get(solver, "black")
        Nc = solver_conf["Nc"]
        dx_coarse = L_domain / Nc

        for cfl in solver_conf["cfls"]:
            global_idx = cfl_index_map[cfl]
            ls = LINE_STYLES[global_idx % len(LINE_STYLES)]
            marker = MARKERS[global_idx % len(MARKERS)]

            times = []
            errors = []

            for Np in NP_LIST:
                save_name = f"{solver}_Np{int(Np)}"
                data = load_result(
                    save_name, Nc, 100, None, Kn, t_final, problem, RESULTS_DIR, cfl
                )
                if data is not None:
                    err = calculate_l1_error(data["rho"], data["x"], rho_ref, x_ref)
                    times.append(data["execution_time"])
                    errors.append(err)
                else:
                    print(f"  -> Missing: {save_name}, Nc={Nc}, CFL={cfl}")

            if times:
                dt_val = cfl * dx_coarse / vmax
                label = f"{solver.upper()} (CFL={cfl if cfl < 1 else int(cfl)})"
                ax.loglog(
                    times,
                    errors,
                    color=base_color,
                    linestyle=ls,
                    marker=marker,
                    label=label,
                    markersize=4,
                    linewidth=1.2,
                )

    ax.set_xlabel("Execution Time (s)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    ax.legend()


def main():
    SAVE_PLOTS = True
    # =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    problem = "riemann"
    Kn = 1e-8
    t_final = 0.07
    # "matlab" uses the .mat Euler reference (only valid for riemann);
    # "deterministic" loads a pre-computed UGKS reference via load_result
    REFERENCE = "deterministic"

    NP_LIST = np.logspace(5, 6, 5)

    SOLVER_COLORS = {"rtsm": "tab:blue", "ugkp": "tab:red"}

    # Define one config dict per panel; each panel can have its own Nc per solver
    PANELS = [
        {
            "title": "$N_c = 100$",
            "solvers": {
                "rtsm": {"cfls": [0.1, 0.9, 2.0], "Nc": 100},
                "ugkp": {"cfls": [0.9, 2.0], "Nc": 100},
            },
        },
        {
            "title": "$N_c = 1000$",
            "solvers": {
                "rtsm": {"cfls": [0.9, 2.0, 5.0, 20.0, 40.0], "Nc": 1000},
                "ugkp": {"cfls": [0.9, 2.0], "Nc": 1000},
            },
        },
    ]

    # =========================================================================
    # 2. LOAD REFERENCE
    # =========================================================================
    if REFERENCE == "matlab":
        mat_path = "code/bgk/reference_solutions/riemann.mat"
        mat_data = sio.loadmat(mat_path)
        x_ref = mat_data["x_refined"].flatten()
        rho_ref = mat_data["rho_xx"].flatten()
    else:
        ref_data = load_result(
            "UGKS_REF", 4000, 100, None, Kn, t_final, problem, RESULTS_DIR, 0.9
        )
        if ref_data is None:
            print("Reference data not found! Please run the generation script first.")
            return
        x_ref = ref_data["x"]
        rho_ref = ref_data["rho"]

    # =========================================================================
    # 3. PLOT
    # =========================================================================
    width, _ = tp.get_figsize(fraction=1.0)
    _, axs = plt.subplots(
        1, len(PANELS), figsize=(width, width * 0.5), constrained_layout=True
    )

    if len(PANELS) == 1:
        axs = [axs]

    all_cfls = sorted(
        {
            cfl
            for panel in PANELS
            for solver_conf in panel["solvers"].values()
            for cfl in solver_conf["cfls"]
        }
    )
    cfl_index_map = {cfl: i for i, cfl in enumerate(all_cfls)}

    for ax, panel in zip(axs, PANELS):
        plot_panel(
            ax,
            panel["solvers"],
            x_ref,
            rho_ref,
            Kn,
            t_final,
            problem,
            NP_LIST,
            SOLVER_COLORS,
            cfl_index_map,
        )
        ax.set_title(panel["title"])

    axs[0].set_ylabel(r"Relative $L_1$ Error in $\rho$")

    filename = (
        f"latex/thesis/figures/ch4/efficiency/work_precision_nonlinear_{problem}.pdf"
    )
    if SAVE_PLOTS:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        tp.save_plot(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
