import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Ensure BGK framework is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import bgk.thesis_plots as tp  # Your custom styling script
from bgk.io.io_hdf5 import load_result


def calculate_l1_error(rho_coarse, x_coarse, rho_ref, x_ref):
    """
    Interpolates the high-res reference density onto the coarse grid
    and computes the discrete L1 relative error.
    """
    rho_ref_interp = np.interp(x_coarse, x_ref, rho_ref)
    l1_diff = np.sum(np.abs(rho_coarse - rho_ref_interp))
    l1_ref = np.sum(np.abs(rho_ref_interp))
    return l1_diff / l1_ref


def main():
    # =========================================================================
    # 1. PLOT CONFIGURATION PANEL
    # =========================================================================
    problem = "infiltration"
    Kn = 0.5
    t_final = 10.0
    Nc_coarse = 200
    # Nvx = 100

    # --- Profile Plot Settings (Left) ---
    PROFILE_CFL = 400  # The CFL you want to inspect visually
    PROFILE_N_INJ = 1e5  # The particle count to use for the profile

    # --- Work Precision Settings (Right) ---
    N_INJ_LIST = [5e4, 1e5, 5e5]
    SOLVERS_TO_PLOT = {
        "rtsm": [20, 50, 100, 200, 400],
        "vj": [20, 50, 100, 200, 400],
        "ugkp": [],
    }

    # Styling definitions
    SOLVER_COLORS = {"rtsm": "#377eb8", "vj": "#e41a1c", "ugkp": "#4daf4a"}
    LINE_STYLES = ["-", "--", ":", "-."]
    MARKERS = ["o", "s", "^", "D", "v"]

    def lighten(color, amount):
        """Blend color towards white by `amount` (0=original, 1=white)."""
        r, g, b = mcolors.to_rgb(color)
        return (
            1 - (1 - r) * (1 - amount),
            1 - (1 - g) * (1 - amount),
            1 - (1 - b) * (1 - amount),
        )

    # =========================================================================
    # 2. LOAD REFERENCE SOLUTION
    # =========================================================================
    ref_data = load_result(
        solver="UGKS_REF",
        Nc=2000,
        Nvx=80,
        Nvy=None,
        Kn=Kn,
        t_final=t_final,
        problem=problem,
        CFL=0.9,
    )
    if ref_data is None:
        print("Reference data not found! Please run the generation script first.")
        return

    x_ref = ref_data["x"]
    rho_ref = ref_data["rho"]

    # =========================================================================
    # 3. SETUP SUBPLOTS
    # =========================================================================
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(
        1, 2, figsize=(width, width * 0.45), constrained_layout=True
    )
    ax_prof, ax_wp = axs[0], axs[1]

    # =========================================================================
    # 4. LEFT PLOT: DENSITY PROFILES
    # =========================================================================
    ax_prof.plot(x_ref, rho_ref, "k--", label="Reference", linewidth=1.5)

    for solver in ["rtsm", "vj"]:
        save_name = f"{solver}_N{int(PROFILE_N_INJ)}"
        data = load_result(
            solver=save_name,
            Nc=Nc_coarse,
            Nvx=40,
            Nvy=None,
            Kn=Kn,
            t_final=t_final,
            problem=problem,
            CFL=PROFILE_CFL,
        )
        if data is not None:
            color = SOLVER_COLORS.get(solver, "black")
            marker = "o" if solver == "rtsm" else "s"
            # Format dt nicely in scientific notation for the legend
            label = f"{solver.upper()} (CFL={PROFILE_CFL})"

            # Use markevery=10 so the markers don't clutter the 200-cell grid
            ax_prof.plot(
                data["x"],
                data["rho"],
                color=color,
                marker=marker,
                markersize=3,
                markevery=10,
                linewidth=1.0,
                label=label,
            )
        else:
            print(f"  -> Profile data missing for {solver} at CFL {PROFILE_CFL}")

    ax_prof.set_xlabel("Position $x$")
    ax_prof.set_ylabel(r"Density $\rho$")
    ax_prof.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_prof.legend(loc="best")
    # You can zoom in on the boundary if you want by uncommenting the next line
    # ax_prof.set_xlim(0.0, 0.4)

    # =========================================================================
    # 5. RIGHT PLOT: WORK PRECISION
    # =========================================================================
    all_times = []
    all_errors = []

    for solver, cfl_list in SOLVERS_TO_PLOT.items():
        n_cfls = len(cfl_list)
        for idx, cfl in enumerate(cfl_list):
            ls = LINE_STYLES[idx % len(LINE_STYLES)]
            marker = MARKERS[idx % len(MARKERS)]
            fade = idx / max(n_cfls - 1, 1) * 0.6
            base_color = lighten(SOLVER_COLORS.get(solver, "black"), fade)

            times = []
            errors = []

            for N in N_INJ_LIST:
                save_name = f"{solver}_N{int(N)}"
                data = load_result(
                    solver=save_name,
                    Nc=Nc_coarse,
                    Nvx=40,
                    Nvy=None,
                    Kn=Kn,
                    t_final=t_final,
                    problem=problem,
                    CFL=cfl,
                )

                if data is not None:
                    rho_coarse = data["rho"]
                    x_coarse = data["x"]
                    err = calculate_l1_error(rho_coarse, x_coarse, rho_ref, x_ref)
                    times.append(data["execution_time"])
                    errors.append(err)
                else:
                    print(f"  -> WP data missing: {save_name} at CFL {cfl}")

            if times:
                label_name = f"{solver.upper()} (CFL={cfl})"
                ax_wp.loglog(
                    times,
                    errors,
                    color=base_color,
                    linestyle=ls,
                    marker=marker,
                    label=label_name,
                    markersize=3,
                    linewidth=1.0,
                )
                all_times.extend(times)
                all_errors.extend(errors)

    ax_wp.set_xlabel("Execution Time (s)")
    ax_wp.set_ylabel(r"Relative $L_1$ Error in $\rho$")
    ax_wp.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_wp.legend(loc="best")

    # =========================================================================
    # 6. FORMATTING & EXPORT
    # =========================================================================
    filename = (
        "latex/thesis/figures/infiltration/work_precision_infiltration_combined.pdf"
    )
    tp.save_plot(filename)
    print(f"\n✅ Saved Combined Plot to {filename}")


if __name__ == "__main__":
    main()
