"""
Fourier Flow — Plot Script.

Loads saved HDF5 results and produces dynamic figures based on your toggles.
Provides options for dimensional vs non-dimensional scaling, specific
macros, zoom insets, and summary heat flux plots.
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.experiments.fourier.analytics_fourier import (
    fourier_heat_flux_continuum,
    fourier_heat_flux_free_molecular,
    fourier_temperature_continuum,
    fourier_temperature_free_molecular,
)
from bgk.io.io_hdf5 import load_result

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# Format: (solver_name, Nc, Kn, t_final, CFL, nvx_override, nvy_override)
# NOTE: Particle solvers are saved with nvx=1, nvy=None
#       Deterministic solvers usually have nvx=28.
PLOT_RUNS = [
    # Particles
    ("ugkp", 10, 0.01, 50.0, 0.9, 1, None, 100_000),
    ("rtsm", 10, 0.01, 50.0, 0.9, 1, None, 100_000),
    # Deterministic (Uncomment to compare!)
    # ("fvm", 20, 0.01, 100.0, 0.9, 28, None),
]

# Provide multiple references for different Kn numbers here (or [None])
# Format: (solver_name, Nc, Kn, t_final, CFL, nvx_override, nvy_override)
REFERENCES = [("fvm", 50, 0.01, 50.0, 0.9, 28, None)]

RESULTS_DIR = "code/bgk/experiments/fourier/simulations"
OUTPUT_DIR = "latex/thesis/figures/ch4/fourier"
SAVE_PLOTS = False  # True → save PDFs, False → show interactively

# ── Feature Toggles ───────────────────────────────────────────────────────────
PLOT_DIMENSIONAL = True
PLOT_LOG_TEMPERATURE = False
PLOT_HEATFLUX_KN_ONLY = False

# Toggle to plot a unique row for each solver!
PLOT_PER_SOLVER_ROWS = False

# Any subset/ordering of: "temperature", "temperature_deviation", "density", "heat_flux", "velocity"
PLOT_MACROS = ["temperature", "density", "heat_flux"]

# Panel grid (Only used if PLOT_PER_SOLVER_ROWS = False)
LAYOUT = "2x2"

# Which panel should display the legend?
LEGEND_MACRO = "empty"

# Optional y-axis limits for each plot: (ymin, ymax) or None
Y_LIMITS = {
    "temperature": None,
    "temperature_deviation": None,
    "density": None,
    "heat_flux": None,
    "velocity": None,
}

# ── Analytical limit toggles ─────────────────────────────────────────────────
PLOT_CONT_BGK = True  # continuum heat flux with Pr=1 (BGK limit)
PLOT_CONT_PHYS = True  # continuum heat flux with Pr=2/3 (physical Argon)

# Select explicitly WHICH Knudsen numbers get a Free Molecular limit line.
# Set to True for all, False for none, or a list like [1000, 10]
PLOT_FM = True

PLOT_HEATFLUX_KN = False  # Produce the global summary plot at the end

# ── Zoom inset ────────────────────────────────────────────────────────────────
ZOOM_MACRO = None
ZOOM_REGION = (0.3, 0.7, 250, 300)
ZOOM_INSET = (0.46, 0.08, 0.35, 0.35)
ZOOM_CONNECTORS = True

# ═════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ═════════════════════════════════════════════════════════════════════════════

SOLVER_COLORS = {
    "strang": "tab:blue",
    "sl": "tab:red",
    "fvm": "tab:green",
    "ugks": "tab:orange",
    "hybrid": "tab:purple",
    "rtsm": "tab:brown",  # Added particle solvers
    "ugkp": "tab:cyan",
}

_MARKERS = ["s", "o", "^", "D", "v", "P", "X", "*"]
_LINES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def run_style(solver_name, variant_idx):
    color = SOLVER_COLORS.get(solver_name.lower(), "k")
    marker = _MARKERS[variant_idx % len(_MARKERS)]
    ls = _LINES[variant_idx % len(_LINES)]
    return color, marker, ls


def save_or_show(filename):
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        base, ext = os.path.splitext(path)
        out, n = path, 1
        while os.path.exists(out):
            out = f"{base}_{n}{ext}"
            n += 1
        tp.save_plot(out)
    else:
        plt.show()


def resolve_layout(n_panels, layout_str):
    explicit = {
        "1x1": (1, 1),
        "1x2": (1, 2),
        "1x3": (1, 3),
        "1x4": (1, 4),
        "2x2": (2, 2),
    }
    if layout_str in explicit:
        nrows, ncols = explicit[layout_str]
        if nrows * ncols >= n_panels:
            return nrows, ncols
    if n_panels == 1:
        return 1, 1
    elif n_panels == 2:
        return 1, 2
    elif n_panels == 3:
        return 1, 3
    else:
        return 2, 2


def make_axes(nrows, ncols, n_panels_to_keep=None):
    total_width, base_height = tp.get_figsize(fraction=1.0, ratio=0.8)
    aspect_ratio = base_height / total_width

    panel_width = total_width / ncols
    panel_height = panel_width * aspect_ratio
    total_height = panel_height * nrows * 1.15

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(total_width, total_height),
        constrained_layout=True,
        squeeze=False,
    )

    if n_panels_to_keep is not None:
        ax_flat = axes.flatten()
        for i in range(n_panels_to_keep, nrows * ncols):
            ax_flat[i].set_visible(False)

    return fig, axes


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
data = {}
for run in PLOT_RUNS:
    solver, Nc, Kn, t_final, CFL = run[:5]
    nvx = run[5] if len(run) > 5 else 28
    nvy = run[6] if len(run) > 6 else None

    r = load_result(solver, Nc, nvx, nvy, Kn, t_final, "fourier", RESULTS_DIR, CFL)
    if r is not None:
        data[(solver, Nc, Kn, CFL)] = r

ref_data_dict = {}
if REFERENCES is not None:
    for ref in REFERENCES:
        if ref is None:
            continue
        solver, Nc, Kn, t_final, CFL = ref[:5]
        nvx = ref[5] if len(ref) > 5 else 28
        nvy = ref[6] if len(ref) > 6 else None

        r_ref = load_result(
            solver, Nc, nvx, nvy, Kn, t_final, "fourier", RESULTS_DIR, CFL
        )
        if r_ref is not None:
            ref_data_dict[Kn] = {"data": r_ref, "tuple": ref}

if not data:
    print("No results found. Run run_fourier_particle.py first.")
    sys.exit(0)

_r0 = next(iter(data.values()))
R_nd = float(_r0["R_nd"])
omega = float(_r0["omega"])
T_low_nd = float(_r0["T_L_nd"])
T_upp_nd = float(_r0["T_R_nd"])
T_ref = float(_r0["T_ref"])
U0 = float(_r0["U0"])
Pr_Ar = float(_r0["Pr_Ar"])
dim_v = int(_r0.get("dim_v", 1))
n_v = dim_v

T_low = float(_r0.get("T_L", T_low_nd * T_ref))
T_upp = float(_r0.get("T_R", T_upp_nd * T_ref))

tau_mode = _r0.get("tau_mode", b"viscosity")
if isinstance(tau_mode, bytes):
    tau_mode = tau_mode.decode()
Pr_sim = 1.0 if tau_mode == "viscosity" else Pr_Ar

x_ref_nd = np.linspace(0.0, 1.0, 300)
print(
    f"  tau_mode={tau_mode}  => simulation converges to Pr={Pr_sim:.4f} continuum limit"
)


def _rho_from_Kn(Kn):
    mu_0_ = float(_r0["mu_0"])
    T_ref_ = float(_r0["T_ref"])
    Pr_Ar_ = float(_r0["Pr_Ar"])
    R_s_ = 208.13
    factor = np.sqrt(np.pi / (2.0 * R_s_ * T_ref_))
    if tau_mode == "viscosity":
        return mu_0_ * factor / Kn
    else:
        return (mu_0_ / Pr_Ar_) * factor / Kn


Kn_list = sorted({Kn for (_, _, Kn, _) in data})


def get_linear_baseline(x_array):
    if PLOT_DIMENSIONAL:
        return T_low + (T_upp - T_low) * x_array
    else:
        return T_low_nd + (T_upp_nd - T_low_nd) * x_array


# ═════════════════════════════════════════════════════════════════════════════
# PANEL HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def add_continuum(ax, macro, Kn, rho_0, qs, label_it=True):
    mu_0_nd = Kn * np.sqrt(2.0 / np.pi)
    lbl = "Continuum (eq.~B.45)" if label_it else None

    if macro == "temperature":
        T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)
        y = T_cont * T_ref if PLOT_DIMENSIONAL else T_cont
        ax.plot(x_ref_nd, y, "k-", lw=1.2, label=lbl, zorder=0)
    elif macro == "temperature_deviation":
        T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)
        y = T_cont * T_ref if PLOT_DIMENSIONAL else T_cont
        y = y - get_linear_baseline(x_ref_nd)
        ax.plot(x_ref_nd, y, "k-", lw=1.2, label=lbl, zorder=0)
    elif macro == "density":
        T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)
        rho_cont_nd = 1.0 / T_cont
        y = rho_cont_nd * rho_0 if PLOT_DIMENSIONAL else rho_cont_nd
        ax.plot(
            x_ref_nd,
            y,
            "k-",
            lw=1.2,
            label=r"Continuum ($\rho \propto 1/T$)" if label_it else None,
            zorder=0,
        )
    elif macro == "velocity":
        ax.axhline(0.0, color="k", lw=1.2, ls="-", label=lbl, zorder=0)
    elif macro == "heat_flux":
        if PLOT_CONT_BGK:
            q_bgk = fourier_heat_flux_continuum(
                T_low_nd, T_upp_nd, omega, mu_0_nd, R_nd, n_v, Pr=1.0
            )
            y = np.abs(q_bgk) * qs if PLOT_DIMENSIONAL else np.abs(q_bgk)
            lbl_bgk = (
                r"Continuum BGK ($\mathrm{Pr}=1$)"
                + (" $\leftarrow$ sim" if Pr_sim == 1.0 else "")
                if label_it
                else None
            )
            ax.axhline(y, color="k", lw=1.2, ls="-", label=lbl_bgk, zorder=0)
        if PLOT_CONT_PHYS:
            q_phys = fourier_heat_flux_continuum(
                T_low_nd, T_upp_nd, omega, mu_0_nd, R_nd, n_v, Pr=Pr_Ar
            )
            y = np.abs(q_phys) * qs if PLOT_DIMENSIONAL else np.abs(q_phys)
            lbl_phys = (
                rf"Continuum Ar ($\mathrm{{Pr}}={Pr_Ar:.2f}$)"
                + (" $\leftarrow$ sim" if Pr_sim == Pr_Ar else "")
                if label_it
                else None
            )
            ax.axhline(y, color="k", lw=1.2, ls="-", label=lbl_phys, zorder=0)


def add_free_molecular(ax, macro, rho_0, qs, label_it=True):
    lbl = rf"Free mol. ($n_v={n_v}$, eq.~B.59)" if label_it else None

    if macro == "temperature":
        T_fm = fourier_temperature_free_molecular(T_low_nd, T_upp_nd)
        y = T_fm * T_ref if PLOT_DIMENSIONAL else T_fm
        ax.axhline(y, color="k", lw=1.0, ls="--", label=lbl, zorder=0)
    elif macro == "temperature_deviation":
        T_fm = fourier_temperature_free_molecular(T_low_nd, T_upp_nd)
        y_scalar = T_fm * T_ref if PLOT_DIMENSIONAL else T_fm
        y_arr = np.full_like(x_ref_nd, y_scalar)
        y_dev = y_arr - get_linear_baseline(x_ref_nd)
        ax.plot(x_ref_nd, y_dev, "k--", lw=1.0, label=lbl, zorder=0)
    elif macro == "density":
        y = rho_0 if PLOT_DIMENSIONAL else 1.0
        ax.axhline(y, color="k", lw=1.0, ls="--", label=lbl, zorder=0)
    elif macro == "velocity":
        ax.axhline(0.0, color="k", lw=1.0, ls="--", label=lbl, zorder=0)
    elif macro == "heat_flux":
        rho_L_nd = 2.0 / (1.0 + np.sqrt(T_low_nd / T_upp_nd))
        q_fm = fourier_heat_flux_free_molecular(rho_L_nd, T_low_nd, T_upp_nd, R_nd, n_v)
        y = np.abs(q_fm) * qs if PLOT_DIMENSIONAL else np.abs(q_fm)
        ax.axhline(y, color="k", lw=1.0, ls="--", label=lbl, zorder=0)


def add_reference(ax, macro, r, rho_0, qs, Kn, ref_tuple):
    solver_name = r.get("solver", ref_tuple[0])
    if isinstance(solver_name, bytes):
        solver_name = solver_name.decode()
    Nc = r.get("Nc", ref_tuple[1])
    n_label = "N_c" if solver_name.lower() in ["ugks", "fvm"] else "N_x"

    kw = dict(
        color="dimgrey",
        marker="o",
        ls="none",
        fillstyle="none",
        ms=2,
        markevery=5,
        zorder=1,
        label=rf"Ref.: {solver_name.upper()} ${n_label}$={Nc} Kn={Kn}",
    )

    if macro == "temperature" and "T" in r:
        y = r["T"] * T_ref if PLOT_DIMENSIONAL else r["T"]
        ax.plot(r["x"], y, **kw)
    elif macro == "temperature_deviation" and "T" in r:
        y = r["T"] * T_ref if PLOT_DIMENSIONAL else r["T"]
        y = y - get_linear_baseline(r["x"])
        ax.plot(r["x"], y, **kw)
    elif macro == "density" and "rho" in r:
        y = r["rho"] * rho_0 if PLOT_DIMENSIONAL else r["rho"]
        ax.plot(r["x"], y, **kw)
    elif macro == "velocity" and "ux" in r:
        y = r["ux"] * U0 if PLOT_DIMENSIONAL else r["ux"]
        ax.plot(r["x"], y, **kw)
    elif macro == "heat_flux" and "qx" in r:
        y = np.abs(r["qx"]) * qs if PLOT_DIMENSIONAL else np.abs(r["qx"])
        ax.semilogy(r["x"], y, **kw)


def add_data(ax, macro, r, col, mk, ls, me, lbl, rho_0, qs):
    kw = dict(color=col, marker=mk, ms=2, lw=0.8, markevery=me, zorder=3, label=lbl)

    if macro == "temperature" and "T" in r:
        y = r["T"] * T_ref if PLOT_DIMENSIONAL else r["T"]
        ax.plot(r["x"], y, ls, **kw)
    elif macro == "temperature_deviation" and "T" in r:
        y = r["T"] * T_ref if PLOT_DIMENSIONAL else r["T"]
        y = y - get_linear_baseline(r["x"])
        ax.plot(r["x"], y, ls, **kw)
    elif macro == "density" and "rho" in r:
        y = r["rho"] * rho_0 if PLOT_DIMENSIONAL else r["rho"]
        ax.plot(r["x"], y, ls, **kw)
    elif macro == "velocity" and "ux" in r:
        y = r["ux"] * U0 if PLOT_DIMENSIONAL else r["ux"]
        ax.plot(r["x"], y, ls, **kw)
    elif macro == "heat_flux" and "qx" in r:
        y = np.abs(r["qx"]) * qs if PLOT_DIMENSIONAL else np.abs(r["qx"])
        ax.semilogy(r["x"], y, ls, **kw)


def add_zoom_inset(ax, macro):
    if ZOOM_MACRO is None or macro != ZOOM_MACRO:
        return
    x0, x1, y0, y1 = ZOOM_REGION
    il, ib, iw, ih = ZOOM_INSET
    axins = ax.inset_axes([il, ib, iw, ih])
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)

    if macro == "temperature" and getattr(
        sys.modules[__name__], "PLOT_LOG_TEMPERATURE", False
    ):
        axins.set_yscale("log")

    for line in ax.get_lines():
        axins.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            lw=line.get_linewidth(),
            ls=line.get_linestyle(),
            marker=line.get_marker(),
            ms=line.get_markersize(),
            markevery=line.get_markevery(),
            zorder=line.get_zorder(),
        )
    axins.tick_params(labelsize=10)
    axins.grid(alpha=0.3, linestyle="--", linewidth=0.4)
    ax.indicate_inset_zoom(axins, edgecolor="black", lw=0.8, alpha=0.7)


def finish_panel(ax, macro):
    ax.set_xlabel(r"$x / L$")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    if macro in Y_LIMITS and Y_LIMITS[macro] is not None:
        ax.set_ylim(Y_LIMITS[macro])

    add_zoom_inset(ax, macro)
    if macro == LEGEND_MACRO:
        ax.legend(loc="best")

    if macro == "temperature":
        ax.set_ylabel(r"$T$ [K]" if PLOT_DIMENSIONAL else r"$T / T_{ref}$")
        if getattr(sys.modules[__name__], "PLOT_LOG_TEMPERATURE", False):
            ax.set_yscale("log")
    elif macro == "temperature_deviation":
        ax.set_ylabel(
            r"$\Delta T$ from linear [K]"
            if PLOT_DIMENSIONAL
            else r"$\Delta T$ (non-dim.)"
        )
        ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)
    elif macro == "density":
        ax.set_ylabel(
            r"$\rho$ [kg/m$^3$]" if PLOT_DIMENSIONAL else r"$\rho / \rho_{ref}$"
        )
    elif macro == "velocity":
        ax.set_ylabel(r"$U_x$ [m/s]" if PLOT_DIMENSIONAL else r"$U_x / U_0$")
    elif macro == "heat_flux":
        ax.set_ylabel(
            r"$|q_x|$ [W/m$^2$]" if PLOT_DIMENSIONAL else r"$|q_x|$ (non-dim.)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FIGURES — COMBINED PROFILES
# ═════════════════════════════════════════════════════════════════════════════

if PLOT_HEATFLUX_KN_ONLY:
    PLOT_MACROS = []
    PLOT_HEATFLUX_KN = True

VALID = ("temperature", "temperature_deviation", "density", "heat_flux", "velocity")
macros = [m for m in PLOT_MACROS if m in VALID]

if not macros and not PLOT_HEATFLUX_KN:
    print("Nothing to plot! Enable PLOT_MACROS or PLOT_HEATFLUX_KN.")
    sys.exit(1)

if macros:
    if getattr(sys.modules[__name__], "PLOT_PER_SOLVER_ROWS", False):
        # ─────────────────────────────────────────────────────────────────
        # MULTI-ROW GRID (Rows = Solvers, Cols = Macros)
        # ─────────────────────────────────────────────────────────────────
        unique_solvers = list(dict.fromkeys([key[0] for key in data.keys()]))
        nrows = len(unique_solvers)
        ncols = len(macros)

        fig, axes = make_axes(nrows, ncols)

        for row_i, solver in enumerate(unique_solvers):
            for col_i, macro in enumerate(macros):
                ax = axes[row_i, col_i]

                solver_variants = defaultdict(int)
                fm_labeled = False
                cont_labeled = False

                for Kn in Kn_list:
                    rho_0 = _rho_from_Kn(Kn)
                    qs = rho_0 * U0**3 if PLOT_DIMENSIONAL else 1.0

                    plot_fm_this_kn = False
                    if isinstance(PLOT_FM, list):
                        plot_fm_this_kn = any(abs(Kn - k) < 1e-9 for k in PLOT_FM)
                    else:
                        plot_fm_this_kn = bool(PLOT_FM)

                    if PLOT_CONT_PHYS or PLOT_CONT_BGK:
                        add_continuum(
                            ax, macro, Kn, rho_0, qs, label_it=not cont_labeled
                        )
                        cont_labeled = True

                    if plot_fm_this_kn:
                        add_free_molecular(
                            ax, macro, rho_0, qs, label_it=not fm_labeled
                        )
                        fm_labeled = True

                    if Kn in ref_data_dict:
                        add_reference(
                            ax,
                            macro,
                            ref_data_dict[Kn]["data"],
                            rho_0,
                            qs,
                            Kn,
                            ref_data_dict[Kn]["tuple"],
                        )

                    solver_runs = [
                        key
                        for key in data.keys()
                        if key[0] == solver and abs(key[2] - Kn) < 1e-9
                    ]

                    for key in solver_runs:
                        r = data[key]
                        col, mk, ls = run_style(solver, solver_variants[solver])
                        solver_variants[solver] += 1

                        me = max(1, len(r["x"]) // 12)
                        n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
                        lbl = rf"{solver.upper()} (${n_label}$={key[1]}, CFL={key[3]}, Kn={Kn})"
                        add_data(ax, macro, r, col, mk, ls, me, lbl, rho_0, qs)

                finish_panel(ax, macro)

                if col_i == 0:
                    ax.annotate(
                        solver.upper(),
                        xy=(0.03, 0.95),
                        xycoords="axes fraction",
                        ha="left",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", ec="silver", alpha=0.9
                        ),
                    )

                if row_i < nrows - 1:
                    ax.set_xlabel("")

    else:
        # ─────────────────────────────────────────────────────────────────
        # STANDARD 1D LAYOUT (All solvers stacked together)
        # ─────────────────────────────────────────────────────────────────
        nrows, ncols = resolve_layout(len(macros), LAYOUT)
        fig, axes = make_axes(nrows, ncols, n_panels_to_keep=len(macros))
        ax_flat = axes.flatten()

        for panel_i, (macro, ax) in enumerate(zip(macros, ax_flat[: len(macros)])):
            global_variant = 0
            fm_labeled = False
            cont_labeled = False

            for Kn in Kn_list:
                rho_0 = _rho_from_Kn(Kn)
                qs = rho_0 * U0**3 if PLOT_DIMENSIONAL else 1.0

                plot_fm_this_kn = False
                if isinstance(PLOT_FM, list):
                    plot_fm_this_kn = any(abs(Kn - k) < 1e-9 for k in PLOT_FM)
                else:
                    plot_fm_this_kn = bool(PLOT_FM)

                if PLOT_CONT_PHYS or PLOT_CONT_BGK:
                    add_continuum(ax, macro, Kn, rho_0, qs, label_it=not cont_labeled)
                    cont_labeled = True

                if plot_fm_this_kn:
                    add_free_molecular(ax, macro, rho_0, qs, label_it=not fm_labeled)
                    fm_labeled = True

                if Kn in ref_data_dict:
                    add_reference(
                        ax,
                        macro,
                        ref_data_dict[Kn]["data"],
                        rho_0,
                        qs,
                        Kn,
                        ref_data_dict[Kn]["tuple"],
                    )

                runs_for_this_Kn = [
                    key for key in data.keys() if abs(key[2] - Kn) < 1e-9
                ]

                for key in runs_for_this_Kn:
                    solver = key[0]
                    r = data[key]

                    col, mk, ls = run_style(solver, global_variant)
                    global_variant += 1

                    me = max(1, len(r["x"]) // 12)
                    n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
                    lbl = rf"{solver.upper()} (${n_label}$={key[1]}, CFL={key[3]}, Kn={Kn})"
                    add_data(ax, macro, r, col, mk, ls, me, lbl, rho_0, qs)

            finish_panel(ax, macro)

        if LEGEND_MACRO == "empty" and len(macros) < len(ax_flat):
            handles, labels = ax_flat[0].get_legend_handles_labels()
            empty_ax = ax_flat[len(macros)]
            empty_ax.set_visible(True)
            empty_ax.axis("off")
            empty_ax.legend(handles, labels, loc="center")

    save_or_show("fourier_particle_profiles_combined.pdf")

# ═════════════════════════════════════════════════════════════════════════════
# HEAT FLUX VS Kn
# ═════════════════════════════════════════════════════════════════════════════
if PLOT_HEATFLUX_KN:
    runs_by_series = defaultdict(list)
    for (solver, Nc, Kn, CFL), r in data.items():
        if "qx" in r:
            i_mid = len(r["qx"]) // 2
            qs = _rho_from_Kn(Kn) * U0**3 if PLOT_DIMENSIONAL else 1.0
            runs_by_series[(solver, Nc, CFL)].append((Kn, np.abs(r["qx"][i_mid]) * qs))

    if runs_by_series:
        width, height = tp.get_figsize(fraction=0.7)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

        Kn_range = np.logspace(-2, 2, 300)
        rho_range = np.array([_rho_from_Kn(K) for K in Kn_range])
        qs_range = rho_range * U0**3 if PLOT_DIMENSIONAL else np.ones_like(Kn_range)
        mu_cont = Kn_range * np.sqrt(2.0 / np.pi)

        if PLOT_CONT_PHYS:
            q_phys_nd = np.abs(
                fourier_heat_flux_continuum(
                    T_low_nd, T_upp_nd, omega, mu_cont, R_nd, n_v, Pr=Pr_Ar
                )
            )
            lbl_phys = (
                rf"Continuum Ar ($\mathrm{{Pr}}={Pr_Ar:.2f}$, $\mathrm{{Kn}}\leq 1$)"
                + (" $\leftarrow$ sim" if Pr_sim == Pr_Ar else "")
            )
            ax.loglog(
                Kn_range[Kn_range <= 1.0],
                (q_phys_nd * qs_range)[Kn_range <= 1.0],
                "k-",
                lw=1.2,
                label=lbl_phys,
            )

        if PLOT_CONT_BGK:
            q_bgk_nd = np.abs(
                fourier_heat_flux_continuum(
                    T_low_nd, T_upp_nd, omega, mu_cont, R_nd, n_v, Pr=1.0
                )
            )
            lbl_bgk = r"Continuum BGK ($\mathrm{Pr}=1$, $\mathrm{Kn}\leq 1$)" + (
                " $\leftarrow$ sim" if Pr_sim == 1.0 else ""
            )
            ax.loglog(
                Kn_range[Kn_range <= 1.0],
                (q_bgk_nd * qs_range)[Kn_range <= 1.0],
                "k-",
                lw=1.2,
                label=lbl_bgk,
            )

        plot_fm_summary = PLOT_FM is True or (
            isinstance(PLOT_FM, list) and len(PLOT_FM) > 0
        )
        if plot_fm_summary:
            rho_L_nd = 2.0 / (1.0 + np.sqrt(T_low_nd / T_upp_nd))
            q_fm_nd = np.abs(
                fourier_heat_flux_free_molecular(
                    rho_L_nd, T_low_nd, T_upp_nd, R_nd, n_v
                )
            )
            lbl_fm = rf"Free mol. ($n_v={n_v}$, eq.~B.59, $\mathrm{{Kn}}\geq 1$)"
            ax.loglog(
                Kn_range[Kn_range >= 1.0],
                (q_fm_nd * qs_range)[Kn_range >= 1.0],
                "k--",
                lw=1.0,
                label=lbl_fm,
            )

        global_variant = 0

        for idx, ((solver, Nc, CFL), pts) in enumerate(runs_by_series.items()):
            pts = sorted(pts)
            col, mk, ls = run_style(solver, global_variant)
            global_variant += 1

            n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
            ax.loglog(
                [p[0] for p in pts],
                [p[1] for p in pts],
                ls,
                color=col,
                marker=mk,
                ms=2,
                lw=0.8,
                label=rf"{solver.upper()} ${n_label}$={Nc} CFL={CFL}",
            )

        ax.set_xlabel(r"$\mathrm{Kn}$")
        ax.set_ylabel(
            r"$|q_x|$ at $x=L/2$ [W/m$^2$]"
            if PLOT_DIMENSIONAL
            else r"$|q_x|$ at $x=L/2$ (non-dim.)"
        )
        # ax.set_title(
        #     rf"Fourier flow — $|q_x|$ vs $\mathrm{{Kn}}$, $N_v={nv_tag}$, variable $\hat{{t}}$"
        # )
        ax.legend()
        ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
        ax.set_xlim(1e-2, 1e2)

        save_or_show("fourier_particle_heatflux_vs_Kn.pdf")

print("\nDone.")
