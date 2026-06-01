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
from matplotlib.lines import Line2D

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

# Each entry: (solver_name, Nc, Kn, t_final, CFL)
PLOT_RUNS = [
    # # Kn = 0.001
    ("strang", 11, 0.001, 200.0, 0.9),
    ("sl", 11, 0.001, 200.0, 0.9),
    ("hybrid", 11, 0.001, 200.0, 0.9),
    ("fvm", 10, 0.001, 200.0, 0.9),
    ("ugks", 10, 0.001, 200.0, 0.9),
    # # Kn = 0.01
    # ("strang", 11, 0.01, 50.0, 0.9),
    # ("sl", 11, 0.01, 50.0, 0.9),
    # ("fvm", 10, 0.01, 50.0, 0.9),
    # ("ugks", 10, 0.01, 50.0, 0.9),
    # ("hybrid", 11, 0.01, 50.0, 0.9),
    # # Kn = 0.1
    # ("strang", 21, 0.1, 50.0, 0.9),
    # ("sl", 21, 0.1, 50.0, 0.9),
    # ("fvm", 20, 0.1, 50.0, 0.9),
    # ("ugks", 20, 0.1, 50.0, 0.9),
    # ("hybrid", 21, 0.1, 50.0, 0.9),
    # Kn = 1
    # ("strang", 11, 1, 50.0, 0.9),
    # ("sl", 11, 1, 50.0, 0.9),
    # ("fvm", 10, 1, 50.0, 0.9),
    # ("ugks", 10, 1, 50.0, 0.9),
    # ("hybrid", 11, 1, 50.0, 0.9),
    # # Kn = 10
    # ("strang", 11, 10, 50.0, 0.9),
    # ("sl", 11, 10, 50.0, 0.9),
    # ("fvm", 10, 10, 50.0, 0.9),
    # ("ugks", 10, 10, 50.0, 0.9),
    # ("hybrid", 11, 10, 50.0, 0.9),
    # Kn = 100
    # ("strang", 11, 100, 50.0, 0.9),
    # ("sl", 11, 100, 50.0, 0.9),
    # ("fvm", 10, 100, 50.0, 0.9),
    # ("ugks", 10, 100, 50.0, 0.9),
    # ("hybrid", 11, 100, 50.0, 0.9),
    # ("strang", 11, 0.001, 200.0, 0.6),
    # # Strang
    # ("strang", 11, 0.001, 200.0, 0.9),
    # ("strang", 21, 0.001, 200.0, 0.9),
    # ("strang", 31, 0.001, 200.0, 0.9),
    # ("strang", 41, 0.001, 200.0, 0.9),
    # ("strang", 101, 0.001, 200.0, 0.9),
    # # SL
    # ("sl", 11, 0.001, 200.0, 0.9),
    # ("sl", 21, 0.001, 200.0, 0.9),
    # ("sl", 31, 0.001, 200.0, 0.9),
    # ("sl", 41, 0.001, 200.0, 0.9),
    # ("sl", 101, 0.001, 200.0, 0.9),
    # # hybrid
    # ("hybrid", 11, 0.001, 200.0, 0.9),
    # ("hybrid", 21, 0.001, 200.0, 0.9),
    # ("hybrid", 31, 0.001, 200.0, 0.9),
    # ("hybrid", 41, 0.001, 200.0, 0.9),
    # ("hybrid", 101, 0.001, 200.0, 0.9),
    # # FVM
    # ("fvm", 10, 0.001, 200.0, 0.9),
    # ("fvm", 20, 0.001, 200.0, 0.9),
    # ("fvm", 30, 0.001, 200.0, 0.9),
    # ("fvm", 40, 0.001, 200.0, 0.9),
    # ("fvm", 100, 0.001, 200.0, 0.9),
    # # UGKS
    # ("ugks", 10, 0.001, 200.0, 0.9),
    # ("ugks", 20, 0.001, 200.0, 0.9),
    # ("ugks", 30, 0.001, 200.0, 0.9),
    # ("ugks", 40, 0.001, 200.0, 0.9),
    # ("ugks", 100, 0.001, 200.0, 0.9),
]

NVX = 28
NVY = None  # None for dim_v=1, int for dim_v=2

# Provide multiple references for different Kn numbers here
# Format: (solver_name, Nc, Kn, t_final, CFL)
REFERENCES = [("fvm", 400, 1, 50.0, 0.9)]

REF_NVX = 28
REF_NVY = None

RESULTS_DIR = "code/bgk/experiments/fourier/simulations"
OUTPUT_DIR = "latex/thesis/figures/ch4/fourier"
SAVE_PLOTS = False  # True → save PDFs, False → show interactively

# ── Feature Toggles ───────────────────────────────────────────────────────────
PLOT_DIMENSIONAL = False
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
PLOT_CONT_BGK = False  # continuum heat flux with Pr=1 (BGK limit)
PLOT_CONT_PHYS = False  # continuum heat flux with Pr=2/3 (physical Argon)

# Select explicitly WHICH Knudsen numbers get a Free Molecular limit line.
# Set to True for all, False for none, or a list like [1000, 10]
PLOT_FM = [100]

PLOT_HEATFLUX_KN = False  # Produce the global summary plot at the end

# ── Legend content toggles ────────────────────────────────────────────────────
LEGEND_SHOW_CFL = False  # include CFL=... in run labels
LEGEND_SHOW_KN = False  # include Kn=...  in run labels (profile plots only)
# True → structured legend with one section per Kn (filled vs hollow markers per Kn)
MULTI_KN_LEGEND = True
# Runs placed at the end of the legend with a custom style.
# Maps (solver, Nc, Kn, CFL) → (color, marker, linestyle).  Set to {} to disable.
LEGEND_TRAILING = {
    ("strang", 11, 0.001, 0.6): ("tab:blue", "*", "-"),
}

# ── Plot order ────────────────────────────────────────────────────────────────
# Solver names in the order they should be drawn (bottom-to-top / first-to-last).
# Set to None or [] to keep the order from PLOT_RUNS.
PLOT_ORDER = ["strang", "sl", "hybrid", "fvm", "ugks"]

# ── Row height ────────────────────────────────────────────────────────────────
# Scales the height of each row relative to its natural (square-ish) size.
# Only used when PLOT_PER_SOLVER_ROWS = True.  1.0 = default, < 1.0 = shorter rows.
ROW_HEIGHT_SCALE = 0.6

# ── Marker density ────────────────────────────────────────────────────────────
# MARKER_EVERY     : markevery for run data.  1 = a marker at every grid point.
# REF_MARKER_EVERY : markevery for the reference.  Increase to thin out markers
#                    when the reference grid is much finer than the run grids.
MARKER_EVERY = 1
REF_MARKER_EVERY = 5

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
}

SOLVER_MARKERS = {
    "strang": "o",
    "sl": "^",
    "fvm": "s",
    "ugks": "D",
    "hybrid": "v",
}

_MARKERS = ["P", "X", "*", "p", "h"]  # fallback for unknown solvers
_LINES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
_VARIANT_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]  # per-run cycling list

MARKER_SIZE = 4  # marker size for simulation and legend entries
LINE_WIDTH = 1.5  # line width for simulation runs
LEGEND_FONTSIZE = 10  # font size for all legends


def run_style(solver_name, variant_idx):
    color = SOLVER_COLORS.get(solver_name.lower(), "k")
    marker = SOLVER_MARKERS.get(
        solver_name.lower(), _MARKERS[variant_idx % len(_MARKERS)]
    )
    ls = _LINES[variant_idx % len(_LINES)]
    return color, marker, ls


def _order_runs(run_keys):
    if not PLOT_ORDER:
        return list(run_keys)
    rank = {s.lower(): i for i, s in enumerate(PLOT_ORDER)}
    return sorted(run_keys, key=lambda k: rank.get(k[0].lower(), len(PLOT_ORDER)))


def make_run_label(solver, Nc, Kn=None, CFL=None):
    n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
    extras = []
    if Kn is not None and LEGEND_SHOW_KN:
        extras.append(rf"$\mathrm{{Kn}}={Kn:g}$")
    if CFL is not None and LEGEND_SHOW_CFL:
        extras.append(f"CFL={CFL}")
    base = rf"{solver.upper()} (${n_label}$={Nc}"
    return (base + ", " + ", ".join(extras) + ")") if extras else (base + ")")


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


def make_axes(nrows, ncols, n_panels_to_keep=None, row_height_scale=1.0):
    total_width, base_height = tp.get_figsize(fraction=1.0, ratio=0.8)
    aspect_ratio = base_height / total_width

    panel_width = total_width / ncols
    panel_height = panel_width * aspect_ratio * row_height_scale
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
for solver, Nc, Kn, t_final, CFL in PLOT_RUNS:
    r = load_result(solver, Nc, NVX, NVY, Kn, t_final, "fourier", RESULTS_DIR, CFL)
    if r is not None:
        data[(solver, Nc, Kn, CFL)] = r

ref_data_dict = {}
if REFERENCES is not None:
    for ref in REFERENCES:
        if ref is None:
            continue
        r_ref = load_result(
            ref[0],
            ref[1],
            REF_NVX,
            REF_NVY,
            ref[2],
            ref[3],
            "fourier",
            RESULTS_DIR,
            ref[4],
        )
        if r_ref is not None:
            ref_data_dict[ref[2]] = {"data": r_ref, "tuple": ref}

if not data:
    print("No results found. Run run_fourier.py first.")
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

nv_tag = str(NVX) if NVY is None else f"{NVX}x{NVY}"
x_ref_nd = np.linspace(0.0, 1.0, 300)


def _rho_from_Kn(Kn):
    mu_0_ = float(_r0["mu_0"])
    T_ref_ = float(_r0["T_ref"])
    Pr_Ar_ = float(_r0["Pr_Ar"])
    R_s_ = 208.13
    tau_mode_ = str(_r0.get("tau_mode", "viscosity"))
    factor = np.sqrt(np.pi / (2.0 * R_s_ * T_ref_))
    if tau_mode_ == "viscosity":
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
    lbl = "Continuum" if label_it else None

    if macro == "temperature":
        T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)
        y = T_cont * T_ref if PLOT_DIMENSIONAL else T_cont
        ax.plot(x_ref_nd, y, "k-", lw=1.2, label=lbl, zorder=0)
    elif macro == "temperature_deviation":
        T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)
        y = T_cont * T_ref if PLOT_DIMENSIONAL else T_cont
        y = y - get_linear_baseline(x_ref_nd)
        ax.plot(x_ref_nd, y, "k-", lw=1.2, label=lbl, zorder=0)
    # elif macro == "density":
    #     T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)
    #     rho_cont_nd = 1.0 / T_cont
    #     y = rho_cont_nd * rho_0 if PLOT_DIMENSIONAL else rho_cont_nd
    #     ax.plot(x_ref_nd, y, "k-", lw=1.2, label=lbl, zorder=0)
    elif macro == "density":
        T_cont = fourier_temperature_continuum(x_ref_nd, T_low_nd, T_upp_nd, omega)

        # Exact analytical normalization constant C
        num = (T_upp_nd ** (omega + 1)) - (T_low_nd ** (omega + 1))
        den = (T_upp_nd**omega) - (T_low_nd**omega)
        C = (omega / (omega + 1.0)) * (num / den)

        rho_cont_nd = C / T_cont
        y = rho_cont_nd * rho_0 if PLOT_DIMENSIONAL else rho_cont_nd
        ax.plot(x_ref_nd, y, "k-", lw=1.2, label=lbl, zorder=0)
    elif macro == "velocity":
        ax.axhline(0.0, color="k", lw=1.2, ls="-", label=lbl, zorder=0)
    elif macro == "heat_flux":
        if PLOT_CONT_BGK:
            q_bgk = fourier_heat_flux_continuum(
                T_low_nd, T_upp_nd, omega, mu_0_nd, R_nd, n_v, Pr=1.0
            )
            y = np.abs(q_bgk) * qs if PLOT_DIMENSIONAL else np.abs(q_bgk)
            lbl_bgk = r"Continuum ($\mathrm{Pr}=1$)" if label_it else None
            ax.axhline(y, color="k", lw=1.2, ls="-", label=lbl_bgk, zorder=0)
        if PLOT_CONT_PHYS:
            q_phys = fourier_heat_flux_continuum(
                T_low_nd, T_upp_nd, omega, mu_0_nd, R_nd, n_v, Pr=Pr_Ar
            )
            y = np.abs(q_phys) * qs if PLOT_DIMENSIONAL else np.abs(q_phys)
            lbl_phys = (
                rf"Continuum Ar ($\mathrm{{Pr}}={Pr_Ar:.2f}$)" if label_it else None
            )
            ax.axhline(y, color="k", lw=1.2, ls="-", label=lbl_phys, zorder=0)


def add_free_molecular(ax, macro, rho_0, qs, label_it=True):
    lbl = r"Free mol." if label_it else None

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
        ms=MARKER_SIZE,
        markevery=REF_MARKER_EVERY,
        zorder=1,
        label=rf"Ref.: {solver_name.upper()} $({n_label}$={Nc})",
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
        if (r["qx"] < 0).any():
            print(
                "  WARNING: reference qx has negative values; plotting absolute value"
            )
        y = np.abs(r["qx"]) * qs if PLOT_DIMENSIONAL else np.abs(r["qx"])
        ax.semilogy(r["x"], y, **kw)


def add_data(ax, macro, r, col, mk, ls, me, lbl, rho_0, qs, fillstyle=None):
    kw = dict(
        color=col,
        marker=mk,
        ms=MARKER_SIZE,
        lw=LINE_WIDTH,
        markevery=me,
        zorder=3,
        label=lbl,
        fillstyle=fillstyle if fillstyle is not None else "full",
    )

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
        if (r["qx"] < 0).any():
            print("  WARNING: qx has negative values; plotting absolute value")
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


def finish_panel(ax, macro, show_legend=True):
    ax.set_xlabel(r"$x / L$")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    if macro in Y_LIMITS and Y_LIMITS[macro] is not None:
        ax.set_ylim(Y_LIMITS[macro])

    add_zoom_inset(ax, macro)
    if show_legend and macro == LEGEND_MACRO:
        ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)

    if macro == "temperature":
        ax.set_ylabel(r"$T$ [K]" if PLOT_DIMENSIONAL else r"$\hat{T}$")
        if getattr(sys.modules[__name__], "PLOT_LOG_TEMPERATURE", False):
            ax.set_yscale("log")
    elif macro == "temperature_deviation":
        ax.set_ylabel(
            r"$\Delta T$ from linear [K]"
            if PLOT_DIMENSIONAL
            else r"$\Delta T$ from linear"
        )
        ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)
    elif macro == "density":
        ax.set_ylabel(r"$\rho$ [kg/m$^3$]" if PLOT_DIMENSIONAL else r"$\hat{\rho}$")
    elif macro == "velocity":
        ax.set_ylabel(r"$U_x$ [m/s]" if PLOT_DIMENSIONAL else r"$U_x / U_0$")
    elif macro == "heat_flux":
        ax.set_ylabel(r"$|q_x|$ [W/m$^2$]" if PLOT_DIMENSIONAL else r"$|\hat{q}_x|$")


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

        fig, axes = make_axes(nrows, ncols, row_height_scale=ROW_HEIGHT_SCALE)

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

                    solver_runs = _order_runs(
                        key
                        for key in data.keys()
                        if key[0] == solver and abs(key[2] - Kn) < 1e-9
                    )

                    for key in solver_runs:
                        r = data[key]
                        variant = solver_variants[solver]
                        col = SOLVER_COLORS.get(solver.lower(), "k")
                        mk = _VARIANT_MARKERS[variant % len(_VARIANT_MARKERS)]
                        ls = _LINES[variant % len(_LINES)]
                        solver_variants[solver] += 1

                        me = MARKER_EVERY
                        lbl = make_run_label(solver, key[1], Kn=Kn, CFL=key[3])
                        add_data(ax, macro, r, col, mk, ls, me, lbl, rho_0, qs)

                # Panel legend: shown per-row when LEGEND_MACRO names a valid macro
                finish_panel(ax, macro, show_legend=LEGEND_MACRO in macros)

                if col_i == 0:
                    ax.annotate(
                        solver.upper(),
                        xy=(0.03, 0.95),
                        xycoords="axes fraction",
                        ha="left",
                        va="top",
                        fontsize=8,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", ec="silver", alpha=0.9
                        ),
                    )

                if row_i < nrows - 1:
                    ax.set_xlabel("")

        # Figure-level right legend: used when LEGEND_MACRO is not a valid macro
        # (e.g. LEGEND_MACRO = "right").  Set LEGEND_MACRO to a macro name to get
        # per-panel legends instead.
        if LEGEND_MACRO not in macros:
            handles_all: list = []
            labels_all: list = []
            for row_i in range(nrows):
                h, lbls = axes[row_i, 0].get_legend_handles_labels()
                handles_all.extend(h)
                labels_all.extend(lbls)
            fig.tight_layout(rect=[0, 0, 0.72, 1])
            fig.legend(
                handles_all,
                labels_all,
                loc="center left",
                bbox_to_anchor=(0.73, 0.5),
                fontsize=LEGEND_FONTSIZE,
            )

    else:
        # ─────────────────────────────────────────────────────────────────
        # STANDARD 1D LAYOUT (All solvers stacked together)
        # ─────────────────────────────────────────────────────────────────
        nrows, ncols = resolve_layout(len(macros), LAYOUT)
        fig, axes = make_axes(nrows, ncols, n_panels_to_keep=len(macros))
        ax_flat = axes.flatten()

        fillstyles = ["full", "none"]

        for panel_i, (macro, ax) in enumerate(zip(macros, ax_flat[: len(macros)])):
            fm_labeled = False
            cont_labeled = False

            for kn_idx, Kn in enumerate(Kn_list):
                fs = fillstyles[kn_idx % 2]
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

                runs_for_this_Kn = _order_runs(
                    key for key in data.keys() if abs(key[2] - Kn) < 1e-9
                )

                for local_idx, key in enumerate(runs_for_this_Kn):
                    solver = key[0]
                    r = data[key]
                    if key in LEGEND_TRAILING:
                        col, mk, ls = LEGEND_TRAILING[key]
                    else:
                        col, mk, ls = run_style(solver, local_idx)
                    me = MARKER_EVERY
                    lbl = make_run_label(solver, key[1], Kn=Kn, CFL=key[3])
                    add_data(
                        ax, macro, r, col, mk, ls, me, lbl, rho_0, qs, fillstyle=fs
                    )

            finish_panel(ax, macro, show_legend=not MULTI_KN_LEGEND)

        if MULTI_KN_LEGEND:
            handles: list = []
            labels: list = []

            if PLOT_CONT_BGK:
                handles.append(Line2D([], [], color="k", ls="-", lw=1.2))
                labels.append(r"Continuum ($\mathrm{Pr}=1$)")
            if PLOT_CONT_PHYS:
                handles.append(Line2D([], [], color="k", ls="-", lw=1.2))
                labels.append(rf"Continuum Ar ($\mathrm{{Pr}}={Pr_Ar:.2f}$)")
            plot_fm_any = PLOT_FM is True or (
                isinstance(PLOT_FM, list) and len(PLOT_FM) > 0
            )
            if plot_fm_any:
                handles.append(Line2D([], [], color="k", ls="--", lw=1.0))
                labels.append(r"Free mol.")

            for kn_idx, Kn in enumerate(Kn_list):
                fs = fillstyles[kn_idx % 2]
                handles.append(Line2D([], [], color="none"))
                labels.append(rf"$\mathbf{{Kn = {Kn:g}}}$")

                if Kn in ref_data_dict:
                    ref_tuple = ref_data_dict[Kn]["tuple"]
                    solver_name = ref_tuple[0]
                    Nc_ref = ref_tuple[1]
                    n_label = "N_c" if solver_name.lower() in ["ugks", "fvm"] else "N_x"
                    handles.append(
                        Line2D(
                            [],
                            [],
                            color="dimgrey",
                            marker="o",
                            ls="none",
                            ms=MARKER_SIZE,
                            fillstyle="none",
                        )
                    )
                    labels.append(
                        rf"Ref.: {solver_name.upper()} (${n_label}$={Nc_ref})"
                    )

                runs_for_this_Kn = _order_runs(
                    key
                    for key in data.keys()
                    if abs(key[2] - Kn) < 1e-9 and key not in LEGEND_TRAILING
                )
                for local_idx, key in enumerate(runs_for_this_Kn):
                    solver = key[0]
                    col, mk, ls = run_style(solver, local_idx)
                    n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
                    cfl_str = (
                        f", CFL={key[3]}"
                        if key[3] is not None and LEGEND_SHOW_CFL
                        else ""
                    )
                    handles.append(
                        Line2D(
                            [],
                            [],
                            color=col,
                            marker=mk,
                            ls=ls,
                            ms=MARKER_SIZE,
                            lw=LINE_WIDTH,
                            fillstyle=fs,
                        )
                    )
                    labels.append(rf"{solver.upper()} (${n_label}$={key[1]}{cfl_str})")

            trailing_keys = _order_runs(
                key for key in data.keys() if key in LEGEND_TRAILING
            )
            if trailing_keys:
                handles.append(Line2D([], [], color="none"))
                labels.append(r"$\mathbf{Extra}$")
                for key in trailing_keys:
                    solver = key[0]
                    col, mk, ls = LEGEND_TRAILING[key]
                    kn_idx = next(
                        (i for i, K in enumerate(Kn_list) if abs(K - key[2]) < 1e-9),
                        0,
                    )
                    fs = fillstyles[kn_idx % 2]
                    n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
                    cfl_str = (
                        f", CFL={key[3]}"
                        if key[3] is not None and LEGEND_SHOW_CFL
                        else ""
                    )
                    kn_str = rf", $\mathrm{{Kn}}={key[2]:g}$" if LEGEND_SHOW_KN else ""
                    handles.append(
                        Line2D(
                            [],
                            [],
                            color=col,
                            marker=mk,
                            ls=ls,
                            ms=MARKER_SIZE,
                            lw=LINE_WIDTH,
                            fillstyle=fs,
                        )
                    )
                    labels.append(
                        rf"{solver.upper()} (${n_label}$={key[1]}{kn_str}{cfl_str})"
                    )

            if LEGEND_MACRO == "empty" and len(macros) < len(ax_flat):
                empty_ax = ax_flat[len(macros)]
                empty_ax.set_visible(True)
                empty_ax.axis("off")
                empty_ax.legend(handles, labels, loc="center", fontsize=LEGEND_FONTSIZE)
            else:
                fig.tight_layout(rect=[0, 0, 0.70, 1])
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(0.72, 0.5),
                    fontsize=LEGEND_FONTSIZE,
                )

        elif LEGEND_MACRO == "empty" and len(macros) < len(ax_flat):
            handles, labels = ax_flat[0].get_legend_handles_labels()
            empty_ax = ax_flat[len(macros)]
            empty_ax.set_visible(True)
            empty_ax.axis("off")
            empty_ax.legend(handles, labels, loc="center")

    save_or_show("fourier_profiles_combined.pdf")

# ═════════════════════════════════════════════════════════════════════════════
# HEAT FLUX VS Kn
# ═════════════════════════════════════════════════════════════════════════════
if PLOT_HEATFLUX_KN:
    runs_by_series = defaultdict(list)
    for (solver, Nc, Kn, CFL), r in data.items():
        if "qx" in r:
            i_mid = len(r["qx"]) // 2
            if r["qx"][i_mid] < 0:
                print("  WARNING: qx[mid] is negative; plotting absolute value")
            qs = _rho_from_Kn(Kn) * U0**3 if PLOT_DIMENSIONAL else 1.0
            runs_by_series[(solver, Nc, CFL)].append((Kn, np.abs(r["qx"][i_mid]) * qs))

    if runs_by_series:
        width, height = tp.get_figsize(fraction=0.8)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

        Kn_range = np.logspace(-3, 2, 300)
        rho_range = np.array([_rho_from_Kn(K) for K in Kn_range])
        qs_range = rho_range * U0**3 if PLOT_DIMENSIONAL else np.ones_like(Kn_range)
        mu_cont = Kn_range * np.sqrt(2.0 / np.pi)

        if PLOT_CONT_PHYS:
            q_phys_nd = np.abs(
                fourier_heat_flux_continuum(
                    T_low_nd, T_upp_nd, omega, mu_cont, R_nd, n_v, Pr=Pr_Ar
                )
            )
            ax.loglog(
                Kn_range,
                q_phys_nd * qs_range,
                "k-",
                lw=1.2,
                label=rf"Continuum Ar ($\mathrm{{Pr}}={Pr_Ar:.2f}$)",
            )

        if PLOT_CONT_BGK:
            q_bgk_nd = np.abs(
                fourier_heat_flux_continuum(
                    T_low_nd, T_upp_nd, omega, mu_cont, R_nd, n_v, Pr=1.0
                )
            )
            ax.loglog(
                Kn_range,
                q_bgk_nd * qs_range,
                "k-",
                lw=1.2,
                label=r"Continuum ($\mathrm{Pr}=1$)",
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
            ax.loglog(Kn_range, q_fm_nd * qs_range, "k--", lw=1.0, label=r"Free mol.")

        # Build a (solver, Nc, CFL) → trailing style lookup
        _trailing_by_series = {
            (s, n, c): style for (s, n, _, c), style in LEGEND_TRAILING.items()
        }

        all_series = _order_runs((s, n, c) for s, n, c in runs_by_series)
        normal_series = [k for k in all_series if k not in _trailing_by_series]
        trailing_series = [k for k in all_series if k in _trailing_by_series]

        global_variant = 0
        for solver, Nc, CFL in normal_series + trailing_series:
            pts = sorted(runs_by_series[(solver, Nc, CFL)])
            if (solver, Nc, CFL) in _trailing_by_series:
                col, mk, ls = _trailing_by_series[(solver, Nc, CFL)]
            else:
                col, mk, ls = run_style(solver, global_variant)
                global_variant += 1

            ax.loglog(
                [p[0] for p in pts],
                [p[1] for p in pts],
                ls,
                color=col,
                marker=mk,
                ms=MARKER_SIZE,
                lw=LINE_WIDTH,
                label=make_run_label(solver, Nc, CFL=CFL),
            )

        ax.set_xlabel(r"$\mathrm{Kn}$")
        ax.set_ylabel(
            r"$|q_x|$ [W/m$^2$]" if PLOT_DIMENSIONAL else r"$|q_x|$ (non-dim.)"
        )
        # ax.set_title(
        #     rf"Fourier flow — $|q_x|$ vs $\mathrm{{Kn}}$, $N_v={nv_tag}$, variable $\hat{{t}}$"
        # )
        ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
        ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
        # ax.set_xlim(1e-2, 1e2)

        save_or_show("fourier_heatflux_vs_Kn.pdf")

print("\nDone.")
