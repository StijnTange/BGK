"""
Couette Flow — Plot Script.

Loads saved HDF5 results and produces one figure per Kn with the
macroscopic profiles you select, plus an optional shear-stress vs Kn
log-log summary.

═══════════════════════════════════════════════════════════════════════
USER SETTINGS  (edit this section only)
═══════════════════════════════════════════════════════════════════════

PLOT_MACROS   : list of macros to include as panels.
                Allowed values (order determines panel order):
                  "velocity"      — u_y(x)
                  "temperature"   — T(x)
                  "density"       — rho(x)  [physical kg/m³]
                  "shear_stress"  — |tau_xy|(x)

LAYOUT        : panel arrangement.  Options:
                  "auto"  — picks the most natural grid automatically:
                              1 panel  → (1,1)
                              2 panels → (1,2)
                              3 panels → (1,3)
                              4 panels → (2,2)
                  "1x1", "1x2", "1x3", "1x4", "2x2"  — explicit grid

PLOT_CONTINUUM      : show NSF continuum reference curves
PLOT_FREE_MOLECULAR : show free-molecular horizontal lines
PLOT_SHEAR_KN       : also produce the tau_xy vs Kn log-log summary figure
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.experiments.couette.analytics_couette import (
    couette_shear_stress_continuum,
    couette_shear_stress_free_molecular,
    couette_temperature_continuum,
    couette_temperature_free_molecular,
    couette_velocity_continuum,
    couette_velocity_free_molecular,
)
from bgk.io.io_hdf5 import load_result

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# Each entry: (solver_name, Nc, Kn, T_final)          — old files (no CFL suffix)
#          or (solver_name, Nc, Kn, T_final, CFL)     — new files (_CFL<n>.h5)

PLOT_RUNS = [
    # Kn = 0.001
    ("strang", 11, 0.001, 800.0),
    ("hybrid", 11, 0.001, 800.0),
    ("sl", 11, 0.001, 800.0),
    ("fvm", 10, 0.001, 800.0),
    ("ugks", 10, 0.001, 800.0),
    # Kn = 0.01
    ("strang", 11, 0.01, 100.0),
    ("hybrid", 11, 0.01, 100.0),
    ("sl", 11, 0.01, 100.0),
    ("fvm", 10, 0.01, 100.0),
    ("ugks", 10, 0.01, 100.0),
    # Kn = 0.1
    ("strang", 11, 0.1, 100.0),
    ("hybrid", 11, 0.1, 100.0),
    ("sl", 11, 0.1, 100.0),
    ("fvm", 10, 0.1, 100.0),
    ("ugks", 10, 0.1, 100.0),
    # Kn = 1
    ("strang", 11, 1, 100.0, 0.9),
    ("hybrid", 11, 1, 100.0, 0.9),
    ("sl", 11, 1, 100.0, 0.9),
    ("fvm", 10, 1, 100.0, 0.9),
    ("ugks", 10, 1, 100.0, 0.9),
    # Kn = 10
    ("strang", 11, 10, 100.0),
    ("hybrid", 11, 10, 100.0),
    ("sl", 11, 10, 100.0),
    ("fvm", 10, 10, 100.0),
    ("ugks", 10, 10, 100.0),
    # Kn = 100
    ("strang", 11, 100, 100.0),
    ("hybrid", 11, 100, 100.0),
    ("sl", 11, 100, 100.0),
    ("fvm", 10, 100, 100.0),
    ("ugks", 10, 100, 100.0),
    ("strang", 11, 0.001, 800.0, 0.5),
]


T_FINAL = 200.0
NVX = 30
NVY = 30  # set to None for dim_v=1

# Reference solution per Kn regime.
# Keys are the Kn values in PLOT_RUNS; each value is
#   (solver_name, Nc, Kn_of_ref_run, T_final)  or  None for no reference.
REFERENCE = {
    0.001: ("fvm", 50, 0.001, 800.0),
    100: None,
}
REF_NVX = 30
REF_NVY = 30

RESULTS_DIR = "code/bgk/experiments/couette/simulations"
# OUTPUT_DIR = "code/bgk/experiments/couette/plots"
OUTPUT_DIR = "latex/thesis/figures/ch4/couette"


SAVE_PLOTS = False  # True → save PDFs,  False → show interactively

# ── What to plot ──────────────────────────────────────────────────────────────
# Any subset/ordering of: "velocity", "temperature", "density", "shear_stress"
PLOT_MACROS = ["density", "velocity", "temperature", "shear_stress"]
LEGEND_MACRO = "density"

# Panel grid: "auto" | "1x1" | "1x2" | "1x3" | "1x4" | "2x2"
LAYOUT = "auto"

# Optional y-axis limits for each plot: (ymin, ymax) or None
# Y_LIMITS = {
#     "velocity": (0.5, 0.7),  # e.g., (0.0, 1.2)
#     "temperature": (1.1, 1.3),  # e.g., (0.9, 1.1)
#     "density": (0.9, 1.1),
#     "shear_stress": (0, 1.0),
# }
Y_LIMITS = {
    "velocity": None,  # e.g., (0.0, 1.2)
    "temperature": None,  # e.g., (0.9, 1.1)
    "density": None,
    "shear_stress": None,
}
PLOT_SHEAR_KN_ONLY = True
# Explicit solver display order (use solver names as in PLOT_RUNS).
# Solvers not listed here appear last, in their PLOT_RUNS order.
# Set to None or [] to keep PLOT_RUNS order.
PLOT_ORDER = ["strang", "sl", "hybrid", "fvm", "ugks"]
FILLED_MARKERS = True  # True → solid markers,  False → hollow markers
MARKER_SIZE = 6  # marker size for simulation runs
LINE_WIDTH = 1  # line width for simulation runs
MULTI_KN_PLOT = False  # True → overlay all Kn values in one figure
DIMENSIONAL = False  # True → physical units,  False → non-dimensional
# ── Reference limits ──────────────────────────────────────────────────────────
PLOT_CONTINUUM = True  # NSF continuum curves
PLOT_FREE_MOLECULAR = True  # free-molecular horizontal lines

# ── Extra figures ─────────────────────────────────────────────────────────────
PLOT_SHEAR_KN = True  # tau_xy vs Kn log-log summary

# ── Zoom inset ────────────────────────────────────────────────────────────────
# Set ZOOM_MACRO to the panel name you want to zoom (e.g. "temperature").
# Set ZOOM_MACRO = None to disable the zoom box entirely.
#
# ZOOM_REGION : (x_min, x_max, y_min, y_max) in data coordinates — the
#               rectangular area that will be magnified.
# ZOOM_INSET  : (left, bottom, width, height) in axes-fraction coordinates —
#               where the inset axes sits inside the parent axes.
# ZOOM_CONNECTORS : True to draw lines from the region corners to the inset.
ZOOM_MACRO = "temperature"
ZOOM_REGION = (0.3, 0.7, 1.085, 1.103)  # ← edit to taste
ZOOM_INSET = (0.3, 0.08, 0.46, 0.46)  # (left, bottom, width, height)
# ZOOM_REGION = (0.3, 0.7, 1.16, 1.163)  # ← edit to taste
# ZOOM_INSET = (0.3, 0.08, 0.46, 0.46)  # (left, bottom, width, height)
ZOOM_CONNECTORS = True

# ── Physical constants (must match run_couette.py) ────────────────────────────
Pr_BGK = 1.0
Pr_Ar = 2.0 / 3.0

# ═════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ═════════════════════════════════════════════════════════════════════════════
_COLORS = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:orange",
    "tab:purple",
]
_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]
_LINES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
CFL_MARKER = "*"  # marker used for runs that specify a CFL (overrides default)
CFL_OVERRIDES_MARKER = False  # set False to keep the solver's default marker
SHOW_CFL_IN_LEGEND = False  # False → omit "CFL=..." from all legend labels

SOLVER_COLORS = {
    "strang": "tab:blue",
    "sl": "tab:red",
    "fvm": "tab:green",
    "ugks": "tab:orange",
    "hybrid": "tab:purple",
}
SOLVER_VARIANTS = {
    "strang": 0,
    "hybrid": 1,
    "sl": 2,
    "fvm": 3,
    "ugks": 4,
}


def _sort_runs(runs):
    """Sort a list of (solver, Nc, Kn, CFL) keys by PLOT_ORDER."""
    if not PLOT_ORDER:
        return runs
    order = {s.lower(): i for i, s in enumerate(PLOT_ORDER)}
    unlisted = len(PLOT_ORDER)
    return sorted(runs, key=lambda k: order.get(k[0].lower(), unlisted))


def run_style(solver_name, idx):
    color = SOLVER_COLORS.get(solver_name.lower(), _COLORS[idx % len(_COLORS)])
    variant = SOLVER_VARIANTS.get(solver_name.lower(), idx)
    return (
        color,
        _MARKERS[variant % len(_MARKERS)],
        _LINES[variant % len(_LINES)],
    )


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


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT HELPER
# ═════════════════════════════════════════════════════════════════════════════
def resolve_layout(n_panels, layout_str):
    """Return (nrows, ncols) for n_panels panels."""
    explicit = {
        "1x1": (1, 1),
        "1x2": (1, 2),
        "1x3": (1, 3),
        "1x4": (1, 4),
        "2x2": (2, 2),
    }
    if layout_str in explicit:
        nrows, ncols = explicit[layout_str]
        if nrows * ncols < n_panels:
            print(
                f"  WARNING: layout {layout_str} only fits "
                f"{nrows * ncols} panels but {n_panels} requested — switching to auto"
            )
        else:
            return nrows, ncols
    # auto
    if n_panels == 1:
        return 1, 1
    elif n_panels == 2:
        return 1, 2
    elif n_panels == 3:
        return 1, 3
    else:
        return 2, 2


def make_axes(n_panels, layout_str):
    """Create figure + flat list of n_panels axes."""
    nrows, ncols = resolve_layout(n_panels, layout_str)
    width, height = tp.get_figsize(fraction=1.0)
    # scale figure size with grid
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width * ncols, height * nrows * 1.15),
        constrained_layout=True,
        squeeze=False,
    )
    ax_flat = axes.flatten()
    for i in range(n_panels, nrows * ncols):
        ax_flat[i].set_visible(False)
    return fig, ax_flat[:n_panels]


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
data = {}
for _entry in PLOT_RUNS:
    solver, Nc, Kn, t_final = _entry[:4]
    CFL = _entry[4] if len(_entry) > 4 else None
    r = load_result(solver, Nc, NVX, NVY, Kn, t_final, "couette", RESULTS_DIR, CFL)
    if r is not None:
        data[(solver, Nc, Kn, CFL)] = r

# Load one reference result per Kn entry that is not None
ref_data = {}  # Kn -> loaded result dict
for kn_key, ref_spec in REFERENCE.items():
    if ref_spec is None:
        continue
    _ref_cfl = ref_spec[4] if len(ref_spec) > 4 else None
    r = load_result(
        ref_spec[0],
        ref_spec[1],
        REF_NVX,
        REF_NVY,
        ref_spec[2],
        ref_spec[3],
        "couette",
        RESULTS_DIR,
        _ref_cfl,
    )
    if r is not None:
        ref_data[kn_key] = r
    else:
        print(f"  WARNING: reference {ref_spec} for Kn={kn_key} not found")

if not data:
    print("No results found. Run run_couette.py first.")
    sys.exit(0)

_r0 = next(iter(data.values()))
R_nd = float(_r0["R_nd"])
T_w_nd = float(_r0["T_w_nd"])
u_w_nd = float(_r0["u_w_nd"])
dim_v = int(_r0.get("dim_v", 2))
n_v = dim_v
cp_nd = (n_v / 2.0 + 1.0) * R_nd  # correct for any dim_v
nv_tag = f"{NVX}\\times{NVY}" if NVY is not None else str(NVX)

# Physical scale factors (derived from first loaded run)
_R_s_phys = float(_r0.get("R_s", 208.13))
_T_w_phys = float(_r0.get("T_w", 273.0))
_L_phys = float(_r0.get("L", 1.0))
_U_0 = float(np.sqrt(_R_s_phys * _T_w_phys))  # reference velocity [m/s]


def _scale(values, macro, rho_0=1.0):
    """Scale non-dim values to physical units when DIMENSIONAL is True."""
    if not DIMENSIONAL:
        return values
    factors = {
        "velocity": _U_0,
        "temperature": _T_w_phys,
        "density": rho_0,
        "shear_stress": rho_0 * _U_0**2,
    }
    return np.asarray(values) * factors.get(macro, 1.0)


def _xscale(x):
    return np.asarray(x) * _L_phys if DIMENSIONAL else x


def get_rho_0(r):
    """
    Read rho_0 [kg/m^3] from a loaded result dict.
    New files have it stored directly. Old files are missing it, so
    we recompute from the saved mu_0, Kn, tau_mode, T_w, R_s, L.
    """
    if "rho_0" in r:
        return float(r["rho_0"])
    # Fallback: reconstruct from saved physical constants
    mu_0 = float(r.get("mu_0", 2.117e-5))
    Kn = float(r.get("Kn", 1.0))
    L = float(r.get("L", 1.0))
    T_ref = float(r.get("T_w", 273.0))  # T_ref = T_w for Couette
    R_s_phys = float(r.get("R_s", 208.13))
    Pr_Ar_ = float(r.get("Pr_Ar", 2.0 / 3.0))
    tau_mode = r.get("tau_mode", "viscosity")
    if isinstance(tau_mode, bytes):
        tau_mode = tau_mode.decode()
    mu_eff = mu_0 if tau_mode == "viscosity" else mu_0 / Pr_Ar_
    factor = np.sqrt(np.pi / (2.0 * R_s_phys * T_ref))
    rho_0 = mu_eff * factor / (Kn * L)
    print(f"  WARNING: rho_0 not in HDF5, recomputed as {rho_0:.4e} kg/m³")
    return rho_0


x_ref = np.linspace(0.0, 1.0, 300)
Kn_list = sorted({Kn for (_, _, Kn, _) in data})

# ═════════════════════════════════════════════════════════════════════════════
# PANEL HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def add_continuum(ax, macro, Kn, label_it=True, rho_0=1.0):
    """Overlay NSF continuum limit curves."""
    mu_0_nd = Kn * np.sqrt(2.0 / np.pi)
    if macro == "velocity":
        y = couette_velocity_continuum(x_ref, u_w_nd)
        ax.plot(
            _xscale(x_ref),
            _scale(y, "velocity"),
            "k-",
            lw=1.2,
            label="Continuum" if label_it else "_nolegend_",
            zorder=0,
        )

    elif macro == "temperature":
        T_bgk = couette_temperature_continuum(x_ref, T_w_nd, u_w_nd, Pr_BGK, cp_nd)
        ax.plot(
            _xscale(x_ref),
            _scale(T_bgk, "temperature"),
            "k-",
            lw=1.2,
            label=r"Continuum ($\mathrm{Pr}=1$)" if label_it else "_nolegend_",
            zorder=0,
        )
        # ax.plot(
        #     _xscale(x_ref),
        #     _scale(couette_temperature_continuum(
        #         x_ref, T_w_nd, u_w_nd, Pr_Ar, cp_nd), "temperature"),
        #     "k:",
        #     lw=1.2,
        #     label=rf"Continuum ($\mathrm{{Pr}}={Pr_Ar:.2f}$)",
        #     zorder=5,
        # )

    elif macro == "density":
        T_bgk = couette_temperature_continuum(x_ref, T_w_nd, u_w_nd, Pr_BGK, cp_nd)
        dx = x_ref[1] - x_ref[0]
        C = 1.0 / np.sum(1.0 / T_bgk) / (len(x_ref) - 1) * (x_ref[-1] - x_ref[0])
        # cleaner:
        C = (x_ref[-1] - x_ref[0]) / np.trapezoid(1.0 / T_bgk, x_ref)
        ax._density_cont_nd = C / T_bgk
        ax._density_cont_label_it = label_it

    elif macro == "shear_stress":
        tau_cont = couette_shear_stress_continuum(u_w_nd, mu_0_nd)
        if tau_cont < 0:
            print("  WARNING: continuum tau_xy is negative; plotting absolute value")
        ax.axhline(
            float(_scale(np.abs(tau_cont), "shear_stress", rho_0)),
            color="k",
            lw=1.2,
            ls="-",
            label="Continuum" if label_it else "_nolegend_",
            zorder=0,
        )


def add_free_molecular(ax, macro, label_it=True, rho_0=1.0):
    """Overlay free-molecular limit."""
    lbl = r"Free mol." if label_it else "_nolegend_"
    if macro == "velocity":
        u_fm = couette_velocity_free_molecular(u_w_nd)
        ax.axhline(
            float(_scale(u_fm, "velocity")),
            color="k",
            lw=1.0,
            ls="--",
            label=lbl,
            zorder=4,
        )

    elif macro == "temperature":
        T_fm = couette_temperature_free_molecular(T_w_nd, u_w_nd, R_nd, n_v)
        ax.axhline(
            float(_scale(T_fm, "temperature")),
            color="k",
            lw=1.0,
            ls="--",
            label=lbl,
            zorder=4,
        )

    elif macro == "density":
        # Free-molecular limit: uniform rho_nd = 1. Drawn in add_data once
        # rho_0 is known; stash the non-dim value here.
        ax._density_fm_nd = 1.0
        ax._density_fm_label_it = label_it

    elif macro == "shear_stress":
        tau_fm = couette_shear_stress_free_molecular(1.0, u_w_nd, R_nd, T_w_nd)
        if tau_fm < 0:
            print("  WARNING: free-molecular tau_xy is negative; plotting abs value")
        ax.axhline(
            float(_scale(np.abs(tau_fm), "shear_stress", rho_0)),
            color="k",
            lw=1.2,
            ls="--",
            label=lbl,
            zorder=4,
        )


def add_reference(ax, macro, r, ref_spec):
    """Overlay the reference solution as hollow circles connected by lines."""
    n_label = "N_c" if ref_spec[0].lower() in ["ugks", "fvm"] else "N_x"
    kw = dict(
        color="dimgrey",
        marker="o",
        ls="-",
        fillstyle="none",
        ms=MARKER_SIZE,
        markevery=1,
        zorder=1,
        label=rf"Ref.: {ref_spec[0].upper()} (${n_label}$={ref_spec[1]})",
    )

    if macro == "velocity" and "uy" in r:
        ax.plot(_xscale(r["x"]), _scale(r["uy"], "velocity"), **kw)
    elif macro == "temperature" and "T" in r:
        ax.plot(_xscale(r["x"]), _scale(r["T"], "temperature"), **kw)
    elif macro == "density" and "rho" in r:
        rho_0_ref = get_rho_0(r)
        ax.plot(_xscale(r["x"]), _scale(r["rho"], "density", rho_0_ref), **kw)
    elif macro == "shear_stress" and "tau_xy" in r:
        rho_0_ref = get_rho_0(r)
        if (-r["tau_xy"] < 0).any():
            print("  WARNING: reference tau_xy has negative values; plotting abs value")
        ax.plot(
            _xscale(r["x"]),
            _scale(np.abs(r["tau_xy"]), "shear_stress", rho_0_ref),
            **kw,
        )


def add_data(ax, macro, r, col, mk, ls, me, lbl, fillstyle=None):
    """Plot one simulation run and, for density, the deferred continuum line."""
    kw = dict(
        color=col,
        marker=mk,
        ms=MARKER_SIZE,
        lw=LINE_WIDTH,
        markevery=me,
        label=lbl,
        fillstyle=fillstyle
        if fillstyle is not None
        else ("full" if FILLED_MARKERS else "none"),
    )

    if macro == "velocity" and "uy" in r:
        ax.plot(_xscale(r["x"]), _scale(r["uy"], "velocity"), ls, **kw)

    elif macro == "temperature" and "T" in r:
        ax.plot(_xscale(r["x"]), _scale(r["T"], "temperature"), ls, **kw)

    elif macro == "density" and "rho" in r:
        rho_0 = get_rho_0(r)
        # Draw continuum and free-molecular lines the first time we know rho_0
        if not getattr(ax, "_density_ref_drawn", False):
            if PLOT_CONTINUUM and hasattr(ax, "_density_cont_nd"):
                _cont_lit = getattr(ax, "_density_cont_label_it", True)
                _lbl = "Continuum" if _cont_lit else "_nolegend_"
                ax.plot(
                    _xscale(x_ref),
                    _scale(ax._density_cont_nd, "density", rho_0),
                    "k-",
                    lw=1.2,
                    label=_lbl,
                    zorder=0,
                )
            if PLOT_FREE_MOLECULAR and hasattr(ax, "_density_fm_nd"):
                _fm_lit = getattr(ax, "_density_fm_label_it", True)
                _lbl = r"Free mol." if _fm_lit else "_nolegend_"
                ax.axhline(
                    float(_scale(ax._density_fm_nd, "density", rho_0)),
                    color="k",
                    lw=1.0,
                    ls="--",
                    label=_lbl,
                    zorder=4,
                )
            ax._density_ref_drawn = True
        ax.plot(_xscale(r["x"]), _scale(r["rho"], "density", rho_0), ls, **kw)

    elif macro == "shear_stress" and "tau_xy" in r:
        rho_0 = get_rho_0(r)
        if (-r["tau_xy"] < 0).any():
            print("  WARNING: tau_xy has negative values; plotting absolute value")
        ax.plot(
            _xscale(r["x"]),
            _scale(np.abs(r["tau_xy"]), "shear_stress", rho_0),
            ls,
            **kw,
        )


def add_zoom_inset(ax, macro):
    """
    Add a zoom inset box to *ax* if this panel matches ZOOM_MACRO.

    The inset shows the data region defined by ZOOM_REGION and sits at the
    position defined by ZOOM_INSET (axes-fraction coords).  All lines already
    drawn on *ax* are re-drawn inside the inset automatically.
    """
    if ZOOM_MACRO is None or macro != ZOOM_MACRO:
        return

    x0, x1, y0, y1 = ZOOM_REGION
    il, ib, iw, ih = ZOOM_INSET

    # Create the inset axes
    axins = ax.inset_axes([il, ib, iw, ih])
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)

    # Copy every Line2D from the parent axes into the inset
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
            fillstyle=line.get_fillstyle(),
            zorder=line.get_zorder(),
        )

    # Style the inset
    axins.tick_params()
    axins.grid(alpha=0.3, linestyle="--", linewidth=0.4)

    # Draw the rectangle on the parent axes and (optionally) connecting lines
    ax.indicate_inset_zoom(axins, edgecolor="black", lw=0.8, alpha=0.7)


def finish_panel(ax, macro, show_legend=True):
    """Axis labels and decorations after all data is drawn."""
    ax.set_xlabel(r"$x$ (m)" if DIMENSIONAL else r"$x / L$")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    # Apply custom y-limits if specified by the user
    if macro in Y_LIMITS and Y_LIMITS[macro] is not None:
        ax.set_ylim(Y_LIMITS[macro])
    add_zoom_inset(ax, macro)  # must come before legend so inset is populated

    # Only draw legend for the chosen macro and rely on thesis_plots.py for styling
    if show_legend and macro == LEGEND_MACRO:
        ax.legend(loc="best")

    if macro == "velocity":
        ax.set_ylabel(r"$u_y$ (m/s)" if DIMENSIONAL else r"$\hat{u}_y$")
        ax.axhline(0.0, color="silver", lw=0.6, ls=":")
        ax.axhline(
            float(_scale(u_w_nd, "velocity")),
            color="silver",
            lw=0.6,
            ls=":",
        )
    elif macro == "temperature":
        ax.set_ylabel(r"$T$ (K)" if DIMENSIONAL else r"$\hat{T}$")
    elif macro == "density":
        ax.set_ylabel(r"$\rho$ (kg/m³)" if DIMENSIONAL else r"$\hat{\rho}$")
    elif macro == "shear_stress":
        ax.set_ylabel(r"$\tau_{xy}$ (Pa)" if DIMENSIONAL else r"$\hat{\tau}_{xy}$")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FIGURES — one per Kn
# ═════════════════════════════════════════════════════════════════════════════

# Override standard settings if isolating the shear stress vs Kn summary plot
if getattr(sys.modules[__name__], "PLOT_SHEAR_KN_ONLY", False):
    PLOT_MACROS = []  # Empty the profile list
    PLOT_SHEAR_KN = True  # Ensure the summary plot is turned on

VALID = ("velocity", "temperature", "density", "shear_stress")
macros = [m for m in PLOT_MACROS if m in VALID]

# Only exit if BOTH the profile macros AND the summary plot are disabled
if not macros and not PLOT_SHEAR_KN:
    print("Nothing to plot! Enable PLOT_MACROS or PLOT_SHEAR_KN.")
    sys.exit(1)

# Only run the profile plotting loop if there are actually macros to plot
if macros:
    if not MULTI_KN_PLOT:
        for Kn in Kn_list:
            runs_this_Kn = _sort_runs(
                [(s, Nc, K, cfl) for (s, Nc, K, cfl) in data if abs(K - Kn) < 1e-9]
            )
            rho_0_kn = get_rho_0(data[runs_this_Kn[0]]) if runs_this_Kn else 1.0

            fig, axes = make_axes(len(macros), LAYOUT)

            for panel_i, (macro, ax) in enumerate(zip(macros, axes)):
                if PLOT_CONTINUUM:
                    add_continuum(ax, macro, Kn, rho_0=rho_0_kn)

                if PLOT_FREE_MOLECULAR:
                    add_free_molecular(ax, macro, rho_0=rho_0_kn)

                r_ref = ref_data.get(Kn)
                if r_ref is not None:
                    add_reference(ax, macro, r_ref, REFERENCE[Kn])

                for idx, key in enumerate(runs_this_Kn):
                    r = data[key]
                    col, mk, ls = run_style(key[0], idx)
                    if key[3] is not None and CFL_OVERRIDES_MARKER:
                        mk = CFL_MARKER
                    me = max(1, len(r["x"]) // 12)
                    n_label = "N_c" if key[0].lower() in ["ugks", "fvm"] else "N_x"
                    cfl_tag = (
                        f", CFL={key[3]}"
                        if (key[3] is not None and SHOW_CFL_IN_LEGEND)
                        else ""
                    )
                    lbl = rf"{key[0].upper()} (${n_label}$={key[1]}{cfl_tag})"
                    add_data(ax, macro, r, col, mk, ls, me, lbl)

                finish_panel(ax, macro)

            save_or_show(f"couette_profiles_Kn_{Kn}_Nc_{Nc}_T_{T_FINAL}.pdf")

    else:
        # Multi-Kn overlay: all Kn values in one figure
        nrows, ncols = resolve_layout(len(macros), LAYOUT)
        width, height = tp.get_figsize(fraction=1.0)
        fig, axes_arr = plt.subplots(
            nrows,
            ncols,
            figsize=(width * ncols, height * nrows * 1.15),
            squeeze=False,
        )
        ax_flat = axes_arr.flatten()
        for i in range(len(macros), nrows * ncols):
            ax_flat[i].set_visible(False)
        axes = ax_flat[: len(macros)]

        fillstyles = ["full", "none"]  # alternate by Kn index

        for kn_idx, Kn in enumerate(Kn_list):
            runs_this_Kn = _sort_runs(
                [(s, Nc, K, cfl) for (s, Nc, K, cfl) in data if abs(K - Kn) < 1e-9]
            )
            fs = fillstyles[kn_idx % 2]
            first_kn = kn_idx == 0
            rho_0_kn = get_rho_0(data[runs_this_Kn[0]]) if runs_this_Kn else 1.0

            for macro, ax in zip(macros, axes):
                if PLOT_CONTINUUM:
                    add_continuum(ax, macro, Kn, label_it=first_kn, rho_0=rho_0_kn)

                if PLOT_FREE_MOLECULAR:
                    add_free_molecular(ax, macro, label_it=first_kn, rho_0=rho_0_kn)

                r_ref = ref_data.get(Kn)
                if r_ref is not None:
                    add_reference(ax, macro, r_ref, REFERENCE[Kn])

                for idx, key in enumerate(runs_this_Kn):
                    r = data[key]
                    col, mk, ls = run_style(key[0], idx)
                    if key[3] is not None and CFL_OVERRIDES_MARKER:
                        mk = CFL_MARKER
                    me = max(1, len(r["x"]) // 12)
                    cfl_tag = (
                        f", CFL={key[3]}"
                        if (key[3] is not None and SHOW_CFL_IN_LEGEND)
                        else ""
                    )
                    lbl = rf"{key[0].upper()} ($\mathrm{{Kn}}={Kn:g}${cfl_tag})"
                    add_data(ax, macro, r, col, mk, ls, me, lbl, fillstyle=fs)

                finish_panel(ax, macro, show_legend=False)

        # Build structured legend: analytical lines, then one section per Kn
        handles: list = []
        labels: list = []

        if PLOT_CONTINUUM:
            handles.append(Line2D([], [], color="k", ls="-", lw=1.2))
            labels.append("Continuum")
        if PLOT_FREE_MOLECULAR:
            handles.append(Line2D([], [], color="k", ls="--", lw=1.0))
            labels.append(r"Free mol.")

        for kn_idx, Kn in enumerate(Kn_list):
            runs_this_Kn = _sort_runs(
                [(s, Nc, K, cfl) for (s, Nc, K, cfl) in data if abs(K - Kn) < 1e-9]
            )
            fs = fillstyles[kn_idx % 2]
            # Section header — invisible line, bold label
            handles.append(Line2D([], [], color="none"))
            labels.append(rf"$\mathbf{{Kn = {Kn:g}}}$")
            # Reference entry for this Kn (if any)
            ref_spec = REFERENCE.get(Kn)
            if ref_spec is not None and Kn in ref_data:
                n_label = "N_c" if ref_spec[0].lower() in ["ugks", "fvm"] else "N_x"
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="dimgrey",
                        marker="o",
                        ls="-",
                        ms=MARKER_SIZE,
                        lw=LINE_WIDTH,
                        fillstyle="none",
                    )
                )
                labels.append(
                    rf"Ref.: {ref_spec[0].upper()} (${n_label}$={ref_spec[1]})"
                )
            # Solver entries
            for idx, key in enumerate(runs_this_Kn):
                col, mk, ls_str = run_style(key[0], idx)
                if key[3] is not None and CFL_OVERRIDES_MARKER:
                    mk = CFL_MARKER
                n_label = "N_c" if key[0].lower() in ["ugks", "fvm"] else "N_x"
                cfl_tag = (
                    f", CFL={key[3]}"
                    if (key[3] is not None and SHOW_CFL_IN_LEGEND)
                    else ""
                )
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=col,
                        marker=mk,
                        ls=ls_str,
                        ms=MARKER_SIZE,
                        lw=LINE_WIDTH,
                        fillstyle=fs,
                    )
                )
                labels.append(rf"{key[0].upper()} (${n_label}$={key[1]}{cfl_tag})")

        fig.tight_layout(rect=[0, 0, 0.70, 1])
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.72, 0.5))

        Kn_str = "_".join(str(Kn) for Kn in Kn_list)
        save_or_show(f"couette_profiles_multiKn_{Kn_str}.pdf")


# # ═════════════════════════════════════════════════════════════════════════════
# # SHEAR STRESS VS Kn — log-log summary
# # ═════════════════════════════════════════════════════════════════════════════
# if PLOT_SHEAR_KN:
#     runs_by_series = defaultdict(list)
#     for (solver, Nc, Kn), r in data.items():
#         if "tau_xy" in r:
#             idx_mid = len(r["tau_xy"]) // 2
#             runs_by_series[(solver, Nc)].append((Kn, np.abs(r["tau_xy"][idx_mid])))

#     if runs_by_series:
#         width, height = tp.get_figsize(fraction=0.7)
#         fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

#         Kn_range = np.logspace(-2, 2, 200)
#         mu_nd = Kn_range * np.sqrt(2.0 / np.pi)
#         tau_fm = np.abs(couette_shear_stress_free_molecular(1.0, u_w_nd, R_nd, T_w_nd))

#         if PLOT_CONTINUUM:
#             ax.loglog(
#                 Kn_range,
#                 np.abs(couette_shear_stress_continuum(u_w_nd, mu_nd)),
#                 "k-",
#                 lw=1.5,
#                 label="Continuum",
#             )
#         if PLOT_FREE_MOLECULAR:
#             ax.loglog(
#                 Kn_range,
#                 tau_fm * np.ones_like(Kn_range),
#                 "k--",
#                 lw=1.5,
#                 label=r"Free mol.",
#             )

#         for idx, ((solver, Nc), pts) in enumerate(runs_by_series.items()):
#             pts = sorted(pts)
#             col, mk, ls = run_style(idx)

#             n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"

#             ax.loglog(
#                 [p[0] for p in pts],
#                 [p[1] for p in pts],
#                 ls,
#                 color=col,
#                 marker=mk,
#                 ms=4,
#                 lw=1.0,
#                 label=rf"{solver.upper()} ${n_label}$={Nc}",
#             )

#         ax.set_xlabel(r"$\mathrm{Kn}$")
#         ax.set_ylabel(r"$|\tau_{xy}|$ (non-dim.)")
#         ax.set_title(
#             rf"Couette flow — $|\tau_{{xy}}|$ vs $\mathrm{{Kn}}$, "
#             rf"$N_v={nv_tag}$, $\hat{{t}}={T_FINAL}$"
#         )
#         ax.legend()
#         ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
#         save_or_show("couette_shearstress_vs_Kn.pdf")

# print("\nDone.")

# ═════════════════════════════════════════════════════════════════════════════
# SHEAR STRESS VS Kn — log-log summary
# ═════════════════════════════════════════════════════════════════════════════
if PLOT_SHEAR_KN:
    runs_by_series = defaultdict(list)
    for (solver, Nc, Kn, CFL), r in data.items():
        if "tau_xy" in r:
            idx_mid = len(r["tau_xy"]) // 2
            if r["tau_xy"][idx_mid] < 0:
                print("  WARNING: tau_xy[mid] is negative; plotting absolute value")
            tau_nd = np.abs(r["tau_xy"][idx_mid])

            # Convert non-dimensional stress to actual dimensional stress (Pascals)
            rho_0 = get_rho_0(r)
            U0_sq = float(r.get("R_s", 208.13)) * float(r.get("T_w", 273.0))
            if "U0" in r:
                U0_sq = float(r["U0"]) ** 2
            p_0 = rho_0 * U0_sq

            runs_by_series[(solver, Nc, CFL)].append((Kn, tau_nd * p_0))

    if runs_by_series:
        width, height = tp.get_figsize(fraction=0.8)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

        Kn_range = np.logspace(-3, 2, 200)
        mu_nd = Kn_range * np.sqrt(2.0 / np.pi)
        tau_fm_nd = np.abs(
            couette_shear_stress_free_molecular(1.0, u_w_nd, R_nd, T_w_nd)
        )

        # Reconstruct the reference pressure for the theoretical lines over Kn_range
        mu_0 = float(_r0.get("mu_0", 2.117e-5))
        T_ref = float(_r0.get("T_w", 273.0))
        R_s_phys = float(_r0.get("R_s", 208.13))
        L = float(_r0.get("L", 1.0))
        Pr_Ar_ = float(_r0.get("Pr_Ar", 2.0 / 3.0))
        tau_mode = _r0.get("tau_mode", "viscosity")
        if isinstance(tau_mode, bytes):
            tau_mode = tau_mode.decode()
        mu_eff = mu_0 if tau_mode == "viscosity" else mu_0 / Pr_Ar_

        factor = np.sqrt(np.pi / (2.0 * R_s_phys * T_ref))
        rho_0_range = mu_eff * factor / (Kn_range * L)
        p0_range = rho_0_range * (R_s_phys * T_ref)

        if PLOT_CONTINUUM:
            tau_cont_nd_raw = couette_shear_stress_continuum(u_w_nd, mu_nd)
            if (tau_cont_nd_raw < 0).any():
                print("  WARNING: continuum tau_xy has negative values; plotting abs")
            tau_cont_nd = np.abs(tau_cont_nd_raw)
            ax.loglog(
                Kn_range,
                tau_cont_nd * p0_range,  # Converted to Pascals
                "k-",
                lw=1.5,
                label="Continuum",
            )
        if PLOT_FREE_MOLECULAR:
            ax.loglog(
                Kn_range,
                tau_fm_nd * p0_range,  # Converted to Pascals
                "k--",
                lw=1.5,
                label=r"Free mol.",
            )

        for idx, ((solver, Nc, CFL), pts) in enumerate(runs_by_series.items()):
            pts = sorted(pts)
            col, mk, ls = run_style(solver, idx)
            if CFL is not None and CFL_OVERRIDES_MARKER:
                mk = CFL_MARKER

            n_label = "N_c" if solver.lower() in ["ugks", "fvm"] else "N_x"
            cfl_tag = f", CFL={CFL}" if (CFL is not None and SHOW_CFL_IN_LEGEND) else ""

            ax.loglog(
                [p[0] for p in pts],
                [p[1] for p in pts],
                ls,
                color=col,
                marker=mk,
                ms=4,
                lw=1.0,
                label=rf"{solver.upper()} (${n_label}$={Nc}{cfl_tag})",
            )

        ax.set_xlabel(r"$\mathrm{Kn}$")
        ax.set_ylabel(r"$|\tau_{xy}|$ (Pa)")
        # ax.set_title(
        #     rf"Couette flow — $|\tau_{{xy}}|$ vs $\mathrm{{Kn}}$, "
        #     rf"$N_v={nv_tag}$, variable steady-state $\hat{{t}}$"
        # )
        ax.legend()
        ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
        save_or_show("couette_shearstress_vs_Kn_dimensional.pdf")

print("\nDone.")
