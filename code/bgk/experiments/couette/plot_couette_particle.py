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

# Each entry:
#   (solver_name, Nc, Kn)           — uses global NVX, NVY
#   (solver_name, Nc, Kn, Nvx, Nvy) — per-run Nv override (use for particle solvers)
# Example for mixing deterministic and particle results:
#   ("ugks",  50, 0.1, 50, 50),   # deterministic
#   ("rtsm",  50, 0.1,  1, None), # particle (saved with Nvx=1, Nvy=None)
#   ("ugkp",  50, 0.1,  1, None), # particle


PLOT_RUNS = [
    ("rtsm", 40, 0.01, 1, None),
    ("ugkp", 40, 0.01, 1, None),
]

T_FINAL = 100.0
NVX = 28
NVY = 28  # set to None for dim_v=1

REFERENCE = ("ugks", 40, 0.01)  # (solver_name, Nc, Kn) or None for no reference
REF_T_FINAL = 100.0
REF_NVX = 28
REF_NVY = 28

RESULTS_DIR = "code/bgk/experiments/couette/simulations"
OUTPUT_DIR = "code/bgk/experiments/couette/plots"

SAVE_PLOTS = True  # True → save PDFs,  False → show interactively

# ── What to plot ──────────────────────────────────────────────────────────────
# Any subset/ordering of: "velocity", "temperature", "density", "shear_stress"
PLOT_MACROS = ["velocity", "temperature", "density", "shear_stress"]

# Panel grid: "auto" | "1x1" | "1x2" | "1x3" | "1x4" | "2x2"
LAYOUT = "auto"

# ── Reference limits ──────────────────────────────────────────────────────────
PLOT_CONTINUUM = True  # NSF continuum curves
PLOT_FREE_MOLECULAR = True  # free-molecular horizontal lines

# ── Extra figures ─────────────────────────────────────────────────────────────
PLOT_SHEAR_KN = True  # tau_xy vs Kn log-log summary

# ── Physical constants (must match run_couette.py) ────────────────────────────
Pr_BGK = 1.0
Pr_Ar = 2.0 / 3.0

# ═════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ═════════════════════════════════════════════════════════════════════════════
_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#ff7f00",
    "#984ea3",
    "#a65628",
    "#f781bf",
    "#999999",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def run_style(idx):
    return (
        _COLORS[idx % len(_COLORS)],
        _MARKERS[idx % len(_MARKERS)],
        _LINES[idx % len(_LINES)],
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
for run in PLOT_RUNS:
    solver, Nc, Kn = run[0], run[1], run[2]
    nvx = run[3] if len(run) > 3 else NVX
    nvy = run[4] if len(run) > 4 else NVY
    r = load_result(solver, Nc, nvx, nvy, Kn, T_FINAL, "couette", RESULTS_DIR)
    if r is not None:
        data[(solver, Nc, Kn)] = r

ref_data = None
if REFERENCE is not None:
    ref_data = load_result(
        REFERENCE[0],
        REFERENCE[1],
        REF_NVX,
        REF_NVY,
        REFERENCE[2],
        REF_T_FINAL,
        "couette",
        RESULTS_DIR,
    )
    if ref_data is None:
        print(f"  WARNING: reference {REFERENCE} not found")

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
nv_tag = f"{NVX}x{NVY}" if NVY is not None else str(NVX)


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
Kn_list = sorted({Kn for (_, _, Kn) in data})

# ═════════════════════════════════════════════════════════════════════════════
# PANEL HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def add_continuum(ax, macro, Kn):
    """Overlay NSF continuum limit curves."""
    mu_0_nd = Kn * np.sqrt(2.0 / np.pi)
    if macro == "velocity":
        ax.plot(
            x_ref,
            couette_velocity_continuum(x_ref, u_w_nd),
            "k--",
            lw=1.5,
            label="Continuum",
            zorder=5,
        )

    elif macro == "temperature":
        T_bgk = couette_temperature_continuum(x_ref, T_w_nd, u_w_nd, Pr_BGK, cp_nd)
        T_phys = couette_temperature_continuum(x_ref, T_w_nd, u_w_nd, Pr_Ar, cp_nd)
        ax.plot(
            x_ref, T_bgk, "k--", lw=1.5, label=r"Continuum ($\mathrm{Pr}=1$)", zorder=5
        )
        # ax.plot(
        #     x_ref,
        #     T_phys,
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

    elif macro == "shear_stress":
        tau_cont = couette_shear_stress_continuum(u_w_nd, mu_0_nd)
        ax.axhline(
            np.abs(tau_cont),
            color="k",
            lw=1.5,
            ls="--",
            label="Continuum",
            zorder=5,
        )


def add_free_molecular(ax, macro):
    """Overlay free-molecular limit."""
    if macro == "velocity":
        u_fm = couette_velocity_free_molecular(u_w_nd)
        ax.axhline(u_fm, color="gray", lw=1.0, ls="--", label=r"Free mol.", zorder=4)

    elif macro == "temperature":
        T_fm = couette_temperature_free_molecular(T_w_nd, u_w_nd, R_nd, n_v)
        ax.axhline(
            T_fm,
            color="gray",
            lw=1.0,
            ls="--",
            label=r"Free mol.",
            zorder=4,
        )

    elif macro == "density":
        # Free-molecular limit: both walls at T_w, no collisions → no viscous
        # heating → uniform density = mean density = rho_nd = 1 everywhere.
        # Physical value: rho_fm = 1 * rho_0. Drawn in add_data once rho_0
        # is known; stash the non-dim value here.
        ax._density_fm_nd = 1.0

    elif macro == "shear_stress":
        tau_fm = couette_shear_stress_free_molecular(1.0, u_w_nd, R_nd, T_w_nd)
        ax.axhline(
            np.abs(tau_fm),
            color="gray",
            lw=1.2,
            ls="--",
            label=r"Free mol. (eq.~B.37)",
            zorder=4,
        )


def add_reference(ax, macro, r):
    """Overlay the reference solution as a thick black line."""
    kw = dict(
        color="k",
        lw=2.0,
        zorder=8,
        label=rf"Ref.: {REFERENCE[0].upper()} $N_c$={REFERENCE[1]}",
    )
    if macro == "velocity" and "uy" in r:
        ax.plot(r["x"], r["uy"], **kw)
    elif macro == "temperature" and "T" in r:
        ax.plot(r["x"], r["T"], **kw)
    elif macro == "density" and "rho" in r:
        ax.plot(r["x"], r["rho"], **kw)
    elif macro == "shear_stress" and "tau_xy" in r:
        ax.plot(r["x"], np.abs(r["tau_xy"]), **kw)


def add_data(ax, macro, r, col, mk, ls, me, lbl):
    """Plot one simulation run and, for density, the deferred continuum line."""
    kw = dict(color=col, marker=mk, ms=4, lw=1.2, markevery=me, label=lbl)

    if macro == "velocity" and "uy" in r:
        ax.plot(r["x"], r["uy"], ls, **kw)

    elif macro == "temperature" and "T" in r:
        ax.plot(r["x"], r["T"], ls, **kw)

    elif macro == "density" and "rho" in r:
        rho_0 = get_rho_0(r)
        # Draw continuum and free-molecular lines the first time we know rho_0
        if not getattr(ax, "_density_ref_drawn", False):
            if PLOT_CONTINUUM and hasattr(ax, "_density_cont_nd"):
                ax.plot(
                    x_ref,
                    ax._density_cont_nd,
                    "k--",
                    lw=1.5,
                    label=r"Continuum",
                    zorder=5,
                )
            if PLOT_FREE_MOLECULAR and hasattr(ax, "_density_fm_nd"):
                ax.axhline(
                    ax._density_fm_nd,
                    color="gray",
                    lw=1.0,
                    ls="--",
                    label=r"Free mol.",
                    zorder=4,
                )
            ax._density_ref_drawn = True
        ax.plot(r["x"], r["rho"], ls, **kw)

    elif macro == "shear_stress" and "tau_xy" in r:
        ax.plot(r["x"], np.abs(r["tau_xy"]), ls, **kw)


def finish_panel(ax, macro):
    """Axis labels and decorations after all data is drawn."""
    ax.set_xlabel(r"$x / L$")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=7)
    if macro == "velocity":
        ax.set_ylabel(r"$u_y / U_0$")
        ax.axhline(0.0, color="silver", lw=0.6, ls=":")
        ax.axhline(u_w_nd, color="silver", lw=0.6, ls=":")
    elif macro == "temperature":
        ax.set_ylabel(r"$T / T_w$")
    elif macro == "density":
        ax.set_ylabel(r"$\rho/\rho_0$")
    elif macro == "shear_stress":
        ax.set_ylabel(r"$|\tau_{xy}|$ (non-dim.)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FIGURES — one per Kn
# ═════════════════════════════════════════════════════════════════════════════
VALID = ("velocity", "temperature", "density", "shear_stress")
macros = [m for m in PLOT_MACROS if m in VALID]
if not macros:
    print(f"No valid macros. Choose from: {VALID}")
    sys.exit(1)

for Kn in Kn_list:
    runs_this_Kn = [(s, Nc, K) for (s, Nc, K) in data if abs(K - Kn) < 1e-9]

    fig, axes = make_axes(len(macros), LAYOUT)

    for panel_i, (macro, ax) in enumerate(zip(macros, axes)):
        if PLOT_CONTINUUM:
            add_continuum(ax, macro, Kn)

        if PLOT_FREE_MOLECULAR:
            add_free_molecular(ax, macro)

        if ref_data is not None and abs(float(ref_data.get("Kn", -1)) - Kn) < 1e-8:
            add_reference(ax, macro, ref_data)

        for idx, key in enumerate(runs_this_Kn):
            r = data[key]
            col, mk, ls = run_style(idx)
            me = max(1, len(r["x"]) // 12)
            lbl = rf"{key[0].upper()} $N_c$={key[1]}"
            add_data(ax, macro, r, col, mk, ls, me, lbl)

        finish_panel(ax, macro)
        # ax.set_title(f"({chr(ord('a') + panel_i)})", loc="left", fontsize=9, pad=3)

    fig.suptitle(rf"Couette flow — $\mathrm{{Kn}}={Kn}$, $\hat{{t}}={T_FINAL}$")
    save_or_show(f"couette_profiles_Kn{Kn}.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# SHEAR STRESS VS Kn — log-log summary
# ═════════════════════════════════════════════════════════════════════════════
if PLOT_SHEAR_KN:
    runs_by_series = defaultdict(list)
    for (solver, Nc, Kn), r in data.items():
        if "tau_xy" in r:
            idx_mid = len(r["tau_xy"]) // 2
            runs_by_series[(solver, Nc)].append((Kn, np.abs(r["tau_xy"][idx_mid])))

    if runs_by_series:
        width, height = tp.get_figsize(fraction=0.7)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

        Kn_range = np.logspace(-2, 1.5, 200)
        mu_nd = Kn_range * np.sqrt(2.0 / np.pi)
        tau_fm = np.abs(couette_shear_stress_free_molecular(1.0, u_w_nd, R_nd, T_w_nd))

        if PLOT_CONTINUUM:
            ax.loglog(
                Kn_range,
                np.abs(couette_shear_stress_continuum(u_w_nd, mu_nd)),
                "k-",
                lw=1.5,
                label="Continuum",
            )
        if PLOT_FREE_MOLECULAR:
            ax.loglog(
                Kn_range,
                tau_fm * np.ones_like(Kn_range),
                "k--",
                lw=1.5,
                label=r"Free mol.",
            )

        for idx, ((solver, Nc), pts) in enumerate(runs_by_series.items()):
            pts = sorted(pts)
            col, mk, ls = run_style(idx)
            ax.loglog(
                [p[0] for p in pts],
                [p[1] for p in pts],
                ls,
                color=col,
                marker=mk,
                ms=4,
                lw=1.0,
                label=rf"{solver.upper()} $N_c$={Nc}",
            )

        ax.set_xlabel(r"$\mathrm{Kn}$")
        ax.set_ylabel(r"$|\tau_{xy}|$ (non-dim.)")
        ax.set_title(
            rf"Couette flow — $|\tau_{{xy}}|$ vs $\mathrm{{Kn}}$, "
            rf"$N_v={nv_tag}$, $\hat{{t}}={T_FINAL}$"
        )
        ax.legend()
        ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
        save_or_show("couette_shearstress_vs_Kn.pdf")

print("\nDone.")
