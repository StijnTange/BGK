import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import bgk.thesis_plots as tp

# =============================================================================
# USER SETTINGS
# =============================================================================

# Deterministic / UGKP: ("solver", Kn, Nx, Nv_or_NinjTot, CFL, vmax, t_final)
# Stochastic (RTSM/VJ): ("solver", Kn, Nx, N_INJ_TOTAL, dt, vmax, t_final)
PLOT_RUNS = [
    # dt = 0.1
    ("rtsm", 1e-3, 100, 1e5, 0.1, 10.0, 4.0),
    ("vj", 1e-3, 100, 1e5, 0.1, 10.0, 4.0),
    # dt. = 0.05
    ("rtsm", 1e-3, 100, 1e5, 0.05, 10.0, 4.0),
    ("vj", 1e-3, 100, 1e5, 0.05, 10.0, 4.0),
    # dt = 0.01
    ("rtsm", 1e-3, 100, 1e5, 0.01, 10.0, 4.0),
    ("vj", 1e-3, 100, 1e5, 0.01, 10.0, 4.0),
    # dt = 0.0025
    ("rtsm", 1e-3, 100, 1e5, 0.0025, 10.0, 4.0),
    ("vj", 1e-3, 100, 1e5, 0.0025, 10.0, 4.0),
    # dt = 0.0005
    ("rtsm", 1e-3, 100, 1e5, 0.0005, 10.0, 4.0),
    ("vj", 1e-3, 100, 1e5, 0.0005, 10.0, 4.0),
    # dt = 0.0001
    ("rtsm", 1e-3, 100, 1e5, 0.0001, 10.0, 4.0),
    ("vj", 1e-3, 100, 1e5, 0.0001, 10.0, 4.0),
]


Nx_ref = 4000
Nv_ref = 60
dx_ref = (1.0 - 0.0) / Nx_ref
v_max = 10.0
CFL_ref = 0.9
dt_ref = CFL_ref * dx_ref / v_max
t_final = 4.0
REFERENCE = ("ugks", 1e-3, Nx_ref, Nv_ref, dt_ref, v_max, t_final)


# Macroscopic variable ("rho", "u", "T", "q")
MACRO_FOR_ERROR = "rho"

# Zoom limits for the spatial profile (Set to None to show full domain)
SPATIAL_X_LIMITS = (0.0, 0.2)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "simulations_wpd")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SAVE_PLOTS = False

# =============================================================================
# HELPERS
# =============================================================================

DET_SOLVERS = {"ugks", "fvm", "sl", "strang"}
STOCH_DT_SOLVERS = {"rtsm", "vj"}
STOCH_CFL_SOLVERS = {"ugkp"}


def _fmt_sci(val):
    if val is None or str(val) == "None":
        return "None"
    return f"{float(val):.1e}".replace("-0", "-", 1).replace("+0", "").replace("+", "")


def make_filename(run_tuple, is_ref=False):
    solver = run_tuple[0]
    Kn = run_tuple[1]
    Nx = run_tuple[2]

    if solver in STOCH_DT_SOLVERS:
        _, _, _, N_INJ_TOTAL, dt, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_NinjTot{_fmt_sci(N_INJ_TOTAL)}_dt{_fmt_sci(dt)}_T{t_final}.h5"
    elif solver in STOCH_CFL_SOLVERS:
        _, _, _, N_INJ_TOTAL, CFL, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_NinjTot{_fmt_sci(N_INJ_TOTAL)}_CFL{CFL}_T{t_final}.h5"
    else:
        _, _, _, Nv, CFL, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_Nv{Nv}_CFL{CFL}_T{t_final}.h5"

    if is_ref:
        fname = fname.replace(f"{solver}_", f"{solver}_ref_")
    return fname


def load_sim(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: file not found: {path}")
        return None
    with h5py.File(path, "r") as f:
        return {
            "x": f["x"][:],
            "rho": f["rho"][:],
            "u": f["u"][:],
            "T": f["T"][:],
            "q": f["q"][:],
            "cpu_time": float(f["cpu_time"][0]),
            "meta": dict(f.attrs),
        }


# =============================================================================
# LOAD AND COMPUTE ERRORS
# =============================================================================

print("Loading Reference Solution...")
ref_data = load_sim(make_filename(REFERENCE, is_ref=True))
if ref_data is None:
    raise FileNotFoundError("Reference simulation not found. Please generate it first.")

ref_x = ref_data["x"]
ref_macro = ref_data[MACRO_FOR_ERROR].flatten()
interp_func = interp1d(ref_x, ref_macro, kind="cubic", fill_value="extrapolate")

solver_data = {}

for run in PLOT_RUNS:
    r = load_sim(make_filename(run))
    if r is None:
        continue

    solver_name = run[0].upper()
    cpu = r["cpu_time"]

    sim_x = r["x"]
    sim_macro = r[MACRO_FOR_ERROR].flatten()
    ref_macro_mapped = interp_func(sim_x)

    err_l2 = np.linalg.norm(sim_macro - ref_macro_mapped) / np.linalg.norm(
        ref_macro_mapped
    )

    # -------------------------------------------------------------
    # FORMAT THE DISPLAY LABEL BASED ON SOLVER TYPE
    # -------------------------------------------------------------
    if run[0] in DET_SOLVERS:
        # Deterministic: Plot CFL directly
        plot_sort_val = run[4]
        label_str = f"CFL={plot_sort_val}"
    elif run[0] in STOCH_DT_SOLVERS:
        # RTSM/VJ: Plot dt directly
        plot_sort_val = run[4]
        label_str = f"$\Delta t$={plot_sort_val}"
    else:  # UGKP
        # UGKP: Tuple defines CFL, but we want to plot dt
        cfl = run[4]
        dx = 3.0 / run[2]  # 3.0 is the domain length xR - xL
        vmax = run[5]
        dt = cfl * dx / vmax
        plot_sort_val = cfl  # Use CFL to sort line thickness mapping below
        # Format the derived dt neatly for the text label
        label_str = f"$\Delta t$={dt:.1e}".replace("-0", "-")

    if solver_name not in solver_data:
        solver_data[solver_name] = {
            "cpu": [],
            "err": [],
            "val": [],
            "lbl": [],
            "x": [],
            "macro": [],
        }

    solver_data[solver_name]["cpu"].append(cpu)
    solver_data[solver_name]["err"].append(err_l2)
    solver_data[solver_name]["val"].append(plot_sort_val)
    solver_data[solver_name]["lbl"].append(label_str)
    solver_data[solver_name]["x"].append(sim_x)
    solver_data[solver_name]["macro"].append(sim_macro)

# Sort each solver's lists by CPU time to ensure lines draw cleanly
for s in solver_data:
    sort_idx = np.argsort(solver_data[s]["cpu"])
    solver_data[s]["cpu"] = np.array(solver_data[s]["cpu"])[sort_idx]
    solver_data[s]["err"] = np.array(solver_data[s]["err"])[sort_idx]
    solver_data[s]["val"] = np.array(solver_data[s]["val"])[sort_idx]
    solver_data[s]["lbl"] = np.array(solver_data[s]["lbl"])[sort_idx]
    solver_data[s]["x"] = [solver_data[s]["x"][i] for i in sort_idx]
    solver_data[s]["macro"] = [solver_data[s]["macro"][i] for i in sort_idx]

# =============================================================================
# PLOT
# =============================================================================

_COLORS = {
    "UGKP": "#4daf4a",
    "RTSM": "#e41a1c",
    "VJ": "#377eb8",
    "UGKS": "#984ea3",
    "STRANG": "#ff7f00",
    "FVM": "#a65628",
    "SL": "#f781bf",
}
_MARKERS = {
    "UGKP": "^",
    "RTSM": "o",
    "VJ": "s",
    "UGKS": "D",
    "STRANG": "p",
    "FVM": "h",
    "SL": "v",
}

width, _ = tp.get_figsize(fraction=1.0)
fig, (ax_wpd, ax_prof) = plt.subplots(
    1, 2, figsize=(width, width * 0.45), constrained_layout=True
)

# -----------------------------------------------------------------------------
# PLOT 1: WORK-PRECISION DIAGRAM
# -----------------------------------------------------------------------------
for solver, metrics in solver_data.items():
    color = _COLORS.get(solver, "black")
    marker = _MARKERS.get(solver, "o")

    ax_wpd.plot(
        metrics["cpu"],
        metrics["err"],
        color=color,
        marker=marker,
        linestyle="-",
        linewidth=2.0,
        markersize=7,
        label=solver,
    )

    for i in range(len(metrics["cpu"])):
        ax_wpd.text(
            metrics["cpu"][i] * 1.1,
            metrics["err"][i],
            metrics["lbl"][i],
            fontsize=8,
            color=color,
            verticalalignment="center",
        )

ax_wpd.set_xscale("log")
ax_wpd.set_yscale("log")
ax_wpd.set_xlabel("CPU Time (seconds)")
ax_wpd.set_ylabel(f"Relative $L_2$ Error ($\\{MACRO_FOR_ERROR}$)")
ax_wpd.set_title("Work-Precision Diagram")
ax_wpd.grid(True, which="major", linestyle="-", alpha=0.5)
ax_wpd.grid(True, which="minor", linestyle="--", alpha=0.2)
ax_wpd.legend(frameon=True, facecolor="white", edgecolor="black")

# -----------------------------------------------------------------------------
# PLOT 2: SPATIAL PROFILE
# -----------------------------------------------------------------------------
ax_prof.plot(ref_x, ref_macro, "k-", linewidth=2.0, zorder=5, label="Reference")

for solver, metrics in solver_data.items():
    color = _COLORS.get(solver, "black")
    marker = _MARKERS.get(solver, "o")

    for i in range(len(metrics["val"])):
        val = metrics["val"][i]
        lbl = metrics["lbl"][i]
        x_arr = metrics["x"][i]
        m_arr = metrics["macro"][i]

        # Style mapping: coarse (large val) = dotted, fine (small val) = solid
        # Note: Depending on your exact dt/CFL values, you may need to tweak these thresholds!
        if val >= 0.9 or val >= 0.03:
            ls = ":"
        elif val >= 0.5 or val >= 0.01:
            ls = "--"
        else:
            ls = "-"

        me = max(1, len(x_arr) // 10)
        ax_prof.plot(
            x_arr,
            m_arr,
            color=color,
            linestyle=ls,
            linewidth=1.2,
            marker=marker,
            ms=4,
            markevery=me,
            alpha=0.8,
            label=f"{solver} ({lbl})",
            zorder=3,
        )

if SPATIAL_X_LIMITS is not None:
    ax_prof.set_xlim(SPATIAL_X_LIMITS)

ax_prof.set_xlabel(r"Position $x$")
ax_prof.set_ylabel(f"Macro: $\\{MACRO_FOR_ERROR}$")
ax_prof.set_title(f"Spatial Distribution at t={PLOT_RUNS[0][6]}")
ax_prof.grid(True, which="both", linestyle="--", alpha=0.4)

ax_prof.legend(
    fontsize="x-small", ncol=2, frameon=True, facecolor="white", edgecolor="black"
)

_outname = f"wpd_hybrid_{MACRO_FOR_ERROR}_Kn{_fmt_sci(PLOT_RUNS[0][1])}.pdf"
if SAVE_PLOTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, _outname)
    tp.save_plot(path)
    print(f"Saved: {path}")
else:
    plt.show()
