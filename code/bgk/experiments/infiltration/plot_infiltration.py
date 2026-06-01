import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import bgk.thesis_plots as tp

# =============================================================================
# USER SETTINGS
# =============================================================================

# Stochastic:   ("solver", Kn, Nx, Np, N_INJ_TOTAL, CFL, vmax, t_final)
# Deterministic: ("solver", Kn, Nx, Nv, CFL, vmax, t_final)
PLOT_RUNS = [
    ("hybrid", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
    ("ugks", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
    ("strang", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
    ("sl", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
    ("fvm", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
    # ("ugkp", 5e-1, 30, None, int(1e5), 0.9, 20.0, 10.0),
    # ("rtsm", 5e-1, 30, None, int(1e5), 0.9, 20.0, 10.0),
    # ("vj", 5e-1, 30, None, int(1e5), 0.9, 20.0, 10.0),
]

# Reference solution tuple, or None
REFERENCE = None
#     "fvm",
#     5e-1,
#     1000,
#     80,
#     0.9,
#     20.0,
#     10.0,
# )  # <-- Set to None to disable reference

# Main domain view boundaries (Set to None to show the whole domain)
MAIN_X_LIMITS = (0.0, 1.0)

# --- NEW: PLOT SELECTION & LAYOUT ---
# Choose from: "rho", "u", "T", "q"
MACROS_TO_PLOT = ["rho", "u", "T", "q"]

# Define the grid layout (rows, columns). E.g., (2, 1) for vertical, (1, 2) for horizontal
PLOT_LAYOUT = (2, 2)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "simulations")
OUTPUT_DIR = "latex/thesis/figures/infiltration"
SAVE_PLOTS = True

# --- LEGEND OPTIONS ---
SHOW_NX_IN_LEGEND = True  # show grid size (Nc/Nx) in legend
SHOW_CFL_IN_LEGEND = False  # show CFL number in legend
SHOW_NINJ_IN_LEGEND = False  # show N_inj in legend (stochastic solvers only)
SHOW_RMSE_IN_LEGEND = True  # show RMSE vs reference per subplot (requires REFERENCE)

# (x, y) in axes coordinates (0-1). Default is top-right (0.97, 0.97).
RMSE_BOX_POS = {
    "rho": (0.97, 0.16),  # density: lower-right, raised a bit
    "u": (0.97, 0.03),  # velocity: bottom-right
    "T": (0.97, 0.97),
    "q": (0.97, 0.97),
}

# --- PLOT ORDER ---
# Solvers are plotted in this order (others follow in PLOT_RUNS order).
# Set to None to use PLOT_RUNS order as-is.
PLOT_ORDER = ["hybrid", "strang", "sl", "fvm", "ugks", "rtsm", "vj", "ugkp"]

# =============================================================================
# FILENAME HELPERS  (Must identically match run_infiltration.py)
# =============================================================================

STOCHASTIC_SOLVERS = {"rtsm", "ugkp", "vj"}
CELL_BASED_SOLVERS = {
    "strang",
    "sl",
    "hybrid",
    "fvm",
    "ugks",
    "ugkp",
}


def compute_noise_metrics(test_x, test_y, ref_x, ref_y):
    # Interpolate the fine reference onto the coarse test grid
    ref_interp = np.interp(test_x, ref_x, ref_y)

    # The noise is the residual
    residuals = test_y - ref_interp

    # 1. Root Mean Square Error (L2)
    rmse = np.sqrt(np.mean(residuals**2))

    # 2. Max Error (L_inf)
    max_err = np.max(np.abs(residuals))

    # 3. Total Variation (TV)
    tv_test = np.sum(np.abs(np.diff(test_y)))
    tv_ref = np.sum(np.abs(np.diff(ref_interp)))
    tv_ratio = tv_test / tv_ref if tv_ref > 1e-12 else 0.0

    return rmse, max_err, tv_ratio


def _fmt_sci(val):
    if val is None or str(val) == "None":
        return "None"
    return f"{float(val):.1e}".replace("-0", "-").replace("+0", "").replace("+", "")


def make_filename(run_tuple, is_ref=False):
    solver = run_tuple[0]
    Kn = run_tuple[1]
    Nx = run_tuple[2]

    if solver in STOCHASTIC_SOLVERS:
        _, _, _, Np, N_INJ_TOTAL, CFL, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_Np{_fmt_sci(Np)}_NinjTot{_fmt_sci(N_INJ_TOTAL)}_CFL{CFL}_vmax{vmax}_T{t_final}.h5"
    else:
        _, _, _, Nv, CFL, vmax, t_final = run_tuple
        fname = (
            f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_Nv{Nv}_CFL{CFL}_vmax{vmax}_T{t_final}.h5"
        )
        if is_ref:
            fname = fname.replace("ugks_", "ugks_ref_")
    return fname


# =============================================================================
# LOAD
# =============================================================================


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


def load_run(run):
    fname = make_filename(run)
    return load_sim(fname)


def load_reference():
    if REFERENCE is None:
        return None
    fname = make_filename(REFERENCE, is_ref=True)
    return load_sim(fname)


# =============================================================================
# STYLE & LABELS
# =============================================================================

_SOLVER_STYLES = {
    "strang": ("#377eb8", "o", "-"),  # blue circle
    "sl": ("#e41a1c", "^", "-"),  # red triangle up
    "fvm": ("#4daf4a", "s", "-"),  # green square
    "ugks": ("#ff7f00", "D", "-"),  # orange diamond
    "hybrid": ("#984ea3", "v", "-"),  # purple triangle down
    "rtsm": ("#377eb8", "o", "-"),  # blue circle
    "vj": ("#e41a1c", "^", "-"),  # red triangle up
    "ugkp": ("#4daf4a", "s", "-"),  # green square
}
_FALLBACK_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
_FALLBACK_MARKERS = ["o", "s", "^", "D", "v"]


def run_style(run):
    solver = run[0].lower()
    if solver in _SOLVER_STYLES:
        return _SOLVER_STYLES[solver]
    idx = hash(solver) % len(_FALLBACK_COLORS)
    return (_FALLBACK_COLORS[idx], _FALLBACK_MARKERS[idx], "-")


def make_label(run, r):
    solver = run[0]
    solver_upper = solver.upper()
    cpu = r["cpu_time"]

    Nx = run[2]
    N_symbol = "N_c" if solver.lower() in CELL_BASED_SOLVERS else "N_x"

    parts = [f"{cpu:.1f}s"]
    if SHOW_NX_IN_LEGEND:
        parts.append(f"${N_symbol}$={Nx}")

    if solver.lower() in STOCHASTIC_SOLVERS:
        N_INJ_TOTAL = run[4]
        CFL = run[5]
        if SHOW_CFL_IN_LEGEND:
            parts.append(f"CFL={CFL}")
        if SHOW_NINJ_IN_LEGEND:
            parts.append(f"$N_{{inj}}$={_fmt_sci(N_INJ_TOTAL)}")
    else:
        CFL = run[4]
        if SHOW_CFL_IN_LEGEND:
            parts.append(f"CFL={CFL}")

    return rf"{solver_upper} ({', '.join(parts)})"


def save_or_show(filename):
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        tp.save_plot(path)
        print(f"Saved: {path}")
    else:
        plt.show()


# --- UPDATED: Heuristic Background Profiles ---
def bg_arrays(x):
    xR = x[-1]
    x_norm = x / xR

    # 1. Background Density (Exponential + Gaussian Ionization Spike)
    n0 = 1.0
    alpha = 2.0
    base_density = n0 * np.exp(alpha * x_norm)
    peak_height = 25.0
    x_center = 0.985
    peak_width = 0.015
    rho_bg = base_density + peak_height * np.exp(
        -0.5 * ((x_norm - x_center) / peak_width) ** 2
    )

    # 2. Background Velocity (Concave Bow)
    u_left = -1.0
    u_right = 1.0
    k = 1.0
    u_bg = (u_left + (u_right - u_left) * x_norm) + k * x_norm * (1.0 - x_norm)

    # 3. Background Temperature (Convex Parabola)
    T_left = 0.1
    T_right = 10.0
    T_bg = T_left + (T_right - T_left) * (x_norm**2)

    return rho_bg, u_bg, T_bg


# =============================================================================
# LOAD ALL DATA
# =============================================================================

data = {}
for run in PLOT_RUNS:
    r = load_run(run)
    if r is not None:
        data[run] = r

# Reorder according to PLOT_ORDER if set
if PLOT_ORDER is not None:

    def _order_key(item):
        solver = item[0][0].lower()
        return PLOT_ORDER.index(solver) if solver in PLOT_ORDER else len(PLOT_ORDER)

    data = dict(sorted(data.items(), key=_order_key))

ref_data = load_reference()

if not data:
    print("No results loaded. Run run_infiltration.py first.")
    sys.exit(0)

_x_bg = ref_data["x"] if ref_data is not None else next(iter(data.values()))["x"]
# Unpack all 3 arrays now
rho_bg_arr, u_bg_arr, T_bg_arr = bg_arrays(_x_bg)

# =============================================================================
# PLOT
# =============================================================================

# Map macro keys to their labels and background data
# Note: Plasma density (rho_bg_arr) is set to None so it doesn't squash the Neutral Density scale.
MACRO_CONFIG_MAP = {
    "rho": (r"$\rho$", None),
    "u": (r"$u$", u_bg_arr),
    "T": (r"$T$", T_bg_arr),
    "q": (r"$q$", None),
}

nrows, ncols = PLOT_LAYOUT
if len(MACROS_TO_PLOT) > nrows * ncols:
    raise ValueError("PLOT_LAYOUT is too small for the number of MACROS_TO_PLOT.")

width, _ = tp.get_figsize(fraction=1.0)
# Scale height dynamically based on the number of rows
fig, axs = plt.subplots(
    nrows, ncols, figsize=(width, width * 0.4 * nrows), constrained_layout=True
)

# Flatten axs to make iteration easier
if nrows * ncols == 1:
    axs_flat = [axs]
else:
    axs_flat = axs.flatten()

for idx, macro in enumerate(MACROS_TO_PLOT):
    if macro not in MACRO_CONFIG_MAP:
        print(f"Warning: Macro '{macro}' not found in configuration map.")
        continue

    ax = axs_flat[idx]
    ylabel, bg_data = MACRO_CONFIG_MAP[macro]

    # Reference
    if ref_data is not None:
        ref_solver = REFERENCE[0].upper()
        ref_Nx = REFERENCE[2]
        ref_N_symbol = "N_c" if REFERENCE[0].lower() in CELL_BASED_SOLVERS else "N_x"
        ref_label = rf"Ref.: {ref_solver} (${ref_N_symbol}$ = {ref_Nx})"
        ax.plot(
            ref_data["x"],
            ref_data[macro].flatten(),
            "k--",
            linewidth=1.5,
            zorder=2,
            label=ref_label,
        )

    # Background
    if bg_data is not None:
        ax.plot(
            _x_bg,
            bg_data,
            color="gray",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            zorder=1,
            label="Background",
        )

    # Simulation runs — stagger markers so no two solvers overlap
    n_runs = len(data)
    min_npts = min(len(r["x"]) for r in data.values())
    me = max(n_runs, min_npts // 8)
    rmse_lines = []
    print(f"\n--- Noise Metrics for {macro.upper()} ---")
    for run_idx, (run, r) in enumerate(data.items()):
        col_c, mk, ls = run_style(run)
        lbl = make_label(run, r)
        offset = run_idx * me // n_runs

        test_x = r["x"]
        test_y = r[macro].flatten()

        if ref_data is not None:
            ref_x = ref_data["x"]
            ref_y = ref_data[macro].flatten()
            rmse, max_err, tv_ratio = compute_noise_metrics(
                test_x, test_y, ref_x, ref_y
            )
            print(
                f"{run[0].upper():<8} | RMSE: {rmse:.4f}"
                f" | Max Err: {max_err:.4f} | TV Ratio: {tv_ratio:.2f}"
            )
            rmse_lines.append((run[0].upper(), rmse))

        ax.plot(
            r["x"],
            r[macro].flatten(),
            ls,
            color=col_c,
            marker=mk,
            ms=4,
            markevery=(offset, me),
            linewidth=1.0,
            label=lbl,
            zorder=3,
        )

    # RMSE text box
    if SHOW_RMSE_IN_LEGEND and rmse_lines:
        box_text = "\\textbf{RMSE}\n" + "\n".join(
            rf"{name}: {rmse:.3f}" for name, rmse in rmse_lines
        )
        bx, by = RMSE_BOX_POS.get(macro, (0.97, 0.97))
        va = "bottom" if by < 0.5 else "top"
        ax.text(
            bx,
            by,
            box_text,
            transform=ax.transAxes,
            fontsize=plt.rcParams["legend.fontsize"],
            verticalalignment=va,
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
            ),
        )

    # Main Axes limits
    if MAIN_X_LIMITS is not None:
        ax.set_xlim(MAIN_X_LIMITS)

    # if macro == "q":
    #     ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$x$")
    ax.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)

# Hide any unused subplots
for idx in range(len(MACROS_TO_PLOT), len(axs_flat)):
    axs_flat[idx].set_visible(False)

# =============================================================================
# UNIFIED LEGEND
# =============================================================================
# 1. Collect all unique handles and labels across every subplot
handles, labels = [], []
for ax in axs_flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

# 2. Draw the unified legend on the first subplot (or use fig.legend)
axs_flat[0].legend(
    handles,
    labels,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    loc="best",  # Matplotlib will try to find a spot that doesn't overlap data
)

_solver_str = "_".join(run[0] for run in data)
_macro_str = "".join(MACROS_TO_PLOT)
_outname = f"infiltration_{_macro_str}_{_solver_str}_Kn{_fmt_sci(PLOT_RUNS[0][1])}_Nc{PLOT_RUNS[0][2]}.pdf"
save_or_show(_outname)
print("\nDone.")
