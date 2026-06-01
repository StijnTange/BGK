"""
Infiltration (Neutral Penetration) — Transient Reference Script.

Plots snapshots of density, velocity, and temperature profiles over time.
"""

import math
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Adjust the path as needed to run from your scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

SOLVERS = [
    "ugks",
]  # Options: "ugks", "fvm", "sl", "strang", "hybrid"
Kn = 0.05
T_FINAL = 20.0
Nc = 40
Nv = 80
CFL = 0.9
N_SNAPSHOTS = 8
VMAX = 20.0

# ═════════════════════════════════════════════════════════════════════════════
# INFILTRATION PROBLEM CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
xL, xR = 0.0, 1.0
R_gas = 1.0
REFLECTANCE_LEFT = 0.9  # 1.0 = specular wall, 0.0 = pure outflow
dim_v = 1

# --- Bi-modal Inflow Parameters ---
R_global = 0.95  # 1.0 = 100% recycling, <1.0 = absorption at target
R_R = 0.6  # Fast back-scattered fraction
R_T = 0.4  # Slow thermal desorption fraction
u_fast = 2.0  # Supersonic return
T_fast = 1.0
u_slow = 0.2  # Slow drift
T_slow = 0.1  # Cold emission
# ----------------------------------

print(f"Infiltration transient  Kn={Kn}  Nc={Nc}  Nv={Nv}")
print(f"  R_global={R_global}  R_R={R_R}  R_T={R_T}\n")

# ═════════════════════════════════════════════════════════════════════════════
# PROBLEM DEFINITION
# ═════════════════════════════════════════════════════════════════════════════


class PenetrationProblem:
    def __init__(self, xL, xR):
        self.name = "neutral_penetration"
        self.source_func = None
        self.x_bounds = (xL, xR)
        self.bc_type = "inflow/outflow"

    def rho_bg_func(self, x):
        """
        Mimics the detached plasma ion density profile.
        Features a slow exponential rise followed by a massive Gaussian spike
        representing the ionization front near the target.
        """
        # Normalize x
        x_norm = x / self.x_bounds[1]

        # 1. Slow exponential rise in the bulk
        n0 = 1.0  # Normalized baseline density upstream
        alpha = 2.0  # Growth rate of the bulk density
        base_density = n0 * np.exp(alpha * x_norm)

        # 2. The Ionization Peak (Gaussian spike)
        peak_height = 25.0  # The peak is roughly 25-30x the baseline
        x_center = 0.98  # Located at 98% of the domain
        peak_width = 0.02  # Extremely narrow spike

        ionization_peak = peak_height * np.exp(
            -0.5 * ((x_norm - x_center) / peak_width) ** 2
        )

        return base_density + ionization_peak

    def u_bg_func(self, x):
        """Concave velocity profile."""
        u_left = -1.0
        u_right = 1.0
        x_norm = x / self.x_bounds[1]

        k = 1.0  # Curvature strength
        linear_part = u_left + (u_right - u_left) * x_norm
        concave_bow = k * x_norm * (1.0 - x_norm)

        return linear_part + concave_bow

    def T_bg_func(self, x):
        """Convex temperature profile."""
        T_left = 0.1
        T_right = 10.0
        x_norm = x / self.x_bounds[1]

        return T_left + (T_right - T_left) * (x_norm**2)

    def f0_func(self, x_mesh, v_mesh):
        return np.zeros_like(x_mesh)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def make_solver(name, config, u_bg_array, T_bg_array, rho_bg_array):
    """Instantiate linear solvers as defined in run_infiltration."""
    if name == "ugks":
        from bgk.solvers.ugks import LinearUGKSSolver

        return LinearUGKSSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
        )
    elif name == "fvm":
        from bgk.solvers.fvm import LinearFVMSolver

        return LinearFVMSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
        )
    elif name == "sl":
        from bgk.solvers.sl import LinearSLSolver

        return LinearSLSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
        )
    elif name == "strang":
        from bgk.solvers.splitting import LinearStrangSolver

        return LinearStrangSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
        )
    elif name == "hybrid":
        from bgk.solvers.hybrid import LinearHybridSolver

        return LinearHybridSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
        )
    else:
        raise ValueError(f"Solver {name} not configured.")


def safe_save(fig, path):
    """Saves figure without overwriting existing ones."""
    base, ext = os.path.splitext(path)
    out, n = path, 1
    while os.path.exists(out):
        out = f"{base}_{n}{ext}"
        n += 1
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def capture_snapshot(runner):
    """Extracts macros at the current time step."""
    macros = runner.df.compute_macroscopics(R=R_gas)
    return {
        "t": runner.t,
        "x": runner.grid.x.copy(),
        "rho": macros[0].flatten().copy(),
        "ux": macros[1].flatten().copy(),  # 1D velocity is at index 1
        "T": macros[-1].flatten().copy(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════


def run_transient(solver_name, Kn):
    dx = (xR - xL) / Nc
    dt = CFL * dx / VMAX
    snap_times = np.linspace(0.0, T_FINAL, N_SNAPSHOTS + 1)[1:]

    print(f"  {solver_name.upper()}  t_final={T_FINAL:.2f}  steps~{int(T_FINAL / dt)}")

    problem = PenetrationProblem(xL, xR)
    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name)
    physics_conf.reflectance_left = REFLECTANCE_LEFT

    grid_conf = GridConfig(
        xL=xL,
        xR=xR,
        Nx=None,
        Nc=Nc,
        Nv=[Nv],
        dim_v=dim_v,
        vmax=VMAX,
        vmin=-VMAX,
        bc_type=problem.bc_type,
    )
    time_conf = TimeConfig(t_final=T_FINAL, dt=dt, CFL=CFL)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    # Calculate background profiles based on cell centers
    x_centers = 0.5 * (
        np.linspace(xL, xR, Nc + 1)[:-1] + np.linspace(xL, xR, Nc + 1)[1:]
    )
    u_bg_array = problem.u_bg_func(x_centers)
    T_bg_array = problem.T_bg_func(x_centers)
    rho_bg_array = problem.rho_bg_func(x_centers)

    # --- Calculate Dynamic Recycling Source (Horsten et al. method) ---
    plasma_flux_at_wall = rho_bg_array[0] * abs(u_bg_array[0])
    target_neutral_flux = R_global * plasma_flux_at_wall

    U_fast = u_fast / np.sqrt(2.0 * R_gas * T_fast)
    unit_flux_fast = (
        (R_R)
        * np.sqrt(R_gas * T_fast / (2.0 * np.pi))
        * (np.exp(-(U_fast**2)) + np.sqrt(np.pi) * U_fast * (1.0 + math.erf(U_fast)))
    )

    U_slow = u_slow / np.sqrt(2.0 * R_gas * T_slow)
    unit_flux_slow = (
        (R_T)
        * np.sqrt(R_gas * T_slow / (2.0 * np.pi))
        * (np.exp(-(U_slow**2)) + np.sqrt(np.pi) * U_slow * (1.0 + math.erf(U_slow)))
    )

    total_unit_flux = unit_flux_fast + unit_flux_slow
    rho_in = target_neutral_flux / total_unit_flux
    # -----------------------------------------------------------------

    solver = make_solver(solver_name, config, u_bg_array, T_bg_array, rho_bg_array)
    runner = Runner(config=config, solver=solver, problem=problem)

    # Initialize bi-modal inflow boundary conditions
    v = runner.grid.v
    pref_fast = (rho_in * R_R) / np.sqrt(2.0 * np.pi * R_gas * T_fast)
    f_fast = pref_fast * np.exp(-((v - u_fast) ** 2) / (2.0 * R_gas * T_fast))

    pref_slow = (rho_in * R_T) / np.sqrt(2.0 * np.pi * R_gas * T_slow)
    f_slow = pref_slow * np.exp(-((v - u_slow) ** 2) / (2.0 * R_gas * T_slow))

    f_inflow = f_fast + f_slow

    runner.df.f_flow_left = f_inflow
    runner.solver.f_flow_left = f_inflow

    snapshots = []
    snap_idx = [0]

    # Snapshot hook to capture data at specific time intervals
    def snapshot_hook(r):
        while snap_idx[0] < len(snap_times) and r.t >= snap_times[snap_idx[0]] - 1e-10:
            snapshots.append(capture_snapshot(r))
            snap_idx[0] += 1

    runner.add_hook(snapshot_hook)
    runner.run()

    if not snapshots or snapshots[-1]["t"] < runner.t - 1e-10:
        snapshots.append(capture_snapshot(runner))

    return snapshots


all_snapshots = {}

for solver_name in SOLVERS:
    print(f"\n{'=' * 55}  {solver_name.upper()}")
    snaps = run_transient(solver_name, Kn)
    all_snapshots[solver_name] = snaps

# ═════════════════════════════════════════════════════════════════════════════
# PER-SOLVER TRANSIENT FIGURES
# ═════════════════════════════════════════════════════════════════════════════
cmap = plt.cm.Blues


def time_color(i, n):
    return cmap(0.25 + 0.75 * i / max(n - 1, 1))


for solver_name, snaps in all_snapshots.items():
    fig, (ax_rho, ax_u, ax_T) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    n = len(snaps)
    for i, snap in enumerate(snaps):
        col = time_color(i, n)
        lw = 1.0 + 0.5 * (i / max(n - 1, 1))
        lbl = f"t = {snap['t']:.2f}"

        ax_rho.plot(snap["x"], snap["rho"], color=col, lw=lw, label=lbl)
        ax_u.plot(snap["x"], snap["ux"], color=col, lw=lw, label=lbl)
        ax_T.plot(snap["x"], snap["T"], color=col, lw=lw, label=lbl)

    ax_rho.set_xlabel("x")
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.set_title("(a) Density")
    ax_rho.legend(fontsize=7, loc="upper right")
    ax_rho.grid(alpha=0.3)

    ax_u.axhline(0.0, color="silver", lw=0.7, ls=":")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel(r"$u_x$")
    ax_u.set_title("(b) Velocity")
    ax_u.legend(fontsize=7, loc="upper right")
    ax_u.grid(alpha=0.3)

    ax_T.set_xlabel("x")
    ax_T.set_ylabel(r"$T$")
    ax_T.set_title("(c) Temperature")
    ax_T.legend(fontsize=7, loc="upper left")
    ax_T.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T_FINAL))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_rho, ax_u, ax_T], label="t", shrink=0.8)

    fig.suptitle(
        f"Infiltration Problem — {solver_name.upper()} — Kn={Kn}  Nc={Nc}  transient snapshots",
        fontsize=11,
    )
    safe_save(fig, f"infiltration_transient_{solver_name}_Kn{Kn}.pdf")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# FINAL STATE COMPARISON (multiple solvers)
# ═════════════════════════════════════════════════════════════════════════════
if len(SOLVERS) > 1:
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

    fig, (ax_rho, ax_u, ax_T) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    for i, solver_name in enumerate(SOLVERS):
        final = all_snapshots[solver_name][-1]
        col = _COLORS[i % len(_COLORS)]
        mk = _MARKERS[i % len(_MARKERS)]
        me = max(1, len(final["x"]) // 12)
        lbl = f"{solver_name.upper()} (t={final['t']:.2f})"

        ax_rho.plot(
            final["x"],
            final["rho"],
            "-",
            color=col,
            marker=mk,
            ms=5,
            lw=1.4,
            markevery=me,
            label=lbl,
        )
        ax_u.plot(
            final["x"],
            final["ux"],
            "-",
            color=col,
            marker=mk,
            ms=5,
            lw=1.4,
            markevery=me,
            label=lbl,
        )
        ax_T.plot(
            final["x"],
            final["T"],
            "-",
            color=col,
            marker=mk,
            ms=5,
            lw=1.4,
            markevery=me,
            label=lbl,
        )

    ax_rho.set_xlabel("x")
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.set_title("Density — final state")
    ax_rho.legend(fontsize=8)
    ax_rho.grid(alpha=0.3)

    ax_u.axhline(0.0, color="silver", lw=0.7, ls=":")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel(r"$u_x$")
    ax_u.set_title("Velocity — final state")
    ax_u.legend(fontsize=8)
    ax_u.grid(alpha=0.3)

    ax_T.set_xlabel("x")
    ax_T.set_ylabel(r"$T$")
    ax_T.set_title("Temperature — final state")
    ax_T.legend(fontsize=8)
    ax_T.grid(alpha=0.3)

    fig.suptitle(
        f"Infiltration Problem — Final state comparison — Kn={Kn}  Nc={Nc}", fontsize=11
    )
    safe_save(fig, f"infiltration_comparison_Kn{Kn}.pdf")
    plt.close(fig)

print("\nDone.")
