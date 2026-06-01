"""
Couette Flow — Transient Reference Script.

Plots snapshots of velocity, temperature, and density profiles over time.
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

SOLVERS = [
    "strang",
]  # order of transient snapshots and final comparison
Kn = 0.001
TAU_MODE = "viscosity"
T_FINAL = 200.0
Nc = 10
CFL = 0.9
N_SNAPSHOTS = 8

# ═════════════════════════════════════════════════════════════════════════════
# ARGON PHYSICAL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
R_s = 208.13
mu_0 = 2.117e-5
Pr_Ar = 2.0 / 3.0
omega = 0.81
T_w = 273.0
u_w = 300.0
L = 1.0
dim_v = 2

U0 = np.sqrt(R_s * T_w)
R_nd = 1.0
T_w_nd = 1.0
u_w_nd = u_w / U0
cp_nd = 2.0 * R_nd  # (dim_v/2 + 1)*R_nd for dim_v=2

Nvx = 28
Nvy = 28
vxmin_nd = -6.0
vxmax_nd = 6.0
vymin_nd = -6.0
vymax_nd = 6.0 + u_w_nd

print(f"Couette transient  Kn={Kn}  TAU_MODE={TAU_MODE}  Nc={Nc}")
print(f"  U0={U0:.2f} m/s  u_w/U0={u_w_nd:.4f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def rho_from_Kn(Kn, tau_mode):
    factor = np.sqrt(np.pi / (2.0 * R_s * T_w))
    return (mu_0 if tau_mode == "viscosity" else mu_0 / Pr_Ar) * factor / (Kn * L)


def cns_reference(x, Pr):
    """Continuum velocity and temperature (eqs. B.9, B.16)."""
    u_y = (x / L) * u_w_nd
    T = T_w_nd + (Pr / (2.0 * cp_nd)) * (x * (L - x) / L**2) * u_w_nd**2
    return u_y, T


def cns_density_nd(x, T_nd):
    """
    Continuum non-dim density from uniform pressure + mass conservation.

    Uniform pressure: rho_nd(x) = C / T_nd(x).
    C is fixed by mass conservation: integral_0^1 rho_nd dx = 1
        => C = 1 / integral_0^1 (1/T_nd) dx

    Simply using 1/T_nd (i.e. C=1) is wrong: viscous heating raises
    T_nd above 1 in the interior, so C > 1 (approx 1.065 for default
    Couette parameters — a 6% error at the wall).
    """
    C = 1.0 / np.trapezoid(1.0 / T_nd, x)
    return C / T_nd


def make_solver(name, config):
    return {
        "ugks": UGKSSolver,
        "sl": SLSolver,
        "fvm": FVMSolver,
        "strang": StrangSolver,
        "hybrid": HybridSolver,
    }[name](config)


def safe_save(fig, path):
    base, ext = os.path.splitext(path)
    out, n = path, 1
    while os.path.exists(out):
        out = f"{base}_{n}{ext}"
        n += 1
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def capture_snapshot(runner, rho_0):
    macros = runner.df.compute_macroscopics(R=R_nd)
    return {
        "t": runner.t,
        "x": runner.grid.x.copy(),
        "uy": macros[2].flatten().copy(),
        "T": macros[-1].flatten().copy(),
        "rho": macros[0].flatten().copy(),  # non-dim
        "rho_0": rho_0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════


def run_transient(solver_name, Kn, tau_mode):
    rho_0 = rho_from_Kn(Kn, tau_mode)
    vmax_abs = max(abs(vxmin_nd), abs(vxmax_nd), abs(vymin_nd), abs(vymax_nd))
    dt = CFL / Nc / vmax_abs
    snap_times = np.linspace(0.0, T_FINAL, N_SNAPSHOTS + 1)[1:]

    print(f"  {solver_name.upper()}  t_final={T_FINAL:.2f}  steps~{int(T_FINAL / dt)}")

    config = Config(
        grid=GridConfig(
            xL=0.0,
            xR=1.0,
            Nx=Nc if solver_name != "ugks" and solver_name != "fvm" else None,
            Nc=Nc if solver_name == "ugks" or solver_name == "fvm" else None,
            Nv=[Nvx, Nvy],
            dim_v=dim_v,
            vmin=[vxmin_nd, vymin_nd],
            vmax=[vxmax_nd, vymax_nd],
            bc_type="diffusive",
        ),
        time=TimeConfig(t_final=T_FINAL, CFL=CFL, dt=dt),
        physics=PhysicsConfig(
            Kn=Kn,
            R=R_nd,
            problem_name="couette",
            omega=omega,
            u_L=0.0,
            u_R=u_w_nd,
            T_L=T_w_nd,
            T_R=T_w_nd,
        ),
    )
    problem = get_problem("couette", config)
    solver = make_solver(solver_name, config)
    snapshots = []
    snap_idx = [0]

    def snapshot_hook(runner):
        while (
            snap_idx[0] < len(snap_times)
            and runner.t >= snap_times[snap_idx[0]] - 1e-10
        ):
            snapshots.append(capture_snapshot(runner, rho_0))
            snap_idx[0] += 1

    runner = Runner(config=config, solver=solver, problem=problem)
    runner.add_hook(snapshot_hook)
    runner.run()

    if not snapshots or snapshots[-1]["t"] < runner.t - 1e-10:
        snapshots.append(capture_snapshot(runner, rho_0))

    return snapshots, rho_0


all_snapshots = {}
rho_0_global = None

for solver_name in SOLVERS:
    print(f"\n{'=' * 55}  {solver_name.upper()}")
    snaps, rho_0_global = run_transient(solver_name, Kn, TAU_MODE)
    all_snapshots[solver_name] = snaps


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICAL REFERENCES
# ═════════════════════════════════════════════════════════════════════════════
x_ref = np.linspace(0.0, 1.0, 300)
u_ss, T_ss = cns_reference(x_ref, Pr=1.0)
rho_ss_nd = cns_density_nd(x_ref, T_ss)
rho_ss_phys = rho_ss_nd * rho_0_global  # [kg/m^3]

cmap = plt.cm.Blues


def time_color(i, n):
    return cmap(0.25 + 0.75 * i / max(n - 1, 1))


# ═════════════════════════════════════════════════════════════════════════════
# PER-SOLVER TRANSIENT FIGURES
# ═════════════════════════════════════════════════════════════════════════════
for solver_name, snaps in all_snapshots.items():
    fig, (ax_u, ax_T, ax_rho) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    ax_u.plot(x_ref, u_ss, "k--", lw=1.5, label="CNS Pr=1 (SS)", zorder=10)
    ax_T.plot(x_ref, T_ss, "k--", lw=1.5, label="CNS Pr=1 (SS)", zorder=10)
    ax_rho.plot(
        x_ref,
        rho_ss_phys,
        "k--",
        lw=1.5,
        label=r"CNS Pr=1 ($\rho = C/T_\mathrm{nd}$, mass-cons.)",
        zorder=10,
    )

    n = len(snaps)
    for i, snap in enumerate(snaps):
        col = time_color(i, n)
        lw = 1.0 + 0.5 * (i / max(n - 1, 1))
        lbl = f"t = {snap['t']:.2f}"

        ax_u.plot(snap["x"], snap["uy"], color=col, lw=lw, label=lbl)
        ax_T.plot(snap["x"], snap["T"], color=col, lw=lw, label=lbl)
        ax_rho.plot(snap["x"], snap["rho"] * snap["rho_0"], color=col, lw=lw, label=lbl)

    ax_u.axhline(0.0, color="silver", lw=0.7, ls=":")
    ax_u.axhline(u_w_nd, color="silver", lw=0.7, ls=":")
    ax_u.set_xlabel("x / L")
    ax_u.set_ylabel(r"$u_y / U_0$")
    ax_u.set_title("(a) Velocity")
    ax_u.legend(fontsize=7, loc="upper left")
    ax_u.grid(alpha=0.3)

    ax_T.set_xlabel("x / L")
    ax_T.set_ylabel(r"$T / T_w$")
    ax_T.set_title("(b) Temperature")
    ax_T.legend(fontsize=7, loc="upper left")
    ax_T.grid(alpha=0.3)

    ax_rho.set_xlabel("x / L")
    ax_rho.set_ylabel(r"$\rho$  [kg/m³]")
    ax_rho.set_title("(c) Density")
    ax_rho.legend(fontsize=7, loc="upper right")
    ax_rho.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T_FINAL))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_u, ax_T, ax_rho], label="t (non-dim)", shrink=0.8)

    fig.suptitle(
        f"Couette Flow — {solver_name.upper()} — Kn={Kn}  Nc={Nc}  transient snapshots",
        fontsize=11,
    )
    # safe_save(fig, f"couette_transient_{solver_name}_Kn{Kn}.pdf")
    # plt.close(fig)
    plt.show()


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

    fig, (ax_u, ax_T, ax_rho) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    ax_u.plot(x_ref, u_ss, "k--", lw=1.8, label="CNS Pr=1 (SS)", zorder=10)
    ax_T.plot(x_ref, T_ss, "k--", lw=1.8, label="CNS Pr=1 (SS)", zorder=10)
    ax_rho.plot(
        x_ref,
        rho_ss_phys,
        "k--",
        lw=1.8,
        label=r"CNS Pr=1 ($\rho = C/T_\mathrm{nd}$, mass-cons.)",
        zorder=10,
    )

    for i, solver_name in enumerate(SOLVERS):
        final = all_snapshots[solver_name][-1]
        col = _COLORS[i % len(_COLORS)]
        mk = _MARKERS[i % len(_MARKERS)]
        me = max(1, len(final["x"]) // 12)
        lbl = f"{solver_name.upper()} (t={final['t']:.2f})"

        ax_u.plot(
            final["x"],
            final["uy"],
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
        ax_rho.plot(
            final["x"],
            final["rho"] * final["rho_0"],
            "-",
            color=col,
            marker=mk,
            ms=5,
            lw=1.4,
            markevery=me,
            label=lbl,
        )

    ax_u.axhline(0.0, color="silver", lw=0.7, ls=":")
    ax_u.axhline(u_w_nd, color="silver", lw=0.7, ls=":")
    ax_u.set_xlabel("x / L")
    ax_u.set_ylabel(r"$u_y / U_0$")
    ax_u.set_title("Velocity — final state")
    ax_u.legend(fontsize=8)
    ax_u.grid(alpha=0.3)

    ax_T.set_xlabel("x / L")
    ax_T.set_ylabel(r"$T / T_w$")
    ax_T.set_title("Temperature — final state")
    ax_T.legend(fontsize=8)
    ax_T.grid(alpha=0.3)

    ax_rho.set_xlabel("x / L")
    ax_rho.set_ylabel(r"$\rho$  [kg/m³]")
    ax_rho.set_title("Density — final state")
    ax_rho.legend(fontsize=8)
    ax_rho.grid(alpha=0.3)

    fig.suptitle(
        f"Couette Flow — Final state comparison — Kn={Kn}  Nc={Nc}", fontsize=11
    )
    safe_save(fig, f"couette_comparison_Kn{Kn}.pdf")
    plt.close(fig)

print("\nDone.")
