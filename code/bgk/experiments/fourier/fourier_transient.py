"""
Fourier Flow — Transient Reference Script.

Runs one or more solvers on a fine grid and plots snapshots of the
temperature, density, and heat flux profiles at multiple time points,
coloured by a gradient from early (light) to late (dark).
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.fourier import get_fourier_f0_func
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

SOLVERS = ["strang"]
Kn = 0.001
TAU_MODE = "viscosity"
T_FINAL = 150.0
Nc = 20
CFL = 0.9
N_SNAPSHOTS = 8
DIM_V = 1
NV_PER_DIM = 28

# ═════════════════════════════════════════════════════════════════════════════
# ARGON PHYSICAL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
R_s = 208.13
mu_0 = 2.117e-5
Pr_Ar = 2.0 / 3.0
omega = 0.81
T_low = 173.0
T_upp = 373.0
T_ref = (T_low + T_upp) / 2.0
L = 1.0

HF_COEFF_BGK = DIM_V / 2.0 + 1.0
HF_COEFF_PHYS = HF_COEFF_BGK / Pr_Ar

U0 = np.sqrt(R_s * T_ref)
R_nd = 1.0
T_low_nd = T_low / T_ref
T_upp_nd = T_upp / T_ref
T_ref_nd = 1.0

v_cut = 6.0 * np.sqrt(T_upp / T_ref)

if DIM_V == 1:
    Nv_list = [NV_PER_DIM]
    vmin_list = [-v_cut]
    vmax_list = [v_cut]
    Nvx, Nvy = NV_PER_DIM, None
else:
    Nv_list = [NV_PER_DIM, NV_PER_DIM]
    vmin_list = [-v_cut, -v_cut]
    vmax_list = [v_cut, v_cut]
    Nvx, Nvy = NV_PER_DIM, NV_PER_DIM

nv_tag = str(Nvx) if Nvy is None else f"{Nvx}x{Nvy}"

print(f"Fourier transient  Kn={Kn}  TAU_MODE={TAU_MODE}  Nc={Nc}")
print(f"  T_low={T_low} K  T_upp={T_upp} K  T_ref={T_ref} K")
print(f"  DIM_V={DIM_V}  Nv={nv_tag}\n")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICAL LIMITS
# ═════════════════════════════════════════════════════════════════════════════


def T_continuum_nd(x_nd):
    return np.sqrt(T_low_nd**2 + (T_upp_nd**2 - T_low_nd**2) * x_nd)


def T_free_molecular_nd():
    return np.sqrt(T_low_nd * T_upp_nd)


def q_continuum_nd(x_nd, mu_0_nd, hf_coeff):
    T_c = T_continuum_nd(x_nd)
    mu_T = mu_0_nd * (T_c**omega)
    dTdx = (T_upp_nd**2 - T_low_nd**2) / (2.0 * T_c)
    return -hf_coeff * mu_T * R_nd * dTdx


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def rho_from_Kn(Kn, tau_mode):
    factor = np.sqrt(np.pi / (2.0 * R_s * T_ref))
    return (mu_0 if tau_mode == "viscosity" else mu_0 / Pr_Ar) * factor / (Kn * L)


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
    """Capture T, rho (non-dim), and qx from current runner state."""
    macros = runner.df.compute_macroscopics(R=R_nd)
    u_arr = macros[1:-1]
    q = runner.df.compute_heat_flux(u=u_arr)
    qx_nd = q[0].flatten() if isinstance(q, tuple) else q.flatten()
    return {
        "t": runner.t,
        "x": runner.grid.x.copy(),
        "T": macros[-1].flatten().copy(),
        "rho": macros[0].flatten().copy(),  # non-dim; scale by rho_0 for physical
        "qx": qx_nd.copy(),
        "rho_0": rho_0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════


def run_transient(solver_name, Kn, tau_mode):
    rho_0 = rho_from_Kn(Kn, tau_mode)
    vmax_abs = max(abs(v) for v in vmin_list + vmax_list)
    dt = CFL / Nc / vmax_abs
    snap_times = np.linspace(0.0, T_FINAL, N_SNAPSHOTS + 1)[1:]

    print(f"  {solver_name.upper()}  t_final={T_FINAL:.2f}  steps~{int(T_FINAL / dt)}")

    config = Config(
        grid=GridConfig(
            xL=0.0,
            xR=1.0,
            Nx=None,
            Nc=Nc,
            Nv=Nv_list,
            dim_v=DIM_V,
            vmin=vmin_list,
            vmax=vmax_list,
            bc_type="diffusive",
        ),
        time=TimeConfig(t_final=T_FINAL, CFL=CFL, dt=dt),
        physics=PhysicsConfig(
            Kn=Kn,
            R=R_nd,
            problem_name="fourier",
            omega=omega,
            u_L=0.0,
            u_R=0.0,
            T_L=T_low_nd,
            T_R=T_upp_nd,
            constant_tau=False,
        ),
    )
    problem = get_problem("fourier", config)
    problem.f0_func = get_fourier_f0_func(config)
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
# ANALYTICAL REFERENCES (computed once)
# ═════════════════════════════════════════════════════════════════════════════
x_ref_nd = np.linspace(0.0, 1.0, 300)
T_cont_nd = T_continuum_nd(x_ref_nd)
T_fm_nd = T_free_molecular_nd()
mu_0_nd = Kn * np.sqrt(2.0 / np.pi)
q_ss_nd = q_continuum_nd(x_ref_nd, mu_0_nd, HF_COEFF_BGK)
q_scale = rho_0_global * U0**3

# Continuum density: p = rho*R*T = const => rho_nd(x) = 1 / T_nd(x)
# Physical: rho_phys(x) = rho_nd(x) * rho_0
rho_cont_nd = 1.0 / T_cont_nd
rho_cont_phys = rho_cont_nd * rho_0_global  # [kg/m^3]

cmap = plt.cm.Oranges


def time_color(i, n):
    return cmap(0.25 + 0.75 * i / max(n - 1, 1))


# ═════════════════════════════════════════════════════════════════════════════
# PER-SOLVER TRANSIENT FIGURES  (T, rho, q)
# ═════════════════════════════════════════════════════════════════════════════
for solver_name, snaps in all_snapshots.items():
    fig, (ax_T, ax_rho, ax_q) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    # Analytical references
    ax_T.plot(x_ref_nd, T_cont_nd * T_ref, "k-", lw=1.8, label="Continuum", zorder=10)
    ax_T.axhline(
        T_fm_nd * T_ref,
        color="k",
        lw=1.8,
        ls="--",
        label=f"Free mol.: {T_fm_nd * T_ref:.1f} K",
        zorder=10,
    )

    ax_rho.plot(
        x_ref_nd,
        rho_cont_phys,
        "k-",
        lw=1.8,
        label=r"Continuum ($\rho \propto 1/T$)",
        zorder=10,
    )

    ax_q.plot(
        x_ref_nd,
        np.abs(q_ss_nd) * q_scale,
        "k-",
        lw=1.8,
        label="BGK continuum",
        zorder=10,
    )

    n = len(snaps)
    for i, snap in enumerate(snaps):
        col = time_color(i, n)
        lw = 1.0 + 0.5 * (i / max(n - 1, 1))
        lbl = f"t = {snap['t']:.2f}"
        qs = snap["rho_0"] * U0**3

        ax_T.plot(snap["x"], snap["T"] * T_ref, color=col, lw=lw, label=lbl)
        ax_rho.plot(snap["x"], snap["rho"] * snap["rho_0"], color=col, lw=lw, label=lbl)
        ax_q.semilogy(snap["x"], np.abs(snap["qx"]) * qs, color=col, lw=lw, label=lbl)

    ax_T.set_xlabel("x / L")
    ax_T.set_ylabel("T  [K]")
    ax_T.set_title("(a) Temperature")
    ax_T.legend(fontsize=7, loc="upper left")
    ax_T.grid(alpha=0.3)

    ax_rho.set_xlabel("x / L")
    ax_rho.set_ylabel(r"$\rho$  [kg/m³]")
    ax_rho.set_title("(b) Density")
    ax_rho.legend(fontsize=7, loc="upper right")
    ax_rho.grid(alpha=0.3)

    ax_q.set_xlabel("x / L")
    ax_q.set_ylabel("|q|  [W/m²]")
    ax_q.set_title("(c) Heat flux")
    ax_q.legend(fontsize=7, loc="upper right")
    ax_q.grid(alpha=0.3, which="both")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T_FINAL))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_T, ax_rho, ax_q], label="t (non-dim)", shrink=0.8)

    fig.suptitle(
        f"Fourier Flow — {solver_name.upper()} — {DIM_V}D — "
        f"Kn={Kn}  Nc={Nc}  transient snapshots",
        fontsize=11,
    )
    # safe_save(fig, f"fourier_transient_{solver_name}_{DIM_V}D_Kn{Kn}.pdf")
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

    fig, (ax_T, ax_rho, ax_q) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    ax_T.plot(x_ref_nd, T_cont_nd * T_ref, "k-", lw=1.8, label="Continuum", zorder=10)
    ax_T.axhline(
        T_fm_nd * T_ref,
        color="k",
        lw=1.8,
        ls="--",
        label=f"Free mol.: {T_fm_nd * T_ref:.1f} K",
        zorder=10,
    )
    ax_rho.plot(
        x_ref_nd,
        rho_cont_phys,
        "k-",
        lw=1.8,
        label=r"Continuum ($\rho \propto 1/T$)",
        zorder=10,
    )
    ax_q.plot(
        x_ref_nd,
        np.abs(q_ss_nd) * q_scale,
        "k-",
        lw=1.8,
        label="BGK continuum",
        zorder=10,
    )

    for i, solver_name in enumerate(SOLVERS):
        final = all_snapshots[solver_name][-1]
        col = _COLORS[i % len(_COLORS)]
        mk = _MARKERS[i % len(_MARKERS)]
        me = max(1, len(final["x"]) // 12)
        qs = final["rho_0"] * U0**3
        lbl = f"{solver_name.upper()} (t={final['t']:.2f})"

        ax_T.plot(
            final["x"],
            final["T"] * T_ref,
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
        ax_q.semilogy(
            final["x"],
            np.abs(final["qx"]) * qs,
            "-",
            color=col,
            marker=mk,
            ms=5,
            lw=1.4,
            markevery=me,
            label=lbl,
        )

    ax_T.set_xlabel("x / L")
    ax_T.set_ylabel("T  [K]")
    ax_T.set_title("Temperature — final state")
    ax_T.legend(fontsize=8)
    ax_T.grid(alpha=0.3)

    ax_rho.set_xlabel("x / L")
    ax_rho.set_ylabel(r"$\rho$  [kg/m³]")
    ax_rho.set_title("Density — final state")
    ax_rho.legend(fontsize=8)
    ax_rho.grid(alpha=0.3)

    ax_q.set_xlabel("x / L")
    ax_q.set_ylabel("|q|  [W/m²]")
    ax_q.set_title("Heat flux — final state")
    ax_q.legend(fontsize=8)
    ax_q.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Fourier Flow — Final state comparison — {DIM_V}D — Kn={Kn}  Nc={Nc}",
        fontsize=11,
    )
    # safe_save(fig, f"fourier_comparison_{DIM_V}D_Kn{Kn}.pdf")
    # plt.close(fig)
    plt.show()

print("\nDone.")
