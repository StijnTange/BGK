"""
Fourier Flow — Run Script.
Saves results to HDF5. Skips existing files.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.io.io_hdf5 import result_exists, save_result
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

# Each entry: (solver_name, Nc, Kn, t_final, CFL)
RUNS = [
    # ("fvm", 400, 1, 50.0, 0.9),
    # Kn = 0.001
    # ("strang", 11, 0.001, 200.0, 0.9),
    # ("sl", 11, 0.001, 200.0, 0.9),
    ("hybrid", 11, 0.001, 200.0, 0.9),
    # ("fvm", 10, 0.001, 200.0, 0.9),
    # ("ugks", 10, 0.001, 200.0, 0.9),
    # ("hybrid", 11, 0.001, 200.0, 0.9),
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
    # # Kn = 1
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
    # # Kn = 100
    # ("strang", 11, 100, 50.0, 0.9),
    # ("sl", 11, 100, 50.0, 0.9),
    # ("fvm", 10, 100, 50.0, 0.9),
    # ("ugks", 10, 100, 50.0, 0.9),
    # ("hybrid", 11, 100, 50.0, 0.9),
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
    # hybrid
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

TAU_MODE = "viscosity"
RESULTS_DIR = "code/bgk/experiments/fourier/simulations"
SKIP_EXISTING = False

# Velocity space dimension: 1 (fast) or 2 (matches Huang et al. 2013)
DIM_V = 1
NV_PER_DIM = 28  # Nv for dim_v=1, or Nvx=Nvy for dim_v=2

# ═════════════════════════════════════════════════════════════════════════════
# ARGON PHYSICAL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
R_s = 208.13
mu_0 = 2.117e-5
Pr_Ar = 2.0 / 3.0
omega = 0.81
T_low = 173.0
T_upp = 373.0
T_ref = 273.0
L = 1.0

U0 = np.sqrt(R_s * T_ref)
R_nd = 1.0
T_low_nd = T_low / T_ref
T_upp_nd = T_upp / T_ref
T_ref_nd = 1.0

v_cut = 6.0 * np.sqrt(T_upp / T_ref)
print(
    f"Using v_cut={v_cut:.2f} for Fourier runs with T_upp={T_upp} K and T_ref={T_ref} K"
)

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


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def rho_from_Kn(Kn):
    factor = np.sqrt(np.pi / (2.0 * R_s * T_ref))
    return (mu_0 if TAU_MODE == "viscosity" else mu_0 / Pr_Ar) * factor / (Kn * L)


def make_solver(name, config):
    return {
        "ugks": UGKSSolver,
        "sl": SLSolver,
        "fvm": FVMSolver,
        "strang": StrangSolver,
        "hybrid": HybridSolver,
    }[name](config)


# ═════════════════════════════════════════════════════════════════════════════
# RUN LOOP
# ═════════════════════════════════════════════════════════════════════════════

for solver_name, Nc, Kn, t_final, CFL in RUNS:
    tag = f"{solver_name.upper()} Nc={Nc} Nv={nv_tag} Kn={Kn} T={t_final} CFL={CFL}"

    if SKIP_EXISTING and result_exists(
        solver_name, Nc, Nvx, Nvy, Kn, t_final, "fourier", RESULTS_DIR, CFL
    ):
        print(f"  SKIP {tag}")
        continue

    print(f"\n{'=' * 60}  {tag}")
    dt = CFL / Nc / v_cut
    print(f"  dt={dt:.4e}  steps={int(t_final / dt)}")

    config = Config(
        grid=GridConfig(
            xL=0.0,
            xR=1.0,
            Nx=Nc if solver_name != "ugks" and solver_name != "fvm" else None,
            Nc=Nc if solver_name == "ugks" or solver_name == "fvm" else None,
            Nv=Nv_list,
            dim_v=DIM_V,
            vmin=vmin_list,
            vmax=vmax_list,
            bc_type="diffusive",
        ),
        time=TimeConfig(t_final=t_final, CFL=CFL, dt=dt),
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
    runner = Runner(
        config=config, solver=make_solver(solver_name, config), problem=problem
    )
    runner.run()

    macros = runner.df.compute_macroscopics(R=R_nd)
    u_arr = macros[1:-1]
    q = runner.df.compute_heat_flux(u=u_arr)
    qx_nd = q[0].flatten() if isinstance(q, tuple) else q.flatten()

    save_result(
        dict(
            x=runner.grid.x,
            ux=macros[1].flatten(),  # <--- Added x-velocity profile here
            T=macros[-1].flatten(),
            qx=qx_nd,
            rho=macros[0].flatten(),
            problem="fourier",
            solver=solver_name,
            Nc=Nc,
            Kn=Kn,
            tau_mode=TAU_MODE,
            omega=omega,
            R_nd=R_nd,
            dim_v=DIM_V,
            Nvx=Nvx,
            Nvy=Nvy,
            t_final_requested=t_final,
            t_final_actual=float(runner.t),
            CFL=CFL,
            T_L_nd=T_low_nd,
            T_R_nd=T_upp_nd,
            T_L=T_low,
            T_R=T_upp,
            T_ref=T_ref,
            U0=U0,
            mu_0=mu_0,
            Pr_Ar=Pr_Ar,
            L=L,
        ),
        RESULTS_DIR,
    )

print("\nAll Fourier runs complete.")
