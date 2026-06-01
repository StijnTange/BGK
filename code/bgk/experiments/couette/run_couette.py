"""
Couette Flow — Run Script.
Saves results to HDF5. Skips existing files.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.io.io_hdf5 import result_exists, save_result
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridBDFSolver
from bgk.solvers.sl import SLBDFSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# Each entry: (solver_name, Nc, Kn, t_final)
RUNS = [
    ("strang", 11, 0.001, 800.0),
    ("sl", 11, 0.001, 800.0),
    ("fvm", 10, 0.001, 800.0),
    ("ugks", 10, 0.001, 800.0),
    ("hybrid", 11, 0.001, 800.0),
    # ("fvm", 50, 0.001, 300.0),
    # # Kn = 1
    # ("strang", 21, 1, 100.0),
    # ("hybrid", 21, 1, 100.0),
    # ("sl", 21, 1, 100.0),
    # ("fvm", 10, 1, 100.0),
    # ("ugks", 10, 1, 100.0),
    # Kn = 100
    #     ("strang", 11, 100, 100.0),
    #     ("hybrid", 11, 100, 100.0),
    #     ("sl", 11, 100, 100.0),
    #     ("fvm", 10, 100, 100.0),
    #     ("ugks", 10, 100, 100.0),
]


T_FINAL = 200.0  # hardcoded simulation end time (non-dim)
TAU_MODE = "viscosity"

CFL = 0.9
RESULTS_DIR = "code/bgk/experiments/couette/simulations"
SKIP_EXISTING = False

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

Nvx = 30
Nvy = 30
vxmin_nd = -6.0
vxmax_nd = 6.0
vymin_nd = -6.0
vymax_nd = 6.0 + u_w_nd


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def rho_from_Kn(Kn):
    factor = np.sqrt(np.pi / (2.0 * R_s * T_w))
    return (mu_0 if TAU_MODE == "viscosity" else mu_0 / Pr_Ar) * factor / (Kn * L)


def make_solver(name, config):
    return {
        "ugks": UGKSSolver,
        "sl": SLBDFSolver,
        "fvm": FVMSolver,
        "strang": StrangSolver,
        "hybrid": HybridBDFSolver,
    }[name](config)


def compute_shear_stress(df, macros):
    vx = df.grid.vx_mesh[None, :, :]
    vy = df.grid.vy_mesh[None, :, :]
    cx = vx - macros[1].flatten()[:, None, None]
    cy = vy - macros[2].flatten()[:, None, None]
    return df._integrate(df.f * cx * cy).flatten()


# ═════════════════════════════════════════════════════════════════════════════
# RUN LOOP
# ═════════════════════════════════════════════════════════════════════════════

for solver_name, Nc, Kn, t_final in RUNS:
    tag = f"{solver_name.upper()} Nc={Nc} Nv={Nvx}x{Nvy} Kn={Kn} T={t_final}"

    if SKIP_EXISTING and result_exists(
        solver_name, Nc, Nvx, Nvy, Kn, t_final, "couette", RESULTS_DIR
    ):
        print(f"  SKIP {tag}")
        continue

    print(f"\n{'=' * 60}  {tag}")
    vmax = max(abs(vxmin_nd), abs(vxmax_nd), abs(vymin_nd), abs(vymax_nd))
    dt = CFL / Nc / vmax
    print(f"  dt={dt:.4e}  steps={int(t_final / dt)}")

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
        time=TimeConfig(t_final=t_final, CFL=CFL, dt=dt),
        physics=PhysicsConfig(
            Kn=Kn,
            R=R_nd,
            problem_name="couette",
            omega=omega,
            u_L=0.0,
            u_R=u_w_nd,
            T_L=T_w_nd,
            T_R=T_w_nd,
            constant_tau=False,
        ),
    )
    problem = get_problem("couette", config)
    runner = Runner(
        config=config, solver=make_solver(solver_name, config), problem=problem
    )
    runner.run()

    macros = runner.df.compute_macroscopics(R=R_nd)
    save_result(
        dict(
            x=runner.grid.x,
            uy=macros[2].flatten(),
            T=macros[-1].flatten(),
            tau_xy=compute_shear_stress(runner.df, macros),
            rho=macros[0].flatten(),
            problem="couette",
            solver=solver_name,
            Nc=Nc,
            Kn=Kn,
            tau_mode=TAU_MODE,
            omega=omega,
            R_nd=R_nd,
            dim_v=dim_v,
            Nvx=Nvx,
            Nvy=Nvy,
            t_final_requested=t_final,
            t_final_actual=float(runner.t),
            CFL=CFL,
            T_w_nd=T_w_nd,
            u_w_nd=u_w_nd,
            T_w=T_w,
            u_w=u_w,
            U0=U0,
            mu_0=mu_0,
            Pr_Ar=Pr_Ar,
            L=L,
            rho_0=rho_from_Kn(Kn),
        ),
        RESULTS_DIR,
    )

print("\nAll Couette runs complete.")
