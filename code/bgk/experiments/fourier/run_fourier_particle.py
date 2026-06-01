"""
Fourier Flow — Run Script.
Saves results to HDF5. Skips existing files.

Supports both deterministic solvers (ugks, fvm, sl, strang, hybrid)
and particle solvers (rtsm, ugkp).

Particle solver notes:
  - Fourier only requires T(x) and q(x), both of which are moments of
    the 1D velocity distribution. No tangential velocity is needed.
  - For RTSM, uses ParticleRunner. For UGKP, uses UGKPRunner.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.io.io_hdf5 import result_exists, save_result
from bgk.problems.fourier import get_fourier_f0_func
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.rtsm import RTSMSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugkp import UGKPSolver
from bgk.solvers.ugks import UGKSSolver

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# Format: (solver_name, Nc, Kn, t_final, CFL, N_particles)
# Set N_particles to None for deterministic solvers.
RUNS = [
    ("fvm", 50, 0.01, 50.0, 0.9, None),  # You can mix and match!
    ("ugkp", 10, 0.01, 50.0, 0.9, 100_000),
    ("rtsm", 10, 0.01, 50.0, 0.9, 100_000),
    ("fvm", 20, 0.01, 100.0, 0.9, None),  # You can mix and match!
]

TAU_MODE = "viscosity"
RESULTS_DIR = "code/bgk/experiments/fourier/simulations"
SKIP_EXISTING = False

# Velocity space for deterministic solvers
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

U0 = np.sqrt(R_s * T_ref)
R_nd = 1.0
T_low_nd = T_low / T_ref
T_upp_nd = T_upp / T_ref

v_cut = 6.0 * np.sqrt(T_upp / T_ref)  # non-dim velocity cutoff

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


def make_det_solver(name, config):
    return {
        "ugks": UGKSSolver,
        "sl": SLSolver,
        "fvm": FVMSolver,
        "strang": StrangSolver,
        "hybrid": HybridSolver,
    }[name](config)


def make_particle_config(Nc, Kn, t_final, CFL):
    """Config for particle solvers: 1D velocity space, diffusive BC."""
    dt = CFL / Nc / v_cut
    return Config(
        grid=GridConfig(
            xL=0.0,
            xR=1.0,
            Nx=None,
            Nc=Nc,
            Nv=[0],
            dim_v=1,
            vmin=-v_cut,
            vmax=v_cut,
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
        ),
    )


def particle_heat_flux(particles):
    return particles.compute_heat_flux()


# ═════════════════════════════════════════════════════════════════════════════
# RUN LOOP
# ═════════════════════════════════════════════════════════════════════════════

for run in RUNS:
    solver_name, Nc, Kn, t_final, CFL, N_particles = run
    is_particle = solver_name in ("rtsm", "ugkp")

    # ── Particle solvers ──────────────────────────────────────────────────────
    if is_particle:
        tag = f"{solver_name.upper()} Nc={Nc} N={N_particles} Kn={Kn} T={t_final} CFL={CFL}"
        if SKIP_EXISTING and result_exists(
            solver_name, Nc, 1, None, Kn, t_final, "fourier", RESULTS_DIR, CFL
        ):
            print(f"  SKIP {tag}")
            continue

        print(f"\n{'=' * 60}  {tag}")
        config = make_particle_config(Nc, Kn, t_final, CFL)
        problem = get_problem("fourier", config)
        dt = config.time.dt
        print(f"  dt={dt:.4e}  steps={int(t_final / dt)}")

        if solver_name == "rtsm":
            solver = RTSMSolver(config)
            runner = ParticleRunner(
                config=config,
                solver=solver,
                problem=problem,
                Np=N_particles,
            )
        else:
            solver = UGKPSolver(config)
            runner = UGKPRunner(
                config=config,
                solver=solver,
                problem=problem,
                Np=N_particles,
            )

        runner.run()
        particles = runner.particles
        x = runner.grid.x

        # Smart unpacking: particles.py returns 3 items for 1D, 4 items for 2D
        moments = particles.compute_cell_moments()
        if len(moments) == 4:
            rho, Ux, Uy, T = moments
        else:
            rho, Ux, T = moments
        qx = particle_heat_flux(particles)

        save_result(
            dict(
                x=x,
                T=T,
                qx=qx,
                rho=rho,
                ux=Ux,  # <--- Extracting particle x-velocity
                problem="fourier",
                solver=solver_name,
                Nc=Nc,
                Kn=Kn,
                tau_mode=TAU_MODE,
                omega=omega,
                R_nd=R_nd,
                dim_v=1,
                Nvx=1,
                Nvy=None,
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
                rho_0=rho_from_Kn(Kn),
            ),
            RESULTS_DIR,
        )

    # ── Deterministic solvers ─────────────────────────────────────────────────
    else:
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
                Nx=None,
                Nc=Nc,
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
            ),
        )
        problem = get_problem("fourier", config)
        problem.f0_func = get_fourier_f0_func(config)
        runner = Runner(
            config=config, solver=make_det_solver(solver_name, config), problem=problem
        )
        runner.run()

        macros = runner.df.compute_macroscopics(R=R_nd)
        u_arr = macros[1:-1]
        q = runner.df.compute_heat_flux(u=u_arr)
        qx_nd = q[0].flatten() if isinstance(q, tuple) else q.flatten()

        save_result(
            dict(
                x=runner.grid.x,
                T=macros[-1].flatten(),
                qx=qx_nd,
                rho=macros[0].flatten(),
                ux=macros[1].flatten(),  # <--- Extracting deterministic x-velocity
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
                rho_0=rho_from_Kn(Kn),
            ),
            RESULTS_DIR,
        )

print("\nAll Fourier runs complete.")
