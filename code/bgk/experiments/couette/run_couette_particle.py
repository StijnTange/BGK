"""
Couette Flow — Run Script.
Saves results to HDF5. Skips existing files.

Supports both deterministic solvers (ugks, fvm, sl, strang, hybrid)
and particle solvers (rtsm, ugkp).

Particle solver notes:
  - Transport is 1D (x-direction only: vx).
  - The tangential velocity u_y is tracked as a separate per-particle
    array 'vy', initialised from the equilibrium Maxwellian at t=0 and
    updated at each diffusive wall reflection.
  - Shear stress tau_xy is estimated as a particle moment.
  - Add N_particles as a 4th element to the RUNS tuple for particle solvers.
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

# Deterministic: (solver_name, Nc, Kn)
# Particle:      (solver_name, Nc, Kn, N_particles)
RUNS = [
    (
        "ugkp",
        10,
        0.001,
        int(1e5),
    ),
    ("rtsm", 10, 0.001, int(1e5)),
]

T_FINAL = 800.0
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

U0 = np.sqrt(R_s * T_w)
R_nd = 1.0
T_w_nd = 1.0
u_w_nd = u_w / U0

# Deterministic solver velocity grid (dim_v=2)
dim_v = 2
Nvx = 28
Nvy = 28
vxmin_nd = -6.0
vxmax_nd = 6.0
vymin_nd = -6.0
vymax_nd = 6.0 + u_w_nd

# Particle solver: vx cut (x-direction only; vy tracked separately)
vx_cut_nd = max(abs(vxmin_nd), abs(vxmax_nd))


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def rho_from_Kn(Kn):
    factor = np.sqrt(np.pi / (2.0 * R_s * T_w))
    return (mu_0 if TAU_MODE == "viscosity" else mu_0 / Pr_Ar) * factor / (Kn * L)


def make_det_solver(name, config):
    return {
        "ugks": UGKSSolver,
        "sl": SLSolver,
        "fvm": FVMSolver,
        "strang": StrangSolver,
        "hybrid": HybridSolver,
    }[name](config)


def make_particle_config(Nc, Kn):
    """Config for particle solvers: 1D velocity space (x-direction)."""
    vmax = vx_cut_nd
    dt = CFL / Nc / vmax
    return Config(
        grid=GridConfig(
            xL=0.0,
            xR=1.0,
            Nx=None,
            Nc=Nc,
            Nv=[0],
            dim_v=1,
            vmin=-vmax,
            vmax=vmax,
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


def make_f0_func_1d(problem_2d):
    """
    Return a 1D marginal f0_func(x, vx) by analytically integrating the
    2D Couette initial Maxwellian over vy.

    The Couette IC is f(x, vx, vy) = rho/(2*pi*R*T) * exp(-(vx^2+vy^2)/(2RT)).
    Integrating over vy in [-inf, inf]:
        f_1d(x, vx) = rho/sqrt(2*pi*R*T) * exp(-vx^2/(2RT))

    We evaluate this by calling the 2D f0_func with vy=0 and multiplying
    by sqrt(2*pi*R*T) to undo the vy normalisation — equivalent to
    analytically marginalising the Gaussian in vy.

    This avoids any dependence on the vy grid bounds or resolution.
    """

    def f0_func_1d(x, vx):
        # f_2d(x, vx, vy=0) = rho/(2*pi*R*T) * exp(-vx^2/(2RT))
        # f_1d(x, vx)       = rho/sqrt(2*pi*R*T) * exp(-vx^2/(2RT))
        # => f_1d = f_2d(vy=0) * sqrt(2*pi*R*T)
        vy_zero = np.zeros_like(vx)
        f2d = problem_2d.f0_func(x, vx, vy_zero)
        # Recover R*T from the ratio f_2d / Maxwellian_vx:
        # Rather than re-extracting R and T, multiply by sqrt(2*pi*R*T).
        # For the Couette IC, T=T_w_nd and R=R_nd everywhere at t=0.
        return f2d * np.sqrt(2.0 * np.pi * R_nd * T_w_nd)

    return f0_func_1d


def init_vy(particles):
    """
    Initialise tangential velocity vy on each particle.
    At t=0: drawn from Maxwellian with u_y=0, T=T_w (equilibrium).
    The diffusive BC in particles.py then updates vy at each wall hit.
    """
    particles.vy = np.random.normal(
        loc=0.0,
        scale=np.sqrt(R_nd * T_w_nd),
        size=particles.N_total,
    )
    return particles


def cell_mean_uy(particles):
    """Cell-averaged u_y from per-particle vy."""
    cell_idx = particles.get_cell_indices()
    mass_sum = np.bincount(cell_idx, weights=particles.m, minlength=particles.n_cells)
    mom_y = np.bincount(
        cell_idx, weights=particles.m * particles.vy, minlength=particles.n_cells
    )
    uy = np.zeros(particles.n_cells)
    valid = mass_sum > 0
    uy[valid] = mom_y[valid] / mass_sum[valid]
    return uy


def compute_shear_stress_particles(particles):
    """
    tau_xy[i] = (1/dx) * sum_{p in cell i} m_p * (vx_p - Ux_i) * (vy_p - Uy_i)
    """
    cell_idx = particles.get_cell_indices()
    mass_sum = np.bincount(cell_idx, weights=particles.m, minlength=particles.n_cells)
    valid = mass_sum > 0
    Ux = np.zeros(particles.n_cells)
    Uy = np.zeros(particles.n_cells)
    Ux[valid] = (
        np.bincount(
            cell_idx, weights=particles.m * particles.v, minlength=particles.n_cells
        )[valid]
        / mass_sum[valid]
    )
    Uy[valid] = (
        np.bincount(
            cell_idx, weights=particles.m * particles.vy, minlength=particles.n_cells
        )[valid]
        / mass_sum[valid]
    )

    cx = particles.v - Ux[cell_idx]
    cy = particles.vy - Uy[cell_idx]

    tau = (
        np.bincount(
            cell_idx, weights=particles.m * cx * cy, minlength=particles.n_cells
        )
        / particles.dx
    )
    return tau


def compute_shear_stress_det(df, macros):
    """Shear stress from distribution function (deterministic solvers)."""
    vx = df.grid.vx_mesh[None, :, :]
    vy = df.grid.vy_mesh[None, :, :]
    cx = vx - macros[1].flatten()[:, None, None]
    cy = vy - macros[2].flatten()[:, None, None]
    return df._integrate(df.f * cx * cy).flatten()


# ═════════════════════════════════════════════════════════════════════════════
# RUN LOOP
# ═════════════════════════════════════════════════════════════════════════════

for run in RUNS:
    solver_name = run[0]
    Nc = run[1]
    Kn = run[2]
    N_particles = run[3] if len(run) > 3 else None
    is_particle = solver_name in ("rtsm", "ugkp")

    # ── Particle solvers ──────────────────────────────────────────────────────
    if is_particle:
        tag = f"{solver_name.upper()} Nc={Nc} N={N_particles} Kn={Kn} T={T_FINAL}"
        # Use Nvx=1, Nvy=None as the particle "Nv" identifier in HDF5
        if SKIP_EXISTING and result_exists(
            solver_name, Nc, 1, None, Kn, T_FINAL, "couette", RESULTS_DIR
        ):
            print(f"  SKIP {tag}")
            continue

        print(f"\n{'=' * 60}  {tag}")
        config = make_particle_config(Nc, Kn)
        # get_problem returns the 2D couette problem; we need the 1D marginal
        # f0_func for ParticleRunner which calls f0_func(x, vx) with 2 args.
        problem_2d = get_problem("couette", config)
        problem = get_problem("couette", config)
        problem.f0_func = make_f0_func_1d(problem_2d)
        dt = config.time.dt
        print(f"  dt={dt:.4e}  steps={int(T_FINAL / dt)}")

        if solver_name == "rtsm":
            solver = RTSMSolver(config)
            runner = ParticleRunner(
                config=config,
                solver=solver,
                problem=problem,
                Np=N_particles,
            )
            runner.particles = init_vy(runner.particles)
            runner.run()
            particles = runner.particles
            x = runner.grid.x

        else:  # ugkp
            solver = UGKPSolver(config)
            runner = UGKPRunner(
                config=config,
                solver=solver,
                problem=problem,
                Np=N_particles,
            )
            runner.particles = init_vy(runner.particles)
            runner.run()
            particles = runner.particles
            x = runner.grid.x

        rho, _, _, T_1d = particles.compute_cell_moments()  # T from vx only

        # Compute 2D temperature (both vx and vy) for diagnostics
        cell_idx = particles.get_cell_indices()
        mass_sum = np.bincount(
            cell_idx, weights=particles.m, minlength=particles.n_cells
        )
        valid = mass_sum > 0
        Ux_c = np.zeros(particles.n_cells)
        Uy_c = np.zeros(particles.n_cells)
        Ux_c[valid] = (
            np.bincount(
                cell_idx, weights=particles.m * particles.v, minlength=particles.n_cells
            )[valid]
            / mass_sum[valid]
        )
        Uy_c[valid] = (
            np.bincount(
                cell_idx,
                weights=particles.m * particles.vy,
                minlength=particles.n_cells,
            )[valid]
            / mass_sum[valid]
        )
        Ex = np.bincount(
            cell_idx, weights=particles.m * particles.v**2, minlength=particles.n_cells
        )
        Ey = np.bincount(
            cell_idx, weights=particles.m * particles.vy**2, minlength=particles.n_cells
        )
        T_2d = np.zeros(particles.n_cells)
        T_2d[valid] = (
            (Ex[valid] - mass_sum[valid] * Ux_c[valid] ** 2)
            + (Ey[valid] - mass_sum[valid] * Uy_c[valid] ** 2)
        ) / (2.0 * mass_sum[valid] * R_nd)

        print(
            f"  T_1d (vx only): wall={T_1d[0]:.4f}/{T_1d[-1]:.4f}  "
            f"centre={T_1d[len(T_1d) // 2]:.4f}"
        )
        print(
            f"  T_2d (vx+vy):   wall={T_2d[0]:.4f}/{T_2d[-1]:.4f}  "
            f"centre={T_2d[len(T_2d) // 2]:.4f}"
        )

        # Save the 2D temperature — this is the physically correct one for a 2D gas
        T = T_2d
        uy = cell_mean_uy(particles)
        tau_xy = compute_shear_stress_particles(particles)

        save_result(
            dict(
                x=x,
                uy=uy,
                T=T,
                tau_xy=tau_xy,
                rho=rho,
                problem="couette",
                solver=solver_name,
                Nc=Nc,
                Kn=Kn,
                tau_mode=TAU_MODE,
                omega=omega,
                R_nd=R_nd,
                dim_v=1,
                Nvx=1,
                Nvy=None,
                t_final_requested=T_FINAL,
                t_final_actual=T_FINAL,
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

    # ── Deterministic solvers ─────────────────────────────────────────────────
    else:
        tag = f"{solver_name.upper()} Nc={Nc} Nv={Nvx}x{Nvy} Kn={Kn} T={T_FINAL}"
        if SKIP_EXISTING and result_exists(
            solver_name, Nc, Nvx, Nvy, Kn, T_FINAL, "couette", RESULTS_DIR
        ):
            print(f"  SKIP {tag}")
            continue

        print(f"\n{'=' * 60}  {tag}")
        vmax = max(abs(vxmin_nd), abs(vxmax_nd), abs(vymin_nd), abs(vymax_nd))
        dt = CFL / Nc / vmax
        print(f"  dt={dt:.4e}  steps={int(T_FINAL / dt)}")

        config = Config(
            grid=GridConfig(
                xL=0.0,
                xR=1.0,
                Nx=None,
                Nc=Nc,
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
        runner = Runner(
            config=config, solver=make_det_solver(solver_name, config), problem=problem
        )
        runner.run()

        macros = runner.df.compute_macroscopics(R=R_nd)
        save_result(
            dict(
                x=runner.grid.x,
                uy=macros[2].flatten(),
                T=macros[-1].flatten(),
                tau_xy=compute_shear_stress_det(runner.df, macros),
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
                t_final_requested=T_FINAL,
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
