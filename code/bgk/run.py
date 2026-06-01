import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.io.hooks import LivePlotHook
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.rtsm import RTSMSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugkp import UGKPSolver
from bgk.solvers.ugks import UGKSSolver
from bgk.solvers.vj import VelocityJumpSolver


def main():
    # select problem
    problem_name = "riemann"  # options: "gaussian" and "riemann"
    problem = get_problem(problem_name)

    # select solver
    solver_name = "sl"  # options: "ugks", "sl", "fvm", "strang"

    N_particles = int(200 * 2000)

    # set parameters
    CFL = 0.9
    vmax = 20.0
    t_final = 0.07
    Kn = 1e-8
    omega = 0.5
    u_L = 0.0
    u_R = 0.0
    T_L = 173.0 / 273.0
    T_R = 373.0 / 273.0
    R = 1.0

    dim_v = 1
    Nv = [40] * dim_v
    Nx = 200
    Nc = None
    if Nc is not None:
        dx = (problem.x_bounds[1] - problem.x_bounds[0]) / (Nc)
    elif Nx is not None:
        dx = (problem.x_bounds[1] - problem.x_bounds[0]) / (Nx - 1)
    else:
        raise ValueError("Either Nx or Nc must be specified.")
    dt = CFL * dx / vmax

    print("dt/tau =", dt / (Kn * np.sqrt(2.0 / np.pi)))

    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=Nx,
        Nc=Nc,
        Nv=Nv,
        dim_v=dim_v,
        vmin=-vmax,
        vmax=vmax,
        bc_type=problem.bc_type,
    )
    time_conf = TimeConfig(t_final=t_final, CFL=CFL, dt=dt)
    physics_conf = PhysicsConfig(
        Kn=Kn,
        R=R,
        problem_name=problem_name,
        omega=omega,
        u_L=u_L,
        u_R=u_R,
        T_L=T_L,
        T_R=T_R,
    )
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    is_particle_method = False
    # create solver
    if solver_name == "ugks":
        solver = UGKSSolver(config)
    elif solver_name == "hybrid":
        solver = HybridSolver(config)
    elif solver_name == "sl":
        solver = SLSolver(config)
    elif solver_name == "fvm":
        solver = FVMSolver(config)
    elif solver_name == "strang":
        solver = StrangSolver(config)
    elif solver_name == "rtsm":
        solver = RTSMSolver(config)
        is_particle_method = True
    elif solver_name == "vj":
        solver = VelocityJumpSolver(config)
        is_particle_method = True
    elif solver_name == "ugkp":
        solver = UGKPSolver(config)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    # create runner
    if solver_name == "ugkp":
        solver = UGKPSolver(config)
        runner = UGKPRunner(
            config=config, solver=solver, problem=problem, Np=N_particles
        )
    elif is_particle_method:
        runner = ParticleRunner(
            config=config, solver=solver, problem=problem, Np=N_particles
        )
    else:
        runner = Runner(config=config, solver=solver, problem=problem)

    # limits for live plotting
    if problem_name == "gaussian":
        my_ylims = {
            "rho": (0.90, 1.10),
            "u": (-0.25, 0.15),
            "T": (0.75, 1.15),
        }
    elif problem_name == "riemann":
        my_ylims = {
            "rho": (0, 1.2),
            "u": (-1, 2),
            "T": (0, 10),
            "q": (-0.2, 0.2),
        }
    else:
        my_ylims = None
    # add reference solution to live plot hook
    if problem_name == "riemann":
        ref_path = "code/bgk/reference_solutions/riemann.mat"
    else:
        ref_path = None

    # Physical scaling
    T0 = 273.0
    L = 1.0
    R_spec = 208.13
    mu_0 = 2.117e-5
    U0 = np.sqrt(R_spec * T0)
    rho_0 = mu_0 / (Kn * L * np.sqrt(2.0 * R_spec * T0 / np.pi))
    q0 = rho_0 * (U0**3)

    # print scaling factors for reference solution
    print("Scaling factors for reference solution:")
    print(f"Density (rho): {rho_0:.4e} kg/m^3")
    print(f"Velocity (u): {U0:.4e} m/s")
    print(f"Temperature (T): {T0:.4e} K")
    print(f"Heat Flux (q): {q0:.4e} W/m^2")

    # scaling_factors = {"rho": rho_0, "u": U0, "T": T0, "q": q0}
    scaling_factors = None
    runner.add_hook(
        LivePlotHook(
            interval_steps=1,
            ylims=my_ylims,
            reference_path=ref_path,
            scaling_factors=scaling_factors,
        )
    )

    # dist_hook = DistributionPlotHook(x_target=0.1, interval_steps=1)
    # runner.hooks.append(dist_hook)

    # run simulation

    runner.run()

    print("Simulation Complete.")


if __name__ == "__main__":
    main()
