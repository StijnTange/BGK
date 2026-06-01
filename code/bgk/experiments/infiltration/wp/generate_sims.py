import math
import os
import sys
import time

import numpy as np

# Ensure BGK framework is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.io.io_hdf5 import result_exists, save_result
from bgk.solvers.rtsm import LinearRTSMSolver
from bgk.solvers.ugkp import LinearUGKPSolver

# Import Solvers
from bgk.solvers.ugks import LinearUGKSSolver
from bgk.solvers.vj import LinearVelocityJumpSolver

SKIP_EXISTING = True  # Set to True to skip already existing runs in io_hdf5


# =============================================================================
# 1. PROBLEM DEFINITION
# =============================================================================
class PenetrationProblem:
    def __init__(self, xL, xR):
        self.name = "infiltration"  # Kept consistent with io_hdf5.py expectation
        self.source_func = None
        self.x_bounds = (xL, xR)
        self.bc_type = "inflow/outflow"

    def rho_bg_func(self, x):
        """Exponential rise followed by a massive Gaussian ionization spike."""
        x_norm = x / self.x_bounds[1]
        n0 = 1.0
        alpha = 2.0
        base_density = n0 * np.exp(alpha * (1 - x_norm))
        peak_height = 25.0
        x_center = 0.015
        peak_width = 0.015
        ionization_peak = peak_height * np.exp(
            -0.5 * ((x_norm - x_center) / peak_width) ** 2
        )
        return base_density + ionization_peak

    def u_bg_func(self, x):
        """Concave velocity profile."""
        u_left = -1.0
        u_right = 1.0
        x_norm = x / self.x_bounds[1]
        k = 1.0
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


# =============================================================================
# 2. RUN & SAVE LOGIC
# =============================================================================
def run_wp_case(
    solver_name_base, Nc, CFL, N_INJ_TOTAL, Kn=0.5, t_final=10.0, is_reference=False
):
    # Determine the strict save name and properties
    if is_reference:
        save_name = "UGKS_REF"
        is_stoch = False
        Nv_val = 80
    else:
        # e.g., VJ_N100000
        save_name = f"{solver_name_base}_N{int(N_INJ_TOTAL)}"
        is_stoch = True
        Nv_val = 40  # Dummy value for stochastic solvers

    # Check if this exact run already exists via io_hdf5
    if (
        result_exists(
            solver=save_name,
            Nc=Nc,
            Nvx=Nv_val,
            Nvy=None,
            Kn=Kn,
            t_final=t_final,
            problem="infiltration",
            CFL=CFL,
        )
        and SKIP_EXISTING
    ):
        print(f"Skipping {save_name} (CFL={CFL}, Nc={Nc}) — already exists.")
        return

    print(f"\n--- Solving {save_name} ---")

    # Global Constants
    xL, xR = 0.0, 1.0
    R_gas = 1.0
    REFLECTANCE_LEFT = 0.9
    R_global = 0.95
    R_R = 0.6
    R_T = 0.4
    u_fast, T_fast = 2.0, 1.0
    u_slow, T_slow = 0.2, 0.1
    vmax = 20.0

    # Grid & Time Math
    L_domain = xR - xL
    dx = L_domain / Nc
    dt = CFL * dx / vmax

    # Setup Configs
    problem = PenetrationProblem(xL, xR)
    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name)
    physics_conf.reflectance_left = REFLECTANCE_LEFT

    grid_conf = GridConfig(
        xL=xL,
        xR=xR,
        Nc=Nc,
        Nx=None,
        Nv=[Nv_val],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type=problem.bc_type,
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    # Evaluate Background Profiles
    x_centers = np.linspace(xL + dx / 2, xR - dx / 2, Nc)
    u_bg_array = problem.u_bg_func(x_centers)
    T_bg_array = problem.T_bg_func(x_centers)
    rho_bg_array = problem.rho_bg_func(x_centers)

    # --- Calculate Dynamic Recycling Source (Horsten et al.) ---
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
    exact_flux = rho_in * total_unit_flux

    # Stochastic parameters
    Np, M_REF, N_inj = None, None, None
    if is_stoch:
        N_inj = float(N_INJ_TOTAL * (dt / t_final))
        M_REF = (exact_flux * dt) / N_inj

    # Instantiate Solver
    if is_reference:
        solver = LinearUGKSSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
        )
    elif solver_name_base == "rtsm":
        solver = LinearRTSMSolver(
            config,
            u_bg=u_bg_array,
            T_bg=T_bg_array,
            rho_bg=rho_bg_array,
            target_N_total=Np,
        )
    elif solver_name_base == "vj":
        solver = LinearVelocityJumpSolver(
            config,
            u_bg=u_bg_array,
            T_bg=T_bg_array,
            rho_bg=rho_bg_array,
            target_N_total=Np,
        )
    elif solver_name_base == "ugkp":
        solver = LinearUGKPSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array, rho_in=rho_in
        )
    else:
        raise ValueError("Unsupported WP solver type.")

    # Instantiate Runner
    if is_stoch:
        RunnerClass = UGKPRunner if solver_name_base == "ugkp" else ParticleRunner
        runner = RunnerClass(
            config=config,
            solver=solver,
            problem=problem,
            Np=Np,
            m_ref=M_REF,
            N_inj=N_inj,
        )

        # Bi-modal injection
        runner.particles.flow_left = [
            {"rho": rho_in * R_R, "u": u_fast, "T": T_fast, "vmax": vmax},
            {"rho": rho_in * R_T, "u": u_slow, "T": T_slow, "vmax": vmax},
        ]
    else:
        runner = Runner(config=config, solver=solver, problem=problem)
        v = runner.grid.v
        pref_fast = (rho_in * R_R) / np.sqrt(2.0 * np.pi * R_gas * T_fast)
        f_fast = pref_fast * np.exp(-((v - u_fast) ** 2) / (2.0 * R_gas * T_fast))
        pref_slow = (rho_in * R_T) / np.sqrt(2.0 * np.pi * R_gas * T_slow)
        f_slow = pref_slow * np.exp(-((v - u_slow) ** 2) / (2.0 * R_gas * T_slow))

        f_inflow = f_fast + f_slow
        runner.df.f_flow_left = f_inflow
        runner.solver.f_flow_left = f_inflow

    # Execute
    t0 = time.time()
    sim = runner.run()
    cpu_time = time.time() - t0
    print(f"[{save_name}] completed in {cpu_time:.2f}s")

    # Save via io_hdf5
    result_dict = {
        "problem": problem.name,
        "solver": save_name,
        "Nc": Nc,
        "Nvx": Nv_val,
        "Nvy": None,
        "Kn": Kn,
        "t_final_requested": t_final,
        "CFL": CFL,
        "x": sim.x,
        "rho": sim.rho[-1].flatten(),  # Save the steady-state final profile
        "u": sim.u[-1].flatten(),
        "T": sim.T[-1].flatten(),
        "q": sim.q[-1].flatten(),
        "execution_time": cpu_time,
    }

    save_result(result_dict)


# =============================================================================
# 3. MAIN WORK-PRECISION GENERATOR LOOP
# =============================================================================
def main():
    print("==================================================")
    print(" GENERATING WORK-PRECISION DATA (INFILTRATION)")
    print("==================================================")

    Kn = 0.5
    t_final = 10.0

    # --- 1. REFERENCE SOLUTION (UGKS Fine Grid) ---
    # run_wp_case(
    #     solver_name_base="ugks",
    #     Nc=400,
    #     CFL=0.9,
    #     N_INJ_TOTAL=None,
    #     Kn=Kn,
    #     t_final=t_final,
    #     is_reference=True,
    # )

    # --- 2. PARTICLE SOLVERS (Coarse Grid Nc=100) ---
    coarse_Nc = 200
    particle_counts = [5e4, 1e5, 5e5]

    # Define solvers and the list of CFLs you want to run for each
    solvers = [
        {"type": "rtsm", "cfls": [0.9, 5, 10, 20, 50, 100, 200, 400, 1000, 2000]},
        {"type": "vj", "cfls": [0.9, 5, 10, 20, 50, 100, 200, 400, 1000, 2000]},
        {"type": "ugkp", "cfls": [0.9]},  # UGKP restricted to low CFL
    ]

    for sol in solvers:
        for cfl in sol["cfls"]:
            for N in particle_counts:
                # We can keep the save name simple; io_hdf5 automatically
                # appends the _CFL attribute to the filename!
                run_wp_case(
                    solver_name_base=sol["type"],
                    Nc=coarse_Nc,
                    CFL=cfl,
                    N_INJ_TOTAL=N,
                    Kn=Kn,
                    t_final=t_final,
                )

    print("\nAll data generated successfully! Run plot_wp_diagram.py to visualize.")


if __name__ == "__main__":
    main()
