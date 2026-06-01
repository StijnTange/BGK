import collections

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


class LivePlotHook:
    """
    Updates a plot of the macroscopic variables every N steps,
    with optional rolling time-averaging to smooth Monte Carlo noise.
    """

    def __init__(
        self,
        interval_steps=10,
        pause_time=0.01,
        ylims=None,
        reference_path=None,
        scaling_factors=None,
        averaging_window=1,
    ):
        """
        Args:
            interval_steps (int): How often to update the plot.
            pause_time (float): Pause time for matplotlib GUI event loop.
            ylims (dict, optional): Dictionary of limits.
                                    e.g., {'rho': (0, 2), 'u': (-1, 1), 'T': (0, 5)}
            averaging_window (int): Number of previous steps to average over.
        """
        self.interval = interval_steps
        self.pause_time = pause_time
        self.ylims = ylims if ylims is not None else {}
        self.step_count = 0
        self.averaging_window = averaging_window

        # Rolling buffers for time-averaging
        self.history_rho = collections.deque(maxlen=self.averaging_window)
        self.history_u = collections.deque(maxlen=self.averaging_window)
        self.history_T = collections.deque(maxlen=self.averaging_window)
        self.history_q = collections.deque(maxlen=self.averaging_window)

        # Setup Figure
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        self.lines = {}

        # Set labels
        self.axs[0].set_ylabel(r"Density $\rho$")
        self.axs[1].set_ylabel(r"Velocity $u$")
        self.axs[2].set_ylabel(r"Temperature $T$")
        self.axs[2].set_xlabel(r"$x$")
        self.axs[3].set_ylabel(r"Heat Flux $q$")
        self.axs[3].set_xlabel(r"$x$")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.reference_data = None

        if reference_path:
            # Load the reference solution from riemann.mat
            mat_data = sio.loadmat(reference_path)
            self.reference_data = {
                "x": mat_data["x_refined"].flatten(),
                "rho": mat_data["rho_xx"].flatten(),
                "u": mat_data["u_xx"].flatten(),
                "T": mat_data["T_xx"].flatten(),
            }
        self.scaling_factors = scaling_factors if scaling_factors is not None else None

    def __call__(self, runner):
        self.step_count += 1

        # Get Data dynamically based on runner type
        if hasattr(runner, "df"):
            # Standard grid-based Eulerian runner
            macros = runner.df.compute_macroscopics(R=runner.config.physics.R)
            rho = macros[:1].flatten()
            u = macros[1 : 1 + runner.dim_v].flatten()
            T = macros[-1:].flatten()
            q = runner.df.compute_heat_flux(u=u).flatten()
        elif hasattr(runner, "particles"):
            # RTSM / VJ: Particle-based runner
            rho, u, T = runner.particles.compute_cell_moments()
            q = runner.particles.compute_heat_flux()
        else:
            raise AttributeError(
                "Runner does not have a recognizable fluid state ('df' or 'particles')."
            )
        if self.scaling_factors:
            rho *= self.scaling_factors.get("rho", 1.0)
            u *= self.scaling_factors.get("u", 1.0)
            T *= self.scaling_factors.get("T", 1.0)
            q *= self.scaling_factors.get("q", 1.0)
        data_map = {"rho": rho, "u": u, "T": T, "q": q}
        x = runner.grid.x

        # Initialize or Update
        if not self.lines:
            if self.reference_data:
                self.axs[0].plot(
                    self.reference_data["x"],
                    self.reference_data["rho"],
                    "k--",
                    alpha=0.5,
                    label="Ref",
                )
                self.axs[1].plot(
                    self.reference_data["x"], self.reference_data["u"], "k--", alpha=0.5
                )
                self.axs[2].plot(
                    self.reference_data["x"], self.reference_data["T"], "k--", alpha=0.5
                )
                self.axs[0].legend()

            # First Draw
            (self.lines["rho"],) = self.axs[0].plot(x, data_map["rho"], "b-")
            (self.lines["u"],) = self.axs[1].plot(x, data_map["u"], "r-")
            (self.lines["T"],) = self.axs[2].plot(x, data_map["T"], "g-")
            (self.lines["q"],) = self.axs[3].plot(x, data_map["q"], "m-")

            # Apply fixed limits immediately if they exist
            keys = ["rho", "u", "T", "q"]
            for i, key in enumerate(keys):
                if key in self.ylims:
                    self.axs[i].set_ylim(self.ylims[key])

            plt.ion()
            plt.show(block=False)
        else:
            # Update Data
            self.lines["rho"].set_ydata(data_map["rho"])
            self.lines["u"].set_ydata(data_map["u"])
            self.lines["T"].set_ydata(data_map["T"])
            self.lines["q"].set_ydata(data_map["q"])

            # Handle Scaling
            keys = ["rho", "u", "T", "q"]
            for i, key in enumerate(keys):
                if key in self.ylims:
                    self.axs[i].set_ylim(self.ylims[key])
                else:
                    self.axs[i].relim()
                    self.axs[i].autoscale_view()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(self.pause_time)

    def finalize(self):
        plt.ioff()
        print("Simulation finished. Close the plot window to exit.")
        plt.show()


class ConservedLivePlotHook:
    """
    Updates a plot of the conserved variables (rho, momentum, energy) every N steps,
    with optional rolling time-averaging.
    """

    def __init__(
        self,
        interval_steps=10,
        pause_time=0.01,
        ylims=None,
        reference_path=None,
        averaging_window=1,
    ):
        self.interval = interval_steps
        self.pause_time = pause_time
        self.ylims = ylims if ylims is not None else {}
        self.step_count = 0
        self.averaging_window = averaging_window

        self.history_rho = collections.deque(maxlen=self.averaging_window)
        self.history_mom = collections.deque(maxlen=self.averaging_window)
        self.history_ene = collections.deque(maxlen=self.averaging_window)

        # Setup Figure
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        self.lines = {}

        # Set labels for conserved variables
        self.axs[0].set_ylabel(r"Density $\rho$")
        self.axs[1].set_ylabel(r"Momentum $\rho u$")
        self.axs[2].set_ylabel(r"Total Energy $E$")
        self.axs[2].set_xlabel(r"$x$")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.reference_data = None

        if reference_path:
            mat_data = sio.loadmat(reference_path)
            self.reference_data = {
                "x": mat_data["x_refined"].flatten(),
                "rho": mat_data["rho_xx"].flatten(),
                "u": mat_data["u_xx"].flatten(),
                "T": mat_data["T_xx"].flatten(),
            }

    def __call__(self, runner):
        self.step_count += 1

        # Get Data dynamically based on runner type
        if hasattr(runner, "df"):
            rho, u, T = runner.df.compute_macroscopics()
        elif hasattr(runner, "particles"):
            rho, u, T = runner.particles.compute_cell_moments()
        else:
            raise AttributeError("Runner does not have a recognizable fluid state.")

        R = runner.config.physics.R
        momentum = rho * u
        energy = 1.5 * rho * R * T + 0.5 * rho * (u**2)

        # Append to history buffers
        self.history_rho.append(rho)
        self.history_mom.append(momentum)
        self.history_ene.append(energy)

        if self.step_count % self.interval != 0:
            return

        avg_rho = np.mean(self.history_rho, axis=0)
        avg_mom = np.mean(self.history_mom, axis=0)
        avg_ene = np.mean(self.history_ene, axis=0)

        # Update Title
        if self.averaging_window > 1:
            self.fig.suptitle(
                f"Simulation t = {runner.t:.4f} (Avg last {len(self.history_rho)}"
                f" steps)",
                fontsize=14,
            )
        else:
            self.fig.suptitle(f"Simulation t = {runner.t:.4f}", fontsize=14)

        data_map = {"rho": avg_rho, "momentum": avg_mom, "energy": avg_ene}
        x = runner.grid.x

        # Initialize or Update
        if not self.lines:
            if self.reference_data:
                ref_rho = self.reference_data["rho"]
                ref_u = self.reference_data["u"]
                ref_T = self.reference_data["T"]

                ref_mom = ref_rho * ref_u
                ref_energy = 1.5 * ref_rho * R * ref_T + 0.5 * ref_rho * (ref_u**2)

                self.axs[0].plot(
                    self.reference_data["x"], ref_rho, "k--", alpha=0.5, label="Ref"
                )
                self.axs[1].plot(self.reference_data["x"], ref_mom, "k--", alpha=0.5)
                self.axs[2].plot(self.reference_data["x"], ref_energy, "k--", alpha=0.5)
                self.axs[0].legend()

            # First Draw
            (self.lines["rho"],) = self.axs[0].plot(x, data_map["rho"], "b-")
            (self.lines["momentum"],) = self.axs[1].plot(x, data_map["momentum"], "r-")
            (self.lines["energy"],) = self.axs[2].plot(x, data_map["energy"], "g-")

            keys = ["rho", "momentum", "energy"]
            for i, key in enumerate(keys):
                if key in self.ylims:
                    self.axs[i].set_ylim(self.ylims[key])

            plt.ion()
            plt.show(block=False)
        else:
            # Update Data
            self.lines["rho"].set_ydata(data_map["rho"])
            self.lines["momentum"].set_ydata(data_map["momentum"])
            self.lines["energy"].set_ydata(data_map["energy"])

            keys = ["rho", "momentum", "energy"]
            for i, key in enumerate(keys):
                if key in self.ylims:
                    self.axs[i].set_ylim(self.ylims[key])
                else:
                    self.axs[i].relim()
                    self.axs[i].autoscale_view()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(self.pause_time)

    def finalize(self):
        plt.ioff()
        print("Simulation finished. Close the plot window to exit.")
        plt.show()


class DistributionPlotHook:
    """
    Updates a plot of the velocity distribution function f(v_x)
    at a specific spatial location x_target.
    """

    def __init__(
        self,
        x_target=0.5,
        interval_steps=10,
        pause_time=0.01,
        ylims=None,
    ):
        """
        Args:
            x_target (float): The spatial coordinate x where f should be plotted.
            interval_steps (int): How often to update the plot.
            pause_time (float): Pause time for matplotlib GUI event loop.
            ylims (tuple, optional): (ymin, ymax) for the plot.
        """
        self.x_target = x_target
        self.interval = interval_steps
        self.pause_time = pause_time
        self.ylims = ylims
        self.step_count = 0

        # Setup Figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 5))
        self.line_f = None
        self.line_M = None

        self.ax.set_xlabel(r"Velocity $v_x$")
        self.ax.set_ylabel(r"Distribution $f(v_x)$")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def __call__(self, runner):
        self.step_count += 1

        if self.step_count % self.interval != 0:
            return

        # Ensure the runner has the df object
        if not hasattr(runner, "df"):
            raise AttributeError(
                "Runner does not have a recognizable fluid state 'df'."
            )

        # Find the closest spatial index to x_target
        x_array = runner.grid.x
        idx = np.argmin(np.abs(x_array - self.x_target))
        actual_x = x_array[idx]

        self.fig.suptitle(
            f"Distribution at x ≈ {actual_x:.3f} | t = {runner.t:.4f}", fontsize=14
        )

        # Extract local f and compute local Maxwellian
        f_local = runner.df.f[idx, :, :]
        M_full = runner.df.maxwellian()
        M_local = M_full[idx, :, :]

        dvy = runner.grid.dvy
        f_vx = np.trapezoid(f_local, dx=dvy, axis=1)
        M_vx = np.trapezoid(M_local, dx=dvy, axis=1)

        vx = runner.grid.vx

        # Initialize or Update
        if self.line_f is None:
            (self.line_f,) = self.ax.plot(
                vx, f_vx, "b-", linewidth=2, label="Actual $f(v_x)$"
            )
            (self.line_M,) = self.ax.plot(
                vx, M_vx, "k--", alpha=0.6, label="Local Maxwellian"
            )
            self.ax.legend()

            if self.ylims:
                self.ax.set_ylim(self.ylims)

            plt.ion()
            plt.show(block=False)
        else:
            # Update Data
            self.line_f.set_ydata(f_vx)
            self.line_M.set_ydata(M_vx)

            # Handle Scaling
            if self.ylims:
                self.ax.set_ylim(self.ylims)
            else:
                self.ax.relim()
                self.ax.autoscale_view()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def finalize(self):
        plt.ioff()
        print("Simulation finished. Close the distribution plot window to exit.")
        plt.show()
