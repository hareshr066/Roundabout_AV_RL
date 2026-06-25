import os
import sys
import time
import traci
import sumolib

class TraCIConnection:
    """
    Helper manager to establish and control a TraCI connection to SUMO.
    Provides standard retrieval and control interface for Reinforcement Learning agents.
    """
    def __init__(self, config_path, gui=False, step_length=0.1, label="default", additional_args=None):
        """
        Initializes the connection manager.
        
        Args:
            config_path (str): Path to the .sumocfg file.
            gui (bool): If True, starts sumo-gui; else starts headless sumo.
            step_length (float): Control step length in seconds.
            label (str): Unique label for the TraCI client connection.
            additional_args (list): List of extra command-line flags to pass to SUMO.
        """
        self.config_path = os.path.abspath(config_path)
        self.gui = gui
        self.step_length = step_length
        self.label = label
        self.additional_args = additional_args or []
        
        self.sumo_binary = None
        self.conn = None
        self.is_connected = False
        
        # Verify SUMO_HOME is set
        self._verify_sumo_home()
        
    def _verify_sumo_home(self):
        """Validates SUMO_HOME environment variable and adds tools path to sys.path."""
        if "SUMO_HOME" not in os.environ:
            raise ImportError(
                "SUMO_HOME environment variable is not defined. "
                "Please install SUMO and configure the system variable."
            )
        
        sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if sumo_tools not in sys.path:
            sys.path.append(sumo_tools)

    def start(self):
        """Starts the SUMO process and connects via TraCI."""
        if self.is_connected:
            return self.conn
            
        # Determine binary (sumo-gui for visualization, sumo for headless runs)
        self.sumo_binary = sumolib.checkBinary("sumo-gui" if self.gui else "sumo")
        
        # Build command-line parameters
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.config_path,
            "--step-length", str(self.step_length)
        ]
        
        # Append default safe options for RL training
        default_args = [
            "--no-warnings", "true",
            "--quit-on-end", "true",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--collision.action", "warn",
            "--random", "true"
        ]
        if self.gui:
            default_args.append("--start")
        
        # Extend command
        sumo_cmd.extend(default_args)
        sumo_cmd.extend(self.additional_args)
        
        # traci.start spawns the SUMO subprocess and registers the TCP connection socket
        traci.start(sumo_cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
        self.is_connected = True
        return self.conn

    def reset(self):
        """
        Resets the simulation to the beginning. Restarts the subprocess in GUI mode for stability,
        or uses traci.load in headless mode for performance.
        """
        if not self.is_connected:
            return self.start()
            
        if self.gui:
            self.close()
            return self.start()
            
        # Reloading the configuration restarts the simulation cleanly in headless mode
        load_cmd = [
            "-c", self.config_path,
            "--step-length", str(self.step_length)
        ]
        load_cmd.extend([
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--random", "true"
        ])
        load_cmd.extend(self.additional_args)
        
        # conn.load re-initializes the active network simulation state
        self.conn.load(load_cmd)
        return self.conn

    def step(self):
        """Advances the simulation by one control step."""
        if not self.is_connected:
            raise RuntimeError("TraCI connection is not active. Call start() before stepping.")
        # conn.simulationStep tells SUMO to calculate the next step_length of vehicle kinematics
        self.conn.simulationStep()

    def close(self):
        """Safely closes the TraCI connection and terminates the SUMO process."""
        if self.is_connected:
            try:
                # conn.close closes the connection socket and terminates the SUMO subprocess
                self.conn.close()
            except traci.exceptions.FatalTraCIError:
                # Process might have already terminated
                pass
            self.is_connected = False
            self.conn = None
            # Allow time for OS to release the socket and SUMO process to clean up
            time.sleep(1.5)

    # --- VEHICLE STATE RETRIEVAL HELPERS ---

    def get_vehicle_speed(self, veh_id):
        """Gets speed of vehicle in m/s using traci.vehicle.getSpeed."""
        # traci.vehicle.getSpeed returns the current speed of the vehicle in the simulation step
        return self.conn.vehicle.getSpeed(veh_id)

    def get_vehicle_position(self, veh_id):
        """Gets 2D (x, y) coordinates of vehicle using traci.vehicle.getPosition."""
        # traci.vehicle.getPosition returns absolute Cartesian coordinates (x, y) of the vehicle
        return self.conn.vehicle.getPosition(veh_id)

    def get_lane_position(self, veh_id):
        """Gets longitudinal position along its current lane using traci.vehicle.getLanePosition."""
        # traci.vehicle.getLanePosition returns the distance from the start of the lane to the vehicle front bumper
        return self.conn.vehicle.getLanePosition(veh_id)

    def get_lane_id(self, veh_id):
        """Gets the ID of the current lane the vehicle is on using traci.vehicle.getLaneID."""
        # traci.vehicle.getLaneID returns the string identifier of the lane currently occupied
        return self.conn.vehicle.getLaneID(veh_id)

    def get_distance_traveled(self, veh_id):
        """Gets the total distance traveled by the vehicle in meters using traci.vehicle.getDistance."""
        # traci.vehicle.getDistance returns the absolute odometer distance the vehicle has moved since spawning
        return self.conn.vehicle.getDistance(veh_id)

    # --- VEHICLE CONTROL HELPERS ---

    def set_vehicle_speed(self, veh_id, speed):
        """Sets target speed of vehicle in m/s using traci.vehicle.setSpeed."""
        # traci.vehicle.setSpeed forces the vehicle to travel at this target speed on the next simulation step
        self.conn.vehicle.setSpeed(veh_id, speed)

    def apply_acceleration(self, veh_id, accel):
        """
        Applies an acceleration command by computing the target speed for the next time step.
        Target Speed = Current Speed + (Acceleration * Time Step duration)
        """
        current_speed = self.get_vehicle_speed(veh_id)
        # Compute Euler integration speed projection
        target_speed = max(0.0, current_speed + accel * self.step_length)
        self.set_vehicle_speed(veh_id, target_speed)

    # Context Manager interface for safe syntax
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
