import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

from env.traci_connection import TraCIConnection
from curriculum.curriculum_manager import CurriculumManager
from curriculum.spatial_curriculum_manager import SpatialCurriculumManager

class RoundaboutEnv(gym.Env):
    """
    Custom Gymnasium Environment for Autonomous Vehicle merging in a mixed-autonomy roundabout.
    Includes safe termination checking and diagnostic logging.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config_file="sumo_network/roundabout.sumocfg", gui=False, max_steps=800, 
                 target_success_rate=0.85, curriculum_window=100, fixed_hdv_ratio=None,
                 use_spatial_curriculum=False, spatial_target_success_rate=0.80, spatial_window_size=50,
                 fixed_spawn_distance=None, label=None, use_context_aware=True, use_gap_reward=True, 
                 verbose=True, traffic_density="medium", warmup_steps=200):
        super(RoundaboutEnv, self).__init__()
        
        self.verbose = verbose
        self.traffic_density = traffic_density
        self.warmup_steps = warmup_steps
        
        self.use_context_aware = use_context_aware
        self.use_gap_reward = use_gap_reward
        
        self.config_file = config_file
        self.gui = gui
        self.max_steps = max_steps
        self.fixed_hdv_ratio = fixed_hdv_ratio
        self.step_count = 0
        
        # Cache for geometry lengths read from TraCI dynamically
        self.entry_lengths = {}
        
        # Ego vehicle parameters
        self.ego_id = "ego"
        self.route_id = "r_N_S"  # Ego spawns on North entry, exits at South
        self.ego_max_speed = 13.89  # m/s (approx. 50 km/h)
        self.dt = 0.1  # Control time step in seconds
        self.last_accel = 0.0  # Kept to compute Jerk
        
        # Accumulators for reward components
        self.ep_progress_reward = 0.0
        self.ep_jerk_penalty = 0.0
        self.ep_timeout_penalty = 0.0
        self.ep_success_reward = 0.0
        self.ep_collision_penalty = 0.0
        self.ep_total_reward = 0.0
        self.prev_dist_to_entry = 170.0
        self.ep_time_in_approach = 0.0
        self.ep_time_in_merge = 0.0
        self.reached_merge_zone = False
        self.attempted_merge = False
        self.merge_success = False
        
        # Spatial Curriculum Settings
        self.use_spatial_curriculum = use_spatial_curriculum
        self.fixed_spawn_distance = fixed_spawn_distance
        if self.use_spatial_curriculum:
            self.spatial_curriculum = SpatialCurriculumManager(
                target_success_rate=spatial_target_success_rate,
                window_size=spatial_window_size
            )
        else:
            self.spatial_curriculum = None
            
        # Initialize Curriculum Manager (only if fixed_hdv_ratio is not specified)
        self.curriculum = CurriculumManager(
            target_success_rate=target_success_rate,
            window_size=curriculum_window
        )
        
        # Resolve traffic density to route file override
        density_map = {
            "low": "sumo_network/routes_low.xml",
            "medium": "sumo_network/routes.xml",
            "high": "sumo_network/routes_high.xml",
            "very_high": "sumo_network/routes_very_high.xml",
            "demo_realistic": "sumo_network/routes_demo_realistic.xml"
        }
        route_file = density_map.get(self.traffic_density.lower(), "sumo_network/routes.xml")
        additional_args = ["--route-files", os.path.abspath(route_file)]
        
        # Initialize TraCI connection manager with a unique instance label
        traci_label = label if label is not None else f"gym_env_client_{fixed_hdv_ratio if fixed_hdv_ratio is not None else 'curr'}_{id(self)}"
        self.sim = TraCIConnection(
            config_path=self.config_file,
            gui=self.gui,
            step_length=self.dt,
            label=traci_label,
            additional_args=additional_args
        )
        
        # Action space: continuous acceleration [-4.0, +2.0] m/s^2
        self.action_space = spaces.Box(
            low=-4.0,
            high=2.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: 6-dimensional compact state space
        # High bounds are set to generous values to adapt to any entry arm length (up to 250m) and gap size (100m)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([15.0, 250.0, 100.0, 15.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.ttc_threshold = 2.0

    @property
    def active_hdv_ratio(self):
        """Returns the currently active HDV ratio, resolving curriculum vs. fixed mode."""
        if self.fixed_hdv_ratio is not None:
            return float(self.fixed_hdv_ratio)
        return self.curriculum.current_hdv_ratio

    @property
    def active_spawn_distance(self):
        """Returns the currently active spawn distance, resolving spatial curriculum vs. fixed mode."""
        if self.use_spatial_curriculum and self.spatial_curriculum is not None:
            return float(self.spatial_curriculum.current_spawn_distance)
        if self.fixed_spawn_distance is None:
            return 170.0  # Safe fallback before dynamic query at reset
        return float(self.fixed_spawn_distance)

    def _apply_dynamic_traffic_types(self):
        """Intercepts departed vehicles and assigns types stochastically using active hdv ratio."""
        departed_vehs = self.sim.conn.simulation.getDepartedIDList()
        ratio = self.active_hdv_ratio
        for veh in departed_vehs:
            if veh != self.ego_id:
                # Remove background vehicles on the ego's entry road only if they pose a collision risk with ego
                try:
                    route = self.sim.conn.vehicle.getRoute(veh)
                    if len(route) > 0 and route[0] == "entry_N":
                        if self.ego_id in self.sim.conn.vehicle.getIDList():
                            ego_lane = self.sim.conn.vehicle.getLaneID(self.ego_id)
                            if "entry_N" in ego_lane:
                                ego_pos = self.sim.conn.vehicle.getLanePosition(self.ego_id)
                                veh_pos = self.sim.conn.vehicle.getLanePosition(veh)
                                # If the background vehicle is in front of ego or too close behind it (within 15m), remove it
                                if veh_pos >= ego_pos - 15.0:
                                    self.sim.conn.vehicle.remove(veh)
                                    continue
                        else:
                            # Ego vehicle doesn't exist yet (during warm-up), so we must remove all departed vehicles
                            # on entry_N to prevent them blocking the ego's future spawn location!
                            self.sim.conn.vehicle.remove(veh)
                            continue
                except traci.exceptions.TraCIException:
                    pass

                if np.random.rand() < ratio:
                    self.sim.conn.vehicle.setType(veh, "passenger_car")
                else:
                    self.sim.conn.vehicle.setType(veh, "av_car")

    def _get_observation(self):
        """Helper to collect environmental observations and compute state variables."""
        ratio = self.active_hdv_ratio
        if self.ego_id not in self.sim.conn.vehicle.getIDList():
            entry_length = self.entry_lengths.get("entry_N_0", 170.0)
            return np.array([0.0, entry_length, 50.0, 0.0, 50.0, ratio], dtype=np.float32)
            
        # 1. Ego Speed
        ego_speed = self.sim.conn.vehicle.getSpeed(self.ego_id)
        
        # 2. Distance to entry
        ego_lane = self.sim.conn.vehicle.getLaneID(self.ego_id)
        ego_pos = self.sim.conn.vehicle.getLanePosition(self.ego_id)
        
        if "entry_N" in ego_lane:
            # Query dynamically if the lane is not yet cached (e.g. lane changed to a sublane)
            if ego_lane not in self.entry_lengths:
                try:
                    self.entry_lengths[ego_lane] = self.sim.conn.lane.getLength(ego_lane)
                except Exception:
                    self.entry_lengths[ego_lane] = 170.0
            entry_length = self.entry_lengths[ego_lane]
            dist_to_entry = max(0.0, entry_length - ego_pos)
        else:
            dist_to_entry = 0.0  # Already merged
            
        # 3 & 4. Nearest approaching circulating vehicle distance and speed
        circ_vehs = self.sim.conn.edge.getLastStepVehicleIDs("circ_E_N")
        circ_length = getattr(self, "circ_length", 36.89)
        
        nearest_circ_dist = 50.0
        nearest_circ_speed = 11.11
        circ_positions = []
        
        for veh in circ_vehs:
            v_pos = self.sim.conn.vehicle.getLanePosition(veh)
            v_speed = self.sim.conn.vehicle.getSpeed(veh)
            dist = max(0.0, circ_length - v_pos)
            circ_positions.append(dist)
            
            if dist < nearest_circ_dist:
                nearest_circ_dist = dist
                nearest_circ_speed = v_speed
                
        # 5. Gap Size in Circulating Traffic
        gap_size = 50.0
        if len(circ_positions) >= 2:
            circ_positions.sort()
            gap_size = min(50.0, circ_positions[1] - circ_positions[0])
        elif len(circ_positions) == 1:
            gap_size = min(50.0, 50.0 - circ_positions[0])
            
        # 6. Current active HDV Ratio
        hdv_ratio = ratio
        
        # Override circulating traffic observation if in APPROACH_ZONE (dist_to_entry > 30 m)
        if self.use_context_aware and dist_to_entry > 30.0:
            nearest_circ_dist = 50.0
            nearest_circ_speed = 0.0
            gap_size = 50.0
        
        obs = np.array([
            ego_speed,
            dist_to_entry,
            nearest_circ_dist,
            nearest_circ_speed,
            gap_size,
            hdv_ratio
        ], dtype=np.float32)
        
        return np.clip(obs, self.observation_space.low, self.observation_space.high)


    def step(self, action):
        """Advances simulation under the commanded acceleration and checks stage updates at episode termination."""
        self.step_count += 1
        
        # Command acceleration to ego vehicle
        accel = float(action[0])
        ego_exists = self.ego_id in self.sim.conn.vehicle.getIDList()
        
        if ego_exists:
            ego_lane = self.sim.conn.vehicle.getLaneID(self.ego_id)
            is_in_ring = any(lane in ego_lane for lane in ["circ_N_W", "circ_W_S", "circ_S_E", "circ_E_N", "exit_S"])
            
            if is_in_ring:
                # Disable PPO control and return control back to SUMO with safety checks enabled
                self.sim.conn.vehicle.setSpeedMode(self.ego_id, 31)
                self.sim.conn.vehicle.setSpeed(self.ego_id, -1)
                
                # Record merge success when first entering a circulating lane
                if any(lane in ego_lane for lane in ["circ_N_W", "circ_W_S", "circ_S_E", "circ_E_N"]):
                    self.merge_success = True
            else:
                # Use PPO actions normally (ego on entry_N)
                current_speed = self.sim.conn.vehicle.getSpeed(self.ego_id)
                target_speed = max(0.0, min(self.ego_max_speed, current_speed + accel * self.dt))
                self.sim.conn.vehicle.setSpeed(self.ego_id, target_speed)
            
        # Advance simulation
        self.sim.step()
        
        # Dynamically classify newly spawned traffic background cars
        self._apply_dynamic_traffic_types()
        
        # Check terminations
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        ego_exists_now = self.ego_id in self.sim.conn.vehicle.getIDList()
        colliding_vehs = self.sim.conn.simulation.getCollidingVehiclesIDList()
        
        # --- SAFE TERMINATION CHECK ORDERING ---
        if ego_exists:
            if self.ego_id in colliding_vehs:
                terminated = True
                info["termination_reason"] = "collision"
            elif not ego_exists_now:
                # Vehicle exited simulation cleanly during step
                terminated = True
                info["termination_reason"] = "success"
            else:
                # Vehicle still exists, check if it has arrived at exit lane
                ego_lane = self.sim.conn.vehicle.getLaneID(self.ego_id)
                if "exit_S" in ego_lane:
                    terminated = True
                    info["termination_reason"] = "success"
                
        if self.step_count >= self.max_steps:
            # Note: A timeout can truncate an episode even if terminated isn't True
            truncated = True
            # If already terminated by success/collision, keep that reason, otherwise timeout
            if "termination_reason" not in info:
                info["termination_reason"] = "timeout"
            
        # Compute rewards
        step_progress_reward = 0.0
        step_jerk_penalty = 0.0
        step_timeout_penalty = 0.0
        step_success_reward = 0.0
        step_collision_penalty = 0.0

        term_reason = info.get("termination_reason")
        
        if self.use_gap_reward:
            # Shaped reward formulation (incorporates comfort acceleration, lower jerk penalty)
            if term_reason == "collision":
                step_collision_penalty = -200.0
                reward = step_collision_penalty
            elif term_reason == "success":
                step_success_reward = 100.0
                reward = step_success_reward
            elif term_reason == "timeout":
                step_timeout_penalty = -300.0
                reward = step_timeout_penalty
            else:
                if ego_exists_now:
                    ego_speed = self.sim.conn.vehicle.getSpeed(self.ego_id)
                    r_wait = -0.5 * (1.0 - (ego_speed / self.ego_max_speed))
                    jerk = (accel - self.last_accel) / self.dt
                    step_jerk_penalty = -0.0001 * (jerk ** 2)
                    self.last_accel = accel
                    r_comfort = -0.1 * abs(min(0.0, accel))
                    reward = r_wait + step_jerk_penalty + r_comfort
                else:
                    reward = -0.5
        else:
            # Baseline (pre-shaping) reward formulation: larger jerk penalty, no timeout penalty
            if term_reason == "collision":
                step_collision_penalty = -200.0
                reward = step_collision_penalty
            elif term_reason == "success":
                step_success_reward = 100.0
                reward = step_success_reward
            elif term_reason == "timeout":
                step_timeout_penalty = 0.0
                reward = step_timeout_penalty
            else:
                if ego_exists_now:
                    ego_speed = self.sim.conn.vehicle.getSpeed(self.ego_id)
                    r_wait = -0.5 * (1.0 - (ego_speed / self.ego_max_speed))
                    jerk = (accel - self.last_accel) / self.dt
                    step_jerk_penalty = -0.01 * (jerk ** 2)
                    self.last_accel = accel
                    r_comfort = -0.1 * abs(min(0.0, accel))
                    reward = r_wait + step_jerk_penalty + r_comfort
                else:
                    reward = -0.5

        # Get observation now to compute progress
        obs = self._get_observation()
        current_dist_to_entry = obs[1]
        
        # Track and log zone transitions
        current_zone = "APPROACH_ZONE" if current_dist_to_entry > 30.0 else "MERGE_ZONE"
        info["zone"] = current_zone
        
        if ego_exists_now:
            if self.last_zone != current_zone:
                print(f"[ZONE] Ego entered {current_zone}")
                self.last_zone = current_zone
                
            if current_zone == "APPROACH_ZONE":
                self.ep_time_in_approach += self.dt
            else:
                self.ep_time_in_merge += self.dt
                self.reached_merge_zone = True
                
            # Check for merge attempt
            ego_lane = self.sim.conn.vehicle.getLaneID(self.ego_id)
            if current_dist_to_entry <= 1.0 or not "entry_N" in ego_lane:
                self.attempted_merge = True
            if "circ" in ego_lane:
                self.entered_circulating = True
        
        # Add progress reward (only for shaped reward)
        if self.use_gap_reward:
            progress = self.prev_dist_to_entry - current_dist_to_entry
            step_progress_reward = 0.1 * progress
            reward += step_progress_reward
        
        # Update prev_dist_to_entry
        self.prev_dist_to_entry = current_dist_to_entry
        
        # Accumulate components
        self.ep_progress_reward += step_progress_reward
        self.ep_jerk_penalty += step_jerk_penalty
        self.ep_timeout_penalty += step_timeout_penalty
        self.ep_success_reward += step_success_reward
        self.ep_collision_penalty += step_collision_penalty
        self.ep_total_reward += reward

        # Populate debug telemetry
        if ego_exists_now:
            info["speed"] = self.sim.conn.vehicle.getSpeed(self.ego_id)
            info["position"] = self.sim.conn.vehicle.getLanePosition(self.ego_id)
            info["merge_success"] = self.merge_success
            
        # --- CURRICULUM UPDATE & LOGGING ---
        if terminated or truncated:
            reason = info.get("termination_reason", "timeout").upper()
            success = (reason == "SUCCESS")
            info["success"] = success
            info["collision"] = (reason == "COLLISION")
            info["timeout"] = (reason == "TIMEOUT")
            
            if self.verbose:
                # Print diagnostic outcome log
                print(f"[{'CURRICULUM' if self.fixed_hdv_ratio is None else 'FIXED_ENV'}] EPISODE END OUTCOME: {reason}")
                
                # Print reward breakdown report
                print(f"\n=========================================")
                print(f"       EPISODE {self.step_count} REWARD BREAKDOWN")
                print(f"       Outcome: {reason}")
                print(f"=========================================")
                print(f"  Progress Reward:      {self.ep_progress_reward:+.4f}")
                print(f"  Jerk Penalty:         {self.ep_jerk_penalty:+.4f}")
                print(f"  Timeout Penalty:      {self.ep_timeout_penalty:+.4f}")
                print(f"  Success Reward:       {self.ep_success_reward:+.4f}")
                print(f"  Collision Penalty:    {self.ep_collision_penalty:+.4f}")
                print(f"-----------------------------------------")
                print(f"  Total Episode Reward: {self.ep_total_reward:+.4f}")
                print(f"-----------------------------------------")
                print(f"  Time in APPROACH_ZONE:{self.ep_time_in_approach:.2f} s")
                print(f"  Time in MERGE_ZONE:   {self.ep_time_in_merge:.2f} s")
                print(f"  Reached MERGE_ZONE:   {'YES' if self.reached_merge_zone else 'NO'}")
                print(f"=========================================\n")

            # Add reward component metrics to info dict for logging in callback
            info["episode_progress_reward"] = self.ep_progress_reward
            info["episode_jerk_penalty"] = self.ep_jerk_penalty
            info["episode_timeout_penalty"] = self.ep_timeout_penalty
            info["episode_success_reward"] = self.ep_success_reward
            info["episode_collision_penalty"] = self.ep_collision_penalty
            info["episode_total_reward"] = self.ep_total_reward
            info["time_in_approach"] = self.ep_time_in_approach
            info["time_in_merge"] = self.ep_time_in_merge
            info["reached_merge_zone"] = self.reached_merge_zone
            info["attempted_merge"] = self.attempted_merge
            
            # Only update curriculum metrics in curriculum learning mode
            if self.fixed_hdv_ratio is None:
                self.curriculum.update_metrics(success)
                info["curriculum_stage"] = self.curriculum.current_stage
                info["hdv_ratio"] = self.curriculum.current_hdv_ratio
            else:
                info["curriculum_stage"] = -1
                info["hdv_ratio"] = self.fixed_hdv_ratio

            # Update spatial curriculum metrics if enabled
            if self.use_spatial_curriculum and self.spatial_curriculum is not None:
                is_collision = (reason == "COLLISION")
                self.spatial_curriculum.update_metrics(
                    success=success,
                    reached_merge=self.reached_merge_zone,
                    attempted_merge=self.attempted_merge,
                    collision=is_collision
                )
                info["spatial_stage"] = self.spatial_curriculum.current_stage
                info["spawn_distance"] = self.spatial_curriculum.current_spawn_distance
                info["rolling_merge_reach_rate"] = self.spatial_curriculum.rolling_merge_reach_rate
                info["rolling_merge_attempt_rate"] = self.spatial_curriculum.rolling_merge_attempt_rate
                info["rolling_collision_rate"] = self.spatial_curriculum.rolling_collision_rate
            else:
                info["spatial_stage"] = -1
                info["spawn_distance"] = self.fixed_spawn_distance
                info["rolling_merge_reach_rate"] = 0.0
                info["rolling_merge_attempt_rate"] = 0.0
                info["rolling_collision_rate"] = 0.0
                
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets environment and configures the start of a new episode."""
        super().reset(seed=seed)
        self.step_count = 0
        self.last_accel = 0.0
        
        # Reset reward component accumulators
        self.ep_progress_reward = 0.0
        self.ep_jerk_penalty = 0.0
        self.ep_timeout_penalty = 0.0
        self.ep_success_reward = 0.0
        self.ep_collision_penalty = 0.0
        self.ep_total_reward = 0.0
        self.ep_time_in_approach = 0.0
        self.ep_time_in_merge = 0.0
        self.reached_merge_zone = False
        self.attempted_merge = False
        self.entered_circulating = False
        self.merge_success = False
        self.last_zone = None
        
        self.sim.reset()
        
        # Query and cache lane lengths dynamically using TraCI
        for arm in ["N", "S", "E", "W"]:
            lane_id = f"entry_{arm}_0"
            try:
                self.entry_lengths[lane_id] = self.sim.conn.lane.getLength(lane_id)
            except Exception:
                self.entry_lengths[lane_id] = 170.0
                
        try:
            self.circ_length = self.sim.conn.lane.getLength("circ_E_N_0")
        except Exception:
            self.circ_length = 36.89
                
        # Resolve fixed spawn distance to full entry length if None
        entry_length = self.entry_lengths.get("entry_N_0", 170.0)
        if self.fixed_spawn_distance is None:
            self.fixed_spawn_distance = entry_length
            
        # Update spatial curriculum Stage D to the actual entry length
        if self.use_spatial_curriculum and self.spatial_curriculum is not None:
            self.spatial_curriculum.update_max_spawn_distance(entry_length)
        
        # Run warmup period to populate the roundabout and entry/exit arms with background traffic
        for _ in range(self.warmup_steps):
            self.sim.step()
            self._apply_dynamic_traffic_types()
        
        # Safe cleanup: Check if the vehicle already exists and remove it.
        # This acts as a robust fallback for duplicate ID add crashes.
        if self.ego_id in self.sim.conn.vehicle.getIDList():
            try:
                self.sim.conn.vehicle.remove(self.ego_id)
                self.sim.step()  # Process deletion step
            except traci.exceptions.TraCIException:
                pass
        
        # Spawn ego vehicle dynamically using actual entry length
        depart_pos = max(0.0, entry_length - self.active_spawn_distance)
        self.sim.conn.vehicle.add(
            vehID=self.ego_id,
            routeID=self.route_id,
            typeID="ego_av",
            departLane="0",
            departPos=f"{depart_pos}",
            departSpeed="0.0"
        )
        self.sim.conn.vehicle.setSpeedMode(self.ego_id, 0)
        self.sim.conn.vehicle.setLaneChangeMode(self.ego_id, 0)
        
        # Step once to insert and spawn background vehicles
        self.sim.step()
        self._apply_dynamic_traffic_types()
        
        obs = self._get_observation()
        self.prev_dist_to_entry = obs[1]
        info = {"speed": 0.0, "position": 0.0}
        
        return obs, info

    def close(self):
        """Cleans up the SUMO process."""
        self.sim.close()

    def _get_ttc_after_merge(self):
        """
        Calculates Time-To-Collision (TTC) between the ego vehicle and the nearest
        interacting vehicle (either leader in front or circulating vehicle approaching behind).
        """
        if self.ego_id not in self.sim.conn.vehicle.getIDList():
            return float('inf')
            
        ego_speed = self.sim.conn.vehicle.getSpeed(self.ego_id)
        ego_lane = self.sim.conn.vehicle.getLaneID(self.ego_id)
        
        # 1. Leader-based TTC (ego approaching a vehicle in front)
        try:
            leader_info = self.sim.conn.vehicle.getLeader(self.ego_id, dist=100.0)
            if leader_info is not None:
                leader_id, gap = leader_info
                if leader_id:
                    leader_speed = self.sim.conn.vehicle.getSpeed(leader_id)
                    speed_diff = ego_speed - leader_speed
                    if speed_diff > 0.1:
                        ttc_leader = gap / speed_diff
                        return ttc_leader
        except Exception:
            pass
            
        # 2. Follower-based / Merge-based TTC (circulating vehicle approaching ego from behind)
        try:
            obs = self._get_observation()
            dist_to_entry = obs[1]
            nearest_circ_dist = obs[2]
            nearest_circ_speed = obs[3]
            
            if nearest_circ_dist < 50.0:
                if "circ" in ego_lane:
                    ego_pos = self.sim.conn.vehicle.getLanePosition(self.ego_id)
                    gap = nearest_circ_dist + ego_pos
                else:
                    gap = nearest_circ_dist + dist_to_entry
                    
                speed_diff = nearest_circ_speed - ego_speed
                if speed_diff > 0.1:
                    ttc_follower = gap / speed_diff
                    return ttc_follower
        except Exception:
            pass
            
        return float('inf')
