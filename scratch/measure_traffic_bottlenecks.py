import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def run_congestion_audit(preset="MEDIUM", num_episodes=3):
    print("=" * 80)
    print(f"      ROUNDABOUT AUDIT: PRESET {preset} ({num_episodes} EPISODES)")
    print("=" * 80)
    
    model_path = "results/models/final_agent_b_curriculum.zip"
    if not os.path.exists(model_path):
        model_path = "results/models/agent_spatial_curriculum_30k.zip"
        
    print(f"Using PPO agent model: {model_path}")
    model = PPO.load(model_path)
    
    # Initialize environment
    env = RoundaboutEnv(
        gui=False,
        traffic_density=preset.lower(),
        fixed_hdv_ratio=0.50,
        fixed_spawn_distance=80.0, # identical to user's demo_mode setting
        max_steps=800,
        label=f"audit_{preset.lower()}_run"
    )
    
    # Metrics to collect
    outcomes = []
    episode_lengths = []
    
    # Queue length trackers per step (speed < 0.1 m/s)
    queues_entry_E = []
    queues_exit_S = []
    queues_circ_W_S = []
    
    # Vehicle count trackers per step
    vehs_entry_E = []
    vehs_exit_S = []
    vehs_circ_W_S = []
    
    # Average waiting time per episode
    all_waiting_times = []
    
    # Bottleneck analysis: record where vehicles are stopped (speed < 0.1 m/s)
    stopped_locations = {}
    
    # Check if background vehicles stop in circulating lanes and block downstream flow
    stopped_circulating_vehicles = []
    
    # Telemetry tracker for timeout episodes
    timeout_details = []
    
    try:
        for ep in range(num_episodes):
            print(f"\n--- Starting Episode {ep+1} ---")
            obs, info = env.reset()
            done = False
            step = 0
            
            # Track waiting times for this episode: veh_id -> steps stopped
            veh_stopped_steps = {}
            
            # Track if ego entered circulating lane and where it is
            ego_positions = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                step += 1
                
                try:
                    # All vehicles in the simulation
                    all_vehs = env.sim.conn.vehicle.getIDList()
                    
                    # Check ego position
                    ego_exists = env.ego_id in all_vehs
                    if ego_exists:
                        ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
                        ego_pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
                        ego_spd = env.sim.conn.vehicle.getSpeed(env.ego_id)
                        ego_positions.append((step, ego_lane, ego_pos, ego_spd))
                    
                    # Queue measurements (speed < 0.1 m/s)
                    q_E = 0
                    q_exit = 0
                    q_circ = 0
                    
                    cnt_E = 0
                    cnt_exit = 0
                    cnt_circ = 0
                    
                    for veh in all_vehs:
                        lane = env.sim.conn.vehicle.getLaneID(veh)
                        speed = env.sim.conn.vehicle.getSpeed(veh)
                        
                        if lane == "entry_E_0":
                            cnt_E += 1
                            if speed < 0.1:
                                q_E += 1
                        elif lane == "exit_S_0":
                            cnt_exit += 1
                            if speed < 0.1:
                                q_exit += 1
                        elif lane == "circ_W_S_0":
                            cnt_circ += 1
                            if speed < 0.1:
                                q_circ += 1
                                
                        # Track stopped vehicles
                        if speed < 0.1:
                            veh_stopped_steps[veh] = veh_stopped_steps.get(veh, 0) + 1
                            stopped_locations[lane] = stopped_locations.get(lane, 0) + 1
                            
                            # Check if background vehicle is stopped on circulating lane
                            if "circ" in lane and veh != env.ego_id:
                                stopped_circulating_vehicles.append({
                                    "step": step,
                                    "veh": veh,
                                    "lane": lane,
                                    "pos": env.sim.conn.vehicle.getLanePosition(veh)
                                })
                                
                    queues_entry_E.append(q_E)
                    queues_exit_S.append(q_exit)
                    queues_circ_W_S.append(q_circ)
                    
                    vehs_entry_E.append(cnt_E)
                    vehs_exit_S.append(cnt_exit)
                    vehs_circ_W_S.append(cnt_circ)
                    
                except Exception as e:
                    print(f"Error retrieving TraCI stats at step {step}: {e}")
                    break
                    
            # End of episode
            outcome = step_info.get("termination_reason", "timeout").upper()
            outcomes.append(outcome)
            episode_lengths.append(step)
            
            # Calculate waiting times for this episode
            ep_waiting_times = [steps * env.dt for steps in veh_stopped_steps.values()]
            if ep_waiting_times:
                all_waiting_times.extend(ep_waiting_times)
                
            print(f"Ep {ep+1:2d} Ended | Outcome: {outcome:<10} | Steps: {step:<4}")
            
            # If timeout, log why
            if outcome == "TIMEOUT":
                # Find last known ego state
                if ego_positions:
                    last_step, last_lane, last_pos, last_spd = ego_positions[-1]
                    timeout_details.append({
                        "episode": ep + 1,
                        "last_step": last_step,
                        "last_lane": last_lane,
                        "last_pos": last_pos,
                        "last_speed": last_spd,
                        "ego_trajectory": ego_positions[-20:] # last 20 steps
                    })
    finally:
        try:
            env.close()
        except Exception:
            pass
            
    # Calculate stats
    avg_q_E = np.mean(queues_entry_E) if queues_entry_E else 0.0
    avg_q_exit = np.mean(queues_exit_S) if queues_exit_S else 0.0
    avg_q_circ = np.mean(queues_circ_W_S) if queues_circ_W_S else 0.0
    
    max_q_E = np.max(queues_entry_E) if queues_entry_E else 0
    max_q_exit = np.max(queues_exit_S) if queues_exit_S else 0
    max_q_circ = np.max(queues_circ_W_S) if queues_circ_W_S else 0
    
    avg_cnt_E = np.mean(vehs_entry_E) if vehs_entry_E else 0.0
    avg_cnt_exit = np.mean(vehs_exit_S) if vehs_exit_S else 0.0
    avg_cnt_circ = np.mean(vehs_circ_W_S) if vehs_circ_W_S else 0.0
    
    avg_wait = np.mean(all_waiting_times) if all_waiting_times else 0.0
    max_wait = np.max(all_waiting_times) if all_waiting_times else 0.0
    
    print("\n" + "=" * 80)
    print("                 AUDIT STATS SUMMARY")
    print("=" * 80)
    print(f"Outcomes: {dict((x, outcomes.count(x)) for x in set(outcomes))}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average Queue Lengths (Speed < 0.1 m/s):")
    print(f"  - entry_E_0:  {avg_q_E:.2f} vehs (Max: {max_q_E}, Avg density: {avg_cnt_E:.2f} vehs)")
    print(f"  - exit_S_0:   {avg_q_exit:.2f} vehs (Max: {max_q_exit}, Avg density: {avg_cnt_exit:.2f} vehs)")
    print(f"  - circ_W_S_0: {avg_q_circ:.2f} vehs (Max: {max_q_circ}, Avg density: {avg_cnt_circ:.2f} vehs)")
    print(f"Average Waiting Time: {avg_wait:.2f} s (Max: {max_wait:.2f} s)")
    
    # Bottleneck analysis
    print("\nTop Stopped Locations (cumulative stopped-vehicle-steps):")
    sorted_stops = sorted(stopped_locations.items(), key=lambda x: x[1], reverse=True)
    for loc, count in sorted_stops[:6]:
        print(f"  - {loc:<15}: {count} veh-steps stopped")
        
    # Check background vehicles stopping in circulating lanes
    print(f"\nBackground vehicles stopped in circulating lanes: {len(stopped_circulating_vehicles)} occurrences")
    if stopped_circulating_vehicles:
        print("Sample occurrences:")
        by_lane = {}
        for occ in stopped_circulating_vehicles:
            by_lane[occ["lane"]] = by_lane.get(occ["lane"], 0) + 1
        for lane, count in by_lane.items():
            print(f"  - Lane {lane}: stopped {count} times")
            
    # Timeout Details
    if timeout_details:
        print("\nTimeout details for Ego vehicle:")
        for td in timeout_details:
            print(f"  Episode {td['episode']} Timeout at Step {td['last_step']}:")
            print(f"    Ego Position: Lane={td['last_lane']}, Pos={td['last_pos']:.2f} m, Speed={td['last_speed']:.2f} m/s")
            print("    Ego Trajectory (last 10 steps):")
            for t_step, t_lane, t_pos, t_spd in td['ego_trajectory'][-10:]:
                print(f"      Step {t_step:3d} | Lane: {t_lane:<15} | Pos: {t_pos:6.2f} m | Speed: {t_spd:5.2f} m/s")
                
    return {
        "outcomes": outcomes,
        "avg_q_E": avg_q_E,
        "avg_q_exit": avg_q_exit,
        "avg_q_circ": avg_q_circ,
        "avg_wait": avg_wait,
        "sorted_stops": sorted_stops,
        "stopped_circ": len(stopped_circulating_vehicles),
        "timeouts": timeout_details
    }

if __name__ == "__main__":
    run_congestion_audit(preset="MEDIUM", num_episodes=3)
