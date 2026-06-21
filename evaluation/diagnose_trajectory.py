import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def run_diagnostics(model_path, num_episodes=20):
    print(f"Loading PPO model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        sys.exit(1)
        
    model = PPO.load(model_path)
    env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
    
    # Aggregated metrics for final report
    entry_time_list = []
    circulating_time_list = []
    exit_time_list = []
    total_time_list = []
    time_in_approach_list = []
    time_in_merge_list = []
    reached_merge_zone_count = 0
    reasons = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0}
    reached_entry_count = 0
    entered_circ_count = 0
    reached_exit_count = 0
    max_distances = []
    
    print("\n=====================================================================")
    print("                STARTING TRAJECTORY DIAGNOSTIC")
    print("=====================================================================\n")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        # Telemetry logs
        max_dist = 0.0
        reached_entry = False
        entered_circ = False
        reached_exit = False
        
        entry_steps = 0
        circulating_steps = 0
        exit_steps = 0
        
        print(f"--- EPISODE {ep + 1} START ---")
        print(f"{'Step':<5} | {'Dist to Entry':<13} | {'Ego Speed':<9} | {'Ego Lane ID':<15} | {'Gap Size':<8} | {'Action':<6}")
        print("-" * 75)
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Fetch variables from obs and simulator
            # obs structure: [ego_speed, dist_to_entry, nearest_circ_dist, nearest_circ_speed, gap_size, hdv_ratio]
            ego_speed = obs[0]
            dist_to_entry = obs[1]
            gap_size = obs[4]
            
            # Use environment conn to fetch current lane and odometer distance
            ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
            if ego_exists:
                ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
                current_dist = env.sim.conn.vehicle.getDistance(env.ego_id)
                max_dist = max(max_dist, current_dist)
            else:
                ego_lane = "none (removed)"
                
            # Track steps per lane region
            if "entry_N" in ego_lane:
                entry_steps += 1
            elif "circ" in ego_lane:
                circulating_steps += 1
                entered_circ = True
            elif "exit_S" in ego_lane:
                exit_steps += 1
                reached_exit = True
                
            # Entry point is reached when the vehicle gets to the end of the entry lane (dist_to_entry <= 1.0)
            # or leaves the entry lane.
            if ego_exists and (dist_to_entry <= 1.0 or not "entry_N" in ego_lane):
                reached_entry = True
                
            # Log telemetry every 10 steps
            if steps % 10 == 0:
                action_val = action[0] if isinstance(action, np.ndarray) else action
                print(f"{steps:<5} | {dist_to_entry:<13.2f} | {ego_speed:<9.2f} | {ego_lane:<15} | {gap_size:<8.2f} | {action_val:+.2f}")
                
        reason = step_info.get("termination_reason", "timeout").upper()
        reasons[reason] += 1
        max_distances.append(max_dist)
        
        if reached_entry:
            reached_entry_count += 1
        if entered_circ:
            entered_circ_count += 1
        if reached_exit:
            reached_exit_count += 1
            
        entry_time_list.append(entry_steps * env.dt)
        circulating_time_list.append(circulating_steps * env.dt)
        exit_time_list.append(exit_steps * env.dt)
        total_time_list.append(steps * env.dt)
        
        time_in_approach_list.append(step_info.get("time_in_approach", 0.0))
        time_in_merge_list.append(step_info.get("time_in_merge", 0.0))
        if step_info.get("reached_merge_zone", False):
            reached_merge_zone_count += 1
        
        # Episode termination summary
        print("-" * 75)
        print(f"Episode {ep + 1} Termination Summary:")
        print(f"  1. Maximum distance traveled:   {max_dist:.2f} meters")
        print(f"  2. Reached entry point:         {'YES' if reached_entry else 'NO'}")
        print(f"  3. Entered circulating lane:    {'YES' if entered_circ else 'NO'}")
        print(f"  4. Reached exit lane:           {'YES' if reached_exit else 'NO'}")
        print(f"  5. Termination reason:          {reason}")
        print(f"  6. Time in APPROACH_ZONE:       {step_info.get('time_in_approach', 0.0):.2f} s")
        print(f"  7. Time in MERGE_ZONE:          {step_info.get('time_in_merge', 0.0):.2f} s")
        print(f"  8. Reached MERGE_ZONE:          {'YES' if step_info.get('reached_merge_zone', False) else 'NO'}")
        print(f"--- EPISODE {ep + 1} END ---\n")
        
    env.close()
    
    # Print Aggregated Diagnostic Telemetry
    avg_entry_time = np.mean(entry_time_list)
    avg_circ_time = np.mean(circulating_time_list)
    avg_exit_time = np.mean(exit_time_list)
    avg_total_time = np.mean(total_time_list)
    
    avg_time_approach = np.mean(time_in_approach_list)
    avg_time_merge = np.mean(time_in_merge_list)
    pct_reached_merge = (reached_merge_zone_count / num_episodes) * 100

    print("=====================================================================")
    print("                AGGREGATED DIAGNOSTIC SUMMARY")
    print("=====================================================================")
    print(f"Outcomes: {reasons}")
    print(f"Reached Entry Rate:           {(reached_entry_count/num_episodes)*100:.1f}%")
    print(f"Entered Circulating Rate:      {(entered_circ_count/num_episodes)*100:.1f}%")
    print(f"Reached Exit Rate:             {(reached_exit_count/num_episodes)*100:.1f}%")
    print(f"Reached MERGE_ZONE Rate:       {pct_reached_merge:.1f}%")
    print(f"Average Max Distance Traveled: {np.mean(max_distances):.2f} m")
    print("-" * 75)
    print(f"Average Time Spent on Entry Road:       {avg_entry_time:.2f} s ({(avg_entry_time/avg_total_time)*100:.1f}%)")
    print(f"Average Time Spent in Roundabout Ring:  {avg_circ_time:.2f} s ({(avg_circ_time/avg_total_time)*100:.1f}%)")
    print(f"Average Time Spent on Exit Road:        {avg_exit_time:.2f} s ({(avg_exit_time/avg_total_time)*100:.1f}%)")
    print("-" * 75)
    print(f"Average Time Spent in APPROACH_ZONE:    {avg_time_approach:.2f} s ({(avg_time_approach/avg_total_time)*100:.1f}%)")
    print(f"Average Time Spent in MERGE_ZONE:       {avg_time_merge:.2f} s ({(avg_time_merge/avg_total_time)*100:.1f}%)")
    print(f"Average Total Episode Time:             {avg_total_time:.2f} s")
    print("=====================================================================\n")

if __name__ == "__main__":
    # Use the curriculum model we trained
    run_diagnostics("results/models/final_agent_b_curriculum.zip")
