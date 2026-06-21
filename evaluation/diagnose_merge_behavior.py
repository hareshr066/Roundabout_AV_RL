import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def run_merge_diagnostics(model_path, num_episodes=50):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        sys.exit(1)
        
    model = PPO.load(model_path)
    env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
    
    # Aggregated metrics for episodes that reach the MERGE_ZONE
    episodes_reaching_merge = 0
    time_in_merge_list = []
    speeds_in_merge_list = []
    actions_in_merge_list = []
    merge_attempts_count = 0
    enters_circulating_count = 0
    
    first_entry_records = []
    merge_outcomes = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0}
    all_outcomes = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0}
    
    print("\n=====================================================================")
    print("             STARTING MERGE_ZONE BEHAVIOR DIAGNOSTIC")
    print("=====================================================================\n")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        # Episode-specific tracking variables
        reached_merge_zone = False
        first_entry_data = None
        merge_zone_speeds = []
        merge_zone_actions = []
        attempted_merge = False
        entered_circulating = False
        merge_zone_steps = 0
        
        print(f"--- EPISODE {ep + 1} ---")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            # Use environment conn to fetch current lane before step
            ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
            if ego_exists:
                ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
            else:
                ego_lane = "none"
                
            dist_to_entry = obs[1]
            
            # Check if entering MERGE_ZONE for the first time
            if ego_exists and dist_to_entry <= 30.0:
                if not reached_merge_zone:
                    reached_merge_zone = True
                    episodes_reaching_merge += 1
                    first_entry_data = {
                        "episode": ep + 1,
                        "distance_to_entry": dist_to_entry,
                        "ego_speed": obs[0],
                        "nearest_circ_dist": obs[2],
                        "nearest_circ_speed": obs[3],
                        "gap_size": obs[4],
                        "action": action[0]
                    }
                    first_entry_records.append(first_entry_data)
                    print("\n>>> FIRST ENTERED MERGE_ZONE <<<")
                    print(f"  Distance to Entry: {dist_to_entry:.2f} m")
                    print(f"  Ego Speed:         {obs[0]:.2f} m/s")
                    print(f"  Nearest Circ Dist: {obs[2]:.2f} m")
                    print(f"  Nearest Circ Spd:  {obs[3]:.2f} m/s")
                    print(f"  Gap Size:          {obs[4]:.2f} m")
                    print(f"  Action Selected:   {action[0]:+.2f} m/s^2")
                    print("-" * 80)
                    print(f"{'Step':<5} | {'Dist to Entry':<13} | {'Ego Speed':<9} | {'Ego Lane ID':<15} | {'Nearest Circ Dist':<17} | {'Action':<6}")
                    print("-" * 80)
                
                # We are in the MERGE_ZONE, log details and record metrics
                merge_zone_steps += 1
                merge_zone_speeds.append(obs[0])
                merge_zone_actions.append(action[0])
                
                # Check for merge attempt (passed the entry point / line)
                if dist_to_entry <= 1.0 or not "entry_N" in ego_lane:
                    attempted_merge = True
                
                # Check if in circulating lane
                if "circ" in ego_lane:
                    entered_circulating = True
                
                # Telemetry print inside MERGE_ZONE
                print(f"{steps:<5} | {dist_to_entry:<13.2f} | {obs[0]:<9.2f} | {ego_lane:<15} | {obs[2]:<17.2f} | {action[0]:+.2f}")
                
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        reason = step_info.get("termination_reason", "timeout").upper()
        all_outcomes[reason] += 1
        
        if reached_merge_zone:
            merge_outcomes[reason] += 1
            time_in_merge_list.append(merge_zone_steps * env.dt)
            if merge_zone_speeds:
                speeds_in_merge_list.append(np.mean(merge_zone_speeds))
            if merge_zone_actions:
                actions_in_merge_list.append(np.mean(merge_zone_actions))
            if attempted_merge:
                merge_attempts_count += 1
            if entered_circulating:
                enters_circulating_count += 1
                
            print("-" * 80)
            print(f"Episode {ep + 1} MERGE_ZONE Summary:")
            print(f"  Time spent in MERGE_ZONE: {merge_zone_steps * env.dt:.2f} s")
            print(f"  Average speed in MERGE:   {np.mean(merge_zone_speeds) if merge_zone_speeds else 0.0:.2f} m/s")
            print(f"  Average action in MERGE:  {np.mean(merge_zone_actions) if merge_zone_actions else 0.0:+.2f} m/s^2")
            print(f"  Attempted Merge:          {'YES' if attempted_merge else 'NO'}")
            print(f"  Entered Circulating:      {'YES' if entered_circulating else 'NO'}")
            print(f"  Outcome:                  {reason}")
            print(f"--- EPISODE {ep + 1} END ---\n")
        else:
            print("  Ego never entered MERGE_ZONE during this episode.")
            print(f"  Outcome:                  {reason}")
            print(f"--- EPISODE {ep + 1} END ---\n")
            
    env.close()
    
    # Process aggregated metrics
    pct_reach_merge = (episodes_reaching_merge / num_episodes) * 100
    avg_time_in_merge = np.mean(time_in_merge_list) if time_in_merge_list else 0.0
    avg_speed_in_merge = np.mean(speeds_in_merge_list) if speeds_in_merge_list else 0.0
    avg_action_in_merge = np.mean(actions_in_merge_list) if actions_in_merge_list else 0.0
    
    # Diagnose failure mode
    diagnose_failure_mode(
        episodes_reaching_merge=episodes_reaching_merge,
        merge_outcomes=merge_outcomes,
        avg_speed_in_merge=avg_speed_in_merge,
        avg_action_in_merge=avg_action_in_merge,
        merge_attempts_count=merge_attempts_count,
        enters_circulating_count=enters_circulating_count
    )
    
    print("=====================================================================")
    print("             MERGE_ZONE AGGREGATED DIAGNOSTIC SUMMARY")
    print("=====================================================================")
    print(f"Total Episodes Run:                  {num_episodes}")
    print(f"Episodes Reaching MERGE_ZONE:        {episodes_reaching_merge} ({pct_reach_merge:.1f}%)")
    print(f"All Episode Outcomes:                {all_outcomes}")
    print(f"MERGE_ZONE Reached Outcomes:         {merge_outcomes}")
    print("-" * 69)
    print(f"1. Average Time spent in MERGE_ZONE: {avg_time_in_merge:.2f} s")
    print(f"2. Average Speed inside MERGE_ZONE:  {avg_speed_in_merge:.2f} m/s")
    print(f"3. Average Action inside MERGE_ZONE: {avg_action_in_merge:+.2f} m/s^2")
    print(f"4. Number of Merge Attempts:         {merge_attempts_count}")
    print(f"5. Enters Circulating Lane:          {enters_circulating_count} (out of {episodes_reaching_merge} episodes)")
    print("=====================================================================\n")
    
    # Save a report
    save_report_artifact(
        num_episodes=num_episodes,
        episodes_reaching_merge=episodes_reaching_merge,
        pct_reach_merge=pct_reach_merge,
        all_outcomes=all_outcomes,
        merge_outcomes=merge_outcomes,
        avg_time_in_merge=avg_time_in_merge,
        avg_speed_in_merge=avg_speed_in_merge,
        avg_action_in_merge=avg_action_in_merge,
        merge_attempts_count=merge_attempts_count,
        enters_circulating_count=enters_circulating_count,
        first_entry_records=first_entry_records
    )

def diagnose_failure_mode(episodes_reaching_merge, merge_outcomes, avg_speed_in_merge, 
                          avg_action_in_merge, merge_attempts_count, enters_circulating_count):
    if episodes_reaching_merge == 0:
        print("DIAGNOSIS: Ego never reached the MERGE_ZONE. The bottleneck is located before the MERGE_ZONE.")
        return
        
    num_collisions = merge_outcomes.get("COLLISION", 0)
    num_timeouts = merge_outcomes.get("TIMEOUT", 0)
    num_successes = merge_outcomes.get("SUCCESS", 0)
    
    print("---------------------------------------------------------------------")
    print("                  BOTTLENECK CAUSE ANALYSIS")
    print("---------------------------------------------------------------------")
    
    # Diagnostic rules:
    # A. Excessive caution: high timeout rate, low average speed in merge, 0 or very low merge attempts.
    # B. Gap acceptance failure: high collision rate, multiple merge attempts but colliding.
    # C. Inability to accelerate: some merge attempts or attempts entering circulating lane, but high timeouts, positive actions but speed doesn't increase.
    # D. Environment logic issue: collisions with no clear traffic or weird vehicle disappearing/spawning issues.
    
    reasons = []
    
    if num_timeouts > 0 and merge_attempts_count == 0:
        reasons.append((
            "A. Excessive caution",
            f"The ego vehicle timed out in {num_timeouts} episodes without making any merge attempts. "
            f"Average speed in the MERGE_ZONE was {avg_speed_in_merge:.2f} m/s and average action was {avg_action_in_merge:+.2f} m/s^2, "
            f"indicating it remains stationary at the yield line."
        ))
        
    if num_collisions > 0:
        reasons.append((
            "B. Gap acceptance failure",
            f"The ego vehicle attempted to merge and resulted in {num_collisions} collisions, "
            f"indicating that it accepts unsafe/too small gaps in the circulating traffic flow."
        ))
        
    if num_timeouts > 0 and merge_attempts_count > 0:
        reasons.append((
            "C. Inability to accelerate",
            f"The ego vehicle attempted to merge in {merge_attempts_count} episodes but timed out anyway. "
            f"It spent an average of {avg_speed_in_merge:.2f} m/s speed inside the zone, indicating it starts moving but fails to merge in time."
        ))
        
    # Check for TraCI environment/physics abnormalities
    if num_collisions > 0 and enters_circulating_count == 0:
        reasons.append((
            "D. Environment logic issue",
            "Collisions occurred before the ego vehicle could enter the circulating lane, "
            "which might point to invalid spawning/teleporting bugs or rear-end collisions by background traffic."
        ))

    if not reasons:
        if num_successes == episodes_reaching_merge:
            print("DIAGNOSIS: Merge behavior is successful! No bottleneck detected inside the MERGE_ZONE.")
        else:
            print("DIAGNOSIS: Undetermined failure mode. Review individual episode telemetry logs.")
    else:
        print("Detected Potential Bottlenecks:")
        for code, desc in reasons:
            print(f"  * {code}: {desc}")
        print("\nPrimary Diagnosis: " + reasons[0][0])
    print("---------------------------------------------------------------------")

def save_report_artifact(num_episodes, episodes_reaching_merge, pct_reach_merge, 
                         all_outcomes, merge_outcomes, avg_time_in_merge, 
                         avg_speed_in_merge, avg_action_in_merge, merge_attempts_count, 
                         enters_circulating_count, first_entry_records):
    # Determine the bottleneck diagnosis text
    num_collisions = merge_outcomes.get("COLLISION", 0)
    num_timeouts = merge_outcomes.get("TIMEOUT", 0)
    num_successes = merge_outcomes.get("SUCCESS", 0)
    
    if episodes_reaching_merge == 0:
        diagnosis_title = "Bottleneck: Pre-Merge Zone Spawning/Approach Failure"
        diagnosis_text = "The ego vehicle never reached the MERGE_ZONE. Check approach-phase policy or spawning logic."
    elif num_timeouts > 0 and merge_attempts_count == 0:
        diagnosis_title = "Bottleneck: A. Excessive caution"
        diagnosis_text = "The ego vehicle times out at the yield line without attempting to merge, indicating it yields excessively to circulating vehicles."
    elif num_collisions > 0:
        diagnosis_title = "Bottleneck: B. Gap acceptance failure"
        diagnosis_text = "The ego vehicle attempts to merge but collides, indicating it accepts gaps that are too small."
    elif num_timeouts > 0 and merge_attempts_count > 0:
        diagnosis_title = "Bottleneck: C. Inability to accelerate"
        diagnosis_text = "The ego vehicle attempts to merge but cannot accelerate quickly enough, timing out inside the roundabout."
    else:
        diagnosis_title = "Successful Merge Behavior"
        diagnosis_text = "The agent successfully merged into the roundabout ring."
        
    report_content = f"""# MERGE_ZONE Behavior Diagnostic Report

This report evaluates the detailed behavior of the ego vehicle after entering the **MERGE_ZONE** (distance to entry <= 30 m) under 50 evaluation episodes.

---

## 1. Aggregated Metrics Summary

| Metric | Value |
| :--- | :--- |
| **Total Evaluation Episodes** | {num_episodes} |
| **Episodes Reaching MERGE_ZONE** | {episodes_reaching_merge} ({pct_reach_merge:.1f}%) |
| **All Episode Outcomes** | {all_outcomes} |
| **MERGE_ZONE Reached Outcomes** | {merge_outcomes} |
| **Average Time in MERGE_ZONE** | {avg_time_in_merge:.2f} s |
| **Average Speed in MERGE_ZONE** | {avg_speed_in_merge:.2f} m/s |
| **Average Action in MERGE_ZONE** | {avg_action_in_merge:+.2f} m/s² |
| **Number of Merge Attempts** | {merge_attempts_count} |
| **Entered Circulating Lane** | {enters_circulating_count} |

---

## 2. Bottleneck Diagnosis
> ### **{diagnosis_title}**
> {diagnosis_text}

---

## 3. First Entry States (Samples)
Below is the telemetry recorded at the exact moment the ego vehicle entered the **MERGE_ZONE** (first 5 samples):

| Episode | Dist to Entry (m) | Ego Speed (m/s) | Nearest Circ Dist (m) | Nearest Circ Spd (m/s) | Gap Size (m) | Action (m/s²) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rec in first_entry_records[:5]:
        report_content += f"| {rec['episode']} | {rec['distance_to_entry']:.2f} | {rec['ego_speed']:.2f} | {rec['nearest_circ_dist']:.2f} | {rec['nearest_circ_speed']:.2f} | {rec['gap_size']:.2f} | {rec['action']:+.2f} |\n"
        
    report_content += """
---

## 4. Failure Mode Analysis & Recommendations

* **Excessive Caution (A):** If the average speed in the merge zone is near 0 and no attempts are made, check if the reward function penalizes collisions too heavily (causing the agent to prefer timing out over risking a merge) or if the gap observation ranges are too conservative.
* **Gap Acceptance Failure (B):** If collisions are high, consider adding a gap-based penalty in the reward function or introducing safer target gaps.
* **Inability to Accelerate (C):** Check if the acceleration action space limit (currently continuous acceleration [-4.0, +2.0] m/s^2) is too restrictive or if the jerk penalty is overly dominant.
"""

    # We need to save the report to the brain/artifacts path
    # Let's save it to C:\Users\hrato\.gemini\antigravity\brain\c8d491cb-0e6a-4843-afb9-b6394036b304\artifacts\merge_behavior_report.md
    artifact_path = r"C:\Users\hrato\.gemini\antigravity\brain\c8d491cb-0e6a-4843-afb9-b6394036b304\artifacts\merge_behavior_report.md"
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved merge behavior report artifact to: {artifact_path}")

if __name__ == "__main__":
    run_merge_diagnostics("results/models/final_agent_b_curriculum.zip", num_episodes=50)
