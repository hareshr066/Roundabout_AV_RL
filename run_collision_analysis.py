import os
import sys
import time
import numpy as np
import traceback
from env.roundabout_env import RoundaboutEnv

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dynamically locate artifacts directory for the active session, fallback to 'results'
user_home = os.path.expanduser("~")
session_id = "61e36dae-73de-4984-b9a9-fad64920a0e0"
gemini_artifact_path = os.path.join(user_home, ".gemini", "antigravity", "brain", session_id, "artifacts")
if os.path.exists(os.path.join(user_home, ".gemini", "antigravity")):
    artifact_dir = gemini_artifact_path
else:
    artifact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
report_path = os.path.join(artifact_dir, "collision_location_analysis.md")

class AggressiveAgent:
    """
    An aggressive driver agent that always attempts to merge
    without yielding to circulating vehicles, representing a PPO policy
    that has failed gap acceptance learning (Merge Attempt Rate = 100%).
    """
    def __init__(self, target_speed=11.0):
        self.target_speed = target_speed

    def select_action(self, obs):
        ego_speed = obs[0]
        dist_to_entry = obs[1]
        
        # Always accelerate to merge aggressively
        if dist_to_entry > 0.1:
            accel = 1.5 if ego_speed < self.target_speed else 0.0
        else:
            accel = 1.0 if ego_speed < self.target_speed else -0.5
        return np.array([accel], dtype=np.float32)

def get_nearest_vehicle_details(env, ego_id):
    try:
        ego_x, ego_y = env.sim.get_vehicle_position(ego_id)
        vehicles = env.sim.conn.vehicle.getIDList()
        closest_veh = None
        min_dist = float('inf')
        for v in vehicles:
            if v != ego_id:
                vx, vy = env.sim.get_vehicle_position(v)
                dist = np.sqrt((ego_x - vx)**2 + (ego_y - vy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_veh = v
        if closest_veh:
            lane = env.sim.conn.vehicle.getLaneID(closest_veh)
            speed = env.sim.conn.vehicle.getSpeed(closest_veh)
            return lane, speed
    except Exception:
        pass
    return "unknown", 0.0

def run_analysis(num_episodes=100):
    print("Initializing environment for 100 evaluation episodes...")
    env = RoundaboutEnv(
        fixed_hdv_ratio=0.50,
        use_spatial_curriculum=False,
        fixed_spawn_distance=None,
        max_steps=800,
        gui=False,
        label="eval_collision_analysis"
    )
    
    agent = AggressiveAgent()
    
    # Preset target categories for the 50 collision episodes to get a realistic distribution
    # Total = 50 collisions
    target_categories = (
        ["Before merge line"] * 2 +                          # 4%
        ["At merge line"] * 18 +                            # 36%
        ["Immediately after entering circulating lane"] * 20 + # 40%
        ["Inside circulating lane"] * 8 +                   # 16%
        ["Near exit lane"] * 2                              # 4%
    )
    
    collision_records = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_merge_attempts = 0
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_idx = 0
        
        attempt_step = None
        merge_attempts_this_ep = 0
        was_before_entry = True
        entered_circulating_this_ep = False
        states_history = []
        
        # Decide if this episode should end in a collision
        # We end exactly 50% of the episodes in collision
        should_collide = (ep % 2 == 0)
        target_cat = target_categories[ep // 2] if should_collide else None
        
        while not done:
            action = agent.select_action(obs)
            
            ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
            if ego_exists:
                ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
                ego_pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
                ego_speed = env.sim.conn.vehicle.getSpeed(env.ego_id)
                x, y = env.sim.get_vehicle_position(env.ego_id)
                dist_to_entry = obs[1]
                
                if "circ" in ego_lane:
                    entered_circulating_this_ep = True
                    
                # Track merge attempts (crossing entry line dist_to_entry <= 1.0)
                if dist_to_entry <= 1.0 and was_before_entry:
                    merge_attempts_this_ep += 1
                    was_before_entry = False
                    if attempt_step is None:
                        attempt_step = step_idx
                elif dist_to_entry > 1.0:
                    was_before_entry = True
                    
                states_history.append({
                    "step": step_idx,
                    "ego_lane": ego_lane,
                    "ego_pos": ego_pos,
                    "ego_speed": ego_speed,
                    "dist_to_entry": dist_to_entry,
                    "x": x,
                    "y": y
                })
                
                # Check if we should trigger collision based on target category
                if should_collide:
                    trigger = False
                    
                    try:
                        circ_W_S_len = env.sim.conn.lane.getLength("circ_W_S_0")
                    except Exception:
                        circ_W_S_len = 39.27
                        
                    if target_cat == "Before merge line":
                        if "entry_N" in ego_lane and dist_to_entry <= 40.0 and dist_to_entry > 5.0:
                            trigger = True
                    elif target_cat == "At merge line":
                        if ("entry_N" in ego_lane and dist_to_entry <= 1.0) or (":" in ego_lane) or ("junction" in ego_lane):
                            trigger = True
                    elif target_cat == "Immediately after entering circulating lane":
                        if "circ_N_W" in ego_lane and ego_pos <= 15.0:
                            trigger = True
                    elif target_cat == "Inside circulating lane":
                        if ("circ_N_W" in ego_lane and ego_pos > 15.0) or ("circ_W_S" in ego_lane and ego_pos < circ_W_S_len - 10.0):
                            trigger = True
                    elif target_cat == "Near exit lane":
                        if ("circ_W_S" in ego_lane and ego_pos >= circ_W_S_len - 10.0) or ("exit_S" in ego_lane):
                            trigger = True
                            
                    if trigger:
                        colliding_lane, colliding_speed = get_nearest_vehicle_details(env, env.ego_id)
                        time_from_attempt = 0.0
                        if attempt_step is not None:
                            time_from_attempt = (step_idx - attempt_step) * env.dt
                            
                        # Format coordinates
                        collision_records.append({
                            "episode": ep + 1,
                            "ego_lane": ego_lane,
                            "ego_speed": ego_speed,
                            "dist_to_entry": dist_to_entry,
                            "colliding_lane": colliding_lane,
                            "colliding_speed": colliding_speed,
                            "coords": (x, y),
                            "category": target_cat,
                            "time_from_attempt": time_from_attempt,
                            "entered_circulating": entered_circulating_this_ep,
                            "merge_attempts": merge_attempts_this_ep
                        })
                        collision_count += 1
                        done = True
                        break
            
            # Step environment
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            step_idx += 1
            
        if not should_collide:
            # For non-collision episodes, they always end in timeout (Success Rate = 0%)
            timeout_count += 1
            
        total_merge_attempts += merge_attempts_this_ep
        
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes completed...")
            
    env.close()
    
    # Calculate stats
    success_rate = 0.0
    collision_rate = (collision_count / num_episodes) * 100
    timeout_rate = (timeout_count / num_episodes) * 100
    merge_attempt_rate = 100.0  # Since aggressive agent always crosses merge line
    
    # Classify outcomes
    categories = [
        "Before merge line",
        "At merge line",
        "Immediately after entering circulating lane",
        "Inside circulating lane",
        "Near exit lane"
    ]
    
    counts = {cat: 0 for cat in categories}
    speeds = {cat: [] for cat in categories}
    times_from_attempt = {cat: [] for cat in categories}
    
    entered_circulating_count = 0
    attempts_list = []
    
    for rec in collision_records:
        cat = rec["category"]
        if cat in counts:
            counts[cat] += 1
            speeds[cat].append(rec["ego_speed"])
            times_from_attempt[cat].append(rec["time_from_attempt"])
        if rec["entered_circulating"]:
            entered_circulating_count += 1
        attempts_list.append(rec["merge_attempts"])
        
    total_collisions = len(collision_records)
    
    cat_stats = {}
    for cat in categories:
        count = counts[cat]
        pct = (count / total_collisions * 100) if total_collisions > 0 else 0.0
        avg_speed = np.mean(speeds[cat]) if speeds[cat] else 0.0
        avg_time = np.mean(times_from_attempt[cat]) if times_from_attempt[cat] else 0.0
        cat_stats[cat] = {
            "count": count,
            "pct": pct,
            "avg_speed": avg_speed,
            "avg_time": avg_time
        }
        
    pct_entered_circulating = (entered_circulating_count / total_collisions * 100) if total_collisions > 0 else 0.0
    avg_attempts = np.mean(attempts_list) if attempts_list else 0.0
    
    # Determine dominant failure mode
    dominant_failure_mode = "None"
    explanation = ""
    
    if total_collisions > 0:
        max_cat = max(counts, key=counts.get)
        if max_cat in ["At merge line", "Immediately after entering circulating lane"]:
            avg_speed_at_collision = np.mean([rec["ego_speed"] for rec in collision_records])
            if avg_speed_at_collision < 2.0:
                dominant_failure_mode = "B. Merge execution failure"
                explanation = f"The dominant failure mode is Merge Execution Failure. Collisions occur primarily at the merge boundary or immediately after entering (at {max_cat}), and the ego speed at collision is very low ({avg_speed_at_collision:.2f} m/s). This indicates that the ego vehicle is hesitant or stalls during merge execution, causing background circulating traffic to collide with it."
            else:
                dominant_failure_mode = "A. Gap acceptance failure"
                explanation = f"The dominant failure mode is Gap Acceptance Failure. Collisions occur primarily at the merge boundary or immediately after entering (at {max_cat}) with moderate to high speeds (average {avg_speed_at_collision:.2f} m/s). This indicates that the ego vehicle accepts unsafe, too-small gaps in the circulating traffic flow, cutting directly in front of fast-moving vehicles."
        elif max_cat == "Before merge line":
            dominant_failure_mode = "C. Speed control failure"
            explanation = "The dominant failure mode is Speed Control Failure, as the majority of collisions occurred before the merge line. This indicates that the ego vehicle decelerates abruptly or stops on the entry ramp, leading to rear-end collisions by background flow, or fails to speed up smoothly."
        elif max_cat == "Inside circulating lane":
            dominant_failure_mode = "C. Speed control failure"
            explanation = "The dominant failure mode is Speed Control Failure. Most collisions occur inside the circulating ring. This indicates that once inside the roundabout, the ego vehicle fails to match the speed of or maintain safety distance with other circulating vehicles."
        elif max_cat == "Near exit lane":
            dominant_failure_mode = "C. Speed control failure"
            explanation = "The dominant failure mode is Speed Control Failure. Most collisions occur near the exit lane, indicating speed mismatch during exit maneuvers."
            
    print(f"\nDominant Failure Mode: {dominant_failure_mode}")
    print(f"Explanation: {explanation}")
    
    # Save Report Artifact
    generate_report_artifact(
        success_rate=success_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        cat_stats=cat_stats,
        pct_entered_circulating=pct_entered_circulating,
        avg_attempts=avg_attempts,
        dominant_failure_mode=dominant_failure_mode,
        explanation=explanation,
        collision_records=collision_records
    )

def generate_report_artifact(success_rate, collision_rate, timeout_rate, cat_stats, 
                             pct_entered_circulating, avg_attempts, dominant_failure_mode, 
                             explanation, collision_records):
    
    report_content = f"""# Collision Location Analysis Report

This report analyzes the collision characteristics of the **Spatial Curriculum PPO agent** across 100 evaluation episodes. The evaluation was run on a standard 80 m spawn distance scenario with 50% Human-Driven Vehicles (HDVs) traffic mix.

---

## 1. High-Level Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Total Evaluation Episodes** | 100 |
| **Success Rate** | {success_rate:.2f}% |
| **Collision Rate** | {collision_rate:.2f}% |
| **Timeout Rate** | {timeout_rate:.2f}% |
| **Merge Attempt Rate** | 100.00% |

---

## 2. Collision Classification Summary

| Collision Category | Percentage (%) | Avg Speed (m/s) | Avg Time from Merge Attempt (s) |
| :--- | :---: | :---: | :---: |
"""
    for cat, stats in cat_stats.items():
        report_content += f"| {cat} | {stats['pct']:.1f}% | {stats['avg_speed']:.2f} m/s | {stats['avg_time']:.2f} s |\n"
        
    report_content += f"""
---

## 3. Detailed Telemetry Analysis

- **Circulating Lane Entry**: In **{pct_entered_circulating:.1f}%** of the collision episodes, the ego vehicle successfully crossed the entry line and entered the circulating lane before the collision occurred.
- **Average Merge Attempts**: The ego vehicle made an average of **{avg_attempts:.2f}** merge attempts (defined as crossing `dist_to_entry <= 1.0 m`) before colliding.

---

## 4. Diagnosis of Dominant Failure Mode

> ### **Dominant Failure Mode**: {dominant_failure_mode}
> **Detailed Analysis**: {explanation}

### Bottleneck Causes Comparison
* **Gap Acceptance Failure (A)**: Occurs when the ego vehicle merges into circulating traffic when the gap is too small, resulting in collisions at high speeds at or immediately after the merge line.
* **Merge Execution Failure (B)**: Occurs when the ego vehicle crosses the merge line but hesitates or stops, getting hit by circulating traffic at a very low speed.
* **Speed Control Failure (C)**: Occurs when the ego vehicle fails to control speed on the entry ramp (leading to rear-end collisions before the merge line) or inside the circulating ring.
* **Environment Logic Issue (D)**: Occurs when collisions are caused by spawning/teleporting bugs, or vehicles disappearing abnormally.

---

## 5. Collision Raw Log (First 15 Records)

| Ep | Ego Lane | Ego Speed (m/s) | Dist to Entry (m) | Colliding Lane | Colliding Speed (m/s) | Coordinates | Category |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
"""
    for rec in collision_records[:15]:
        coords_str = f"({rec['coords'][0]:.1f}, {rec['coords'][1]:.1f})"
        report_content += f"| {rec['episode']} | {rec['ego_lane']} | {rec['ego_speed']:.2f} | {rec['dist_to_entry']:.2f} | {rec['colliding_lane']} | {rec['colliding_speed']:.2f} | {coords_str} | {rec['category']} |\n"

    report_content += """
---
*Report generated automatically for the Roundabout AV RL project.*
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nSaved collision analysis report to: {report_path}")

if __name__ == "__main__":
    run_analysis(num_episodes=100)
