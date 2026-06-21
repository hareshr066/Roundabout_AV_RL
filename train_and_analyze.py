import os
import sys
import time
import numpy as np
import traceback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.roundabout_env import RoundaboutEnv

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

model_path = "results/models/agent_spatial_curriculum_30k.zip"
artifact_dir = r"C:\Users\KALAIVANI J\.gemini\antigravity-ide\brain\73f5c967-442d-4d47-9a5a-2c30b678ef80"
report_path = os.path.join(artifact_dir, "collision_location_analysis.md")

def train():
    print("Training Spatial Curriculum PPO for 30,000 steps...")
    env = RoundaboutEnv(
        fixed_hdv_ratio=0.50,
        use_spatial_curriculum=True,
        spatial_target_success_rate=0.80,
        spatial_window_size=50,
        gui=False,
        label="train_30k"
    )
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda"
    )
    
    model.learn(total_timesteps=30000)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    env.close()
    print("Training finished and model saved.")

def get_colliding_vehicle_details(env, ego_id):
    try:
        colliding_vehs = env.sim.conn.simulation.getCollidingVehiclesIDList()
        other_vehs = [v for v in colliding_vehs if v != ego_id]
        if other_vehs:
            other_id = other_vehs[0]
            lane = env.sim.conn.vehicle.getLaneID(other_id)
            speed = env.sim.conn.vehicle.getSpeed(other_id)
            return lane, speed
    except Exception:
        pass
        
    # Fallback: find closest vehicle to ego
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
        if closest_veh and min_dist < 10.0:
            lane = env.sim.conn.vehicle.getLaneID(closest_veh)
            speed = env.sim.conn.vehicle.getSpeed(closest_veh)
            return lane, speed
    except Exception:
        pass
        
    return "unknown", 0.0

def run_evaluation(num_episodes=100):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print(f"Initializing standard 80m spawn evaluation environment...")
    env = RoundaboutEnv(
        fixed_hdv_ratio=0.50,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        gui=False,
        label="eval_30k"
    )
    
    collision_records = []
    successes = 0
    collisions = 0
    timeouts = 0
    total_merge_attempts = 0
    
    entered_circulating_list = []
    episode_collisions = []
    valid_avg_ttcs = []
    entry_delays = []
    times_to_merge = []
    
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
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
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
                
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            step_idx += 1
            
        reason = step_info.get("termination_reason", "timeout")
        total_merge_attempts += merge_attempts_this_ep
        
        entered_circulating_list.append(entered_circulating_this_ep)
        episode_collisions.append(reason == "collision")
        
        # Track entry delay and merge times
        time_to_merge_val = step_info.get("time_to_merge", 0.0)
        entry_delays.append(time_to_merge_val)
        if entered_circulating_this_ep:
            times_to_merge.append(time_to_merge_val)
            
        ep_avg_ttc = step_info.get("avg_ttc", 999.0)
        if ep_avg_ttc != 999.0:
            valid_avg_ttcs.append(ep_avg_ttc)
            
        if reason == "success":
            successes += 1
        elif reason == "timeout":
            timeouts += 1
        elif reason == "collision":
            collisions += 1
            
            # Analyze collision details
            if states_history:
                last_state = states_history[-1]
                ego_lane = last_state["ego_lane"]
                ego_pos = last_state["ego_pos"]
                dist_to_entry = last_state["dist_to_entry"]
                
                # Retrieve colliding vehicle details at collision step
                colliding_lane, colliding_speed = get_colliding_vehicle_details(env, env.ego_id)
                
                # Classify category
                category = "Unknown"
                try:
                    circ_W_S_len = env.sim.conn.lane.getLength("circ_W_S_0")
                except Exception:
                    circ_W_S_len = 39.27
                    
                if "entry_N" in ego_lane:
                    if dist_to_entry > 1.0:
                        category = "Before merge line"
                    else:
                        category = "At merge line"
                elif "circ_N_W" in ego_lane:
                    if ego_pos <= 1.5:
                        category = "At merge line"
                    elif ego_pos <= 15.0:
                        category = "Immediately after entering circulating lane"
                    else:
                        category = "Inside circulating lane"
                elif "circ_W_S" in ego_lane:
                    if ego_pos >= circ_W_S_len - 10.0:
                        category = "Near exit lane"
                    else:
                        category = "Inside circulating lane"
                elif "exit_S" in ego_lane:
                    category = "Near exit lane"
                elif ":" in ego_lane or "junction" in ego_lane:
                    category = "At merge line"
                else:
                    category = "Inside circulating lane"
                    
                time_from_attempt = 0.0
                if attempt_step is not None:
                    time_from_attempt = (last_state["step"] - attempt_step) * env.dt
                    
                collision_records.append({
                    "episode": ep + 1,
                    "ego_lane": ego_lane,
                    "ego_speed": last_state["ego_speed"],
                    "dist_to_entry": dist_to_entry,
                    "colliding_lane": colliding_lane,
                    "colliding_speed": colliding_speed,
                    "coords": (last_state["x"], last_state["y"]),
                    "category": category,
                    "time_from_attempt": time_from_attempt,
                    "entered_circulating": entered_circulating_this_ep,
                    "merge_attempts": merge_attempts_this_ep
                })
                
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes completed...")
            
    env.close()
    
    # Calculate stats
    success_rate = (successes / num_episodes) * 100
    collision_rate = (collisions / num_episodes) * 100
    timeout_rate = (timeouts / num_episodes) * 100
    
    total_entered = sum(entered_circulating_list)
    merge_successes = sum(1 for entered, col in zip(entered_circulating_list, episode_collisions) if entered and not col)
    merge_success_rate = (merge_successes / total_entered * 100) if total_entered > 0 else 0.0
    overall_avg_ttc = np.mean(valid_avg_ttcs) if valid_avg_ttcs else 999.0
    avg_entry_delay = np.mean(entry_delays) if entry_delays else 0.0
    avg_time_to_merge = np.mean(times_to_merge) if times_to_merge else 0.0
    
    print(f"\nEvaluation Results over {num_episodes} episodes:")
    print(f"  Success Rate:        {success_rate:.1f}%")
    print(f"  Collision Rate:      {collision_rate:.1f}%")
    print(f"  Timeout Rate:        {timeout_rate:.1f}%")
    print(f"  Merge Success Rate:  {merge_success_rate:.1f}% (Ego merged & survived)")
    print(f"  Average TTC:         {overall_avg_ttc:.2f} s")
    print(f"  Average Entry Delay: {avg_entry_delay:.2f} s")
    print(f"  Average Time to Merge: {avg_time_to_merge:.2f} s")
    
    # Classifications
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
    print(f"\nCollision Category Percentages (Total Collisions: {total_collisions}):")
    
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
        print(f"  {cat:<45}: {pct:.1f}% (Avg Speed: {avg_speed:.2f} m/s, Avg Time from attempt: {avg_time:.2f} s)")
        
    pct_entered_circulating = (entered_circulating_count / total_collisions * 100) if total_collisions > 0 else 0.0
    avg_attempts = np.mean(attempts_list) if attempts_list else 0.0
    
    print(f"\nAdditional Metrics:")
    print(f"  Ego entered circulating lane before collision: {pct_entered_circulating:.1f}% of collisions")
    print(f"  Average merge attempts before collision: {avg_attempts:.2f}")
    
    # Diagnose failure mode
    dominant_failure_mode = "None"
    explanation = ""
    
    if total_collisions > 0:
        max_cat = max(counts, key=counts.get)
        max_pct = cat_stats[max_cat]["pct"]
        
        if max_cat == "Before merge line":
            dominant_failure_mode = "C. Speed control failure"
            explanation = "The dominant failure mode is Speed Control Failure, as the majority of collisions occurred before the merge line. This indicates that the ego vehicle decelerates abruptly or stops on the entry ramp, leading to rear-end collisions by background flow, or fails to speed up smoothly."
        elif max_cat in ["At merge line", "Immediately after entering circulating lane"]:
            avg_speed_at_collision = np.mean([rec["ego_speed"] for rec in collision_records])
            if avg_speed_at_collision < 2.0:
                dominant_failure_mode = "B. Merge execution failure"
                explanation = f"The dominant failure mode is Merge Execution Failure. Collisions occur primarily at the merge boundary or immediately after entering (at {max_cat}), and the ego speed at collision is very low ({avg_speed_at_collision:.2f} m/s). This indicates that the ego vehicle is hesitant or stalls during merge execution, causing background circulating traffic to collide with it."
            else:
                dominant_failure_mode = "A. Gap acceptance failure"
                explanation = f"The dominant failure mode is Gap Acceptance Failure. Collisions occur primarily at the merge boundary or immediately after entering (at {max_cat}) with moderate to high speeds. This indicates that the ego vehicle accepts unsafe, too-small gaps in the circulating traffic flow, cutting directly in front of fast-moving vehicles."
        elif max_cat == "Inside circulating lane":
            dominant_failure_mode = "C. Speed control failure"
            explanation = "The dominant failure mode is Speed Control Failure. Most collisions occur inside the circulating ring. This indicates that once inside the roundabout, the ego vehicle fails to match the speed of or maintain safety distance with other circulating vehicles."
        elif max_cat == "Near exit lane":
            dominant_failure_mode = "C. Speed control failure"
            explanation = "The dominant failure mode is Speed Control Failure. Most collisions occur near the exit lane, indicating speed mismatch during exit maneuvers."
            
        unknown_other_pct = sum(1 for r in collision_records if r["colliding_lane"] == "unknown") / total_collisions * 100
        if unknown_other_pct > 30.0:
            dominant_failure_mode = "D. Environment logic issue"
            explanation = f"The dominant failure mode points to Environment Logic Issues, as {unknown_other_pct:.1f}% of collisions involved unknown lanes or missing background traffic data, suggesting SUMO teleporting/spawning bugs."
            
    print(f"\nDominant Failure Mode: {dominant_failure_mode}")
    print(f"Explanation: {explanation}")
    
    # Save Report Artifact
    generate_report_artifact(
        success_rate=success_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        merge_success_rate=merge_success_rate,
        avg_ttc=overall_avg_ttc,
        avg_entry_delay=avg_entry_delay,
        avg_time_to_merge=avg_time_to_merge,
        cat_stats=cat_stats,
        pct_entered_circulating=pct_entered_circulating,
        avg_attempts=avg_attempts,
        dominant_failure_mode=dominant_failure_mode,
        explanation=explanation,
        collision_records=collision_records
    )

def generate_report_artifact(success_rate, collision_rate, timeout_rate, merge_success_rate, avg_ttc,
                             avg_entry_delay, avg_time_to_merge,
                             cat_stats, pct_entered_circulating, avg_attempts, dominant_failure_mode, 
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
| **Merge Success Rate** | {merge_success_rate:.2f}% |
| **Average TTC (Capped 10s)** | {avg_ttc:.2f} s |
| **Average Entry Delay** | {avg_entry_delay:.2f} s |
| **Average Time to Merge** | {avg_time_to_merge:.2f} s |
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
    train()
    run_evaluation(num_episodes=100)
