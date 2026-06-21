import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

# Model path mapping
MODEL_PATHS = {
    "A": "results/models/final_agent_a_fixed_50.zip",
    "B": "results/models/agent_spatial_curriculum.zip",
    "C": "results/models/agent_spatial_curriculum_30k.zip"
}

MODEL_NAMES = {
    "A": "Baseline PPO",
    "B": "Spatial Curriculum PPO",
    "C": "Latest Reward-Shaped PPO"
}

def main():
    print("=" * 80)
    print("                 SUMO-GUI VISUAL EVALUATION CONTROLLER")
    print("=" * 80)
    print("Select Model to Evaluate:")
    print("  [A] Baseline PPO")
    print("      Path: results/models/final_agent_a_fixed_50.zip")
    print("  [B] Spatial Curriculum PPO")
    print("      Path: results/models/agent_spatial_curriculum.zip")
    print("  [C] Latest Reward-Shaped PPO")
    print("      Path: results/models/agent_spatial_curriculum_30k.zip")
    print("=" * 80)
    
    choice = ""
    while choice not in ["A", "B", "C"]:
        try:
            choice = input("Enter choice (A/B/C): ").strip().upper()
        except KeyboardInterrupt:
            print("\nExiting visual evaluation.")
            sys.exit(0)
            
    model_path = MODEL_PATHS[choice]
    model_name = MODEL_NAMES[choice]
    
    print(f"\nVerifying model existence: {model_path}...")
    if not os.path.exists(model_path):
        print(f"\033[1;31mError: Model file '{model_path}' not found.\033[0m")
        print("\nAvailable files in results/models/:\n")
        if os.path.exists("results/models"):
            for f in os.listdir("results/models"):
                print(f" - {f}")
        else:
            print("results/models/ directory does not exist.")
        sys.exit(1)
        
    print(f"Loading {model_name} model...")
    model = PPO.load(model_path)
    print("Model loaded successfully.")
    
    print("\nLaunching SUMO-GUI...")
    # Initialize standard evaluation environment with GUI mode enabled
    env = RoundaboutEnv(
        gui=True,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        label="visual_eval"
    )
    
    episodes_summary = []
    num_episodes = 10
    
    try:
        for ep in range(num_episodes):
            print(f"\n=========================================")
            print(f"        STARTING EPISODE {ep + 1} / {num_episodes}")
            print(f"        Model: {model_name}")
            print(f"=========================================")
            
            obs, info = env.reset()
            
            # Automatically track the ego vehicle in the SUMO-GUI window
            try:
                env.sim.conn.gui.trackVehicle("View #0", env.ego_id)
                env.sim.conn.gui.setZoom("View #0", 150.0)
            except Exception:
                pass
                
            done = False
            steps = 0
            
            reached_merge_zone_triggered = False
            attempted_merge_triggered = False
            entered_circulating_triggered = False
            
            episode_ttcs = []
            min_ttc = float('inf')
            time_to_merge_val = 0.0
            
            # Stage variables
            stage_str = "N/A"
            if "spatial_stage" in info:
                stage_str = f"Spatial Stage {info['spatial_stage']}"
            elif "curriculum_stage" in info:
                stage_str = f"Curriculum Stage {info['curriculum_stage']}"
                
            hdv_ratio = info.get("hdv_ratio", 0.50)
            
            while not done:
                # Deterministic prediction
                action, _ = model.predict(obs, deterministic=True)
                
                # Extract observation features
                speed = obs[0]
                dist_to_entry = obs[1]
                gap_size = obs[4]
                
                # Step environment
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                steps += 1
                
                # Slow down simulation (approx 0.33x real-time)
                time.sleep(0.3)
                
                # Live Telemetry
                print(f"[Ep {ep+1:2d}] Step {steps:3d} | Spd: {speed:5.2f}m/s | Dist: {dist_to_entry:6.2f}m | Gap: {gap_size:6.2f}m | Action: {action[0]:+5.2f} | Rew: {reward:+6.2f}")
                
                # Highlight Events
                if step_info.get("reached_merge_zone", False) and not reached_merge_zone_triggered:
                    reached_merge_zone_triggered = True
                    print("\033[1;36m>>> [EVENT] Ego vehicle entered MERGE_ZONE! <<\033[0m")
                    
                if dist_to_entry <= 1.0 and not attempted_merge_triggered:
                    attempted_merge_triggered = True
                    print("\033[1;33m>>> [EVENT] Ego vehicle is attempting to merge! <<\033[0m")
                    
                if step_info.get("success", False) and not entered_circulating_triggered:
                    entered_circulating_triggered = True
                    time_to_merge_val = step_info.get("time_to_merge", steps * env.dt)
                    print("\033[1;32m>>> [EVENT] Ego vehicle successfully entered Circulating Lane! <<\033[0m")
                    
                # Unsafe TTC warnings
                ttc = env._get_ttc_after_merge()
                if ttc != float('inf'):
                    ttc = min(ttc, 10.0)
                    episode_ttcs.append(ttc)
                    if ttc < min_ttc:
                        min_ttc = ttc
                    if ttc < env.ttc_threshold:
                        print(f"\033[1;31m!!! [WARNING] Unsafe TTC: {ttc:.2f} s !!!\033[0m")
                        
            # Determine outcome
            outcome = step_info.get("termination_reason", "timeout").upper()
            if outcome == "SUCCESS":
                print("\033[1;32m*** [SUCCESS] Ego vehicle successfully exited the roundabout! ***\033[0m")
            elif outcome == "COLLISION":
                print("\033[1;31m!!! [COLLISION] Ego vehicle crashed! !!!\033[0m")
            elif outcome == "TIMEOUT":
                print("\033[1;33m[TIMEOUT] Episode timed out after max steps.\033[0m")
                
            avg_ttc = np.mean(episode_ttcs) if episode_ttcs else 999.0
            min_ttc_val = min_ttc if min_ttc != float('inf') else 999.0
            
            episodes_summary.append({
                "episode": ep + 1,
                "outcome": outcome,
                "time_to_merge": time_to_merge_val if entered_circulating_triggered else "N/A",
                "avg_ttc": avg_ttc,
                "min_ttc": min_ttc_val
            })
            
            print(f"\nEpisode {ep + 1} finished. Pausing simulation for 3 seconds...")
            time.sleep(3.0)
            
    except KeyboardInterrupt:
        print("\nVisual evaluation interrupted by user.")
    finally:
        env.close()
        print("SUMO-GUI closed.")
        
    # Generate summary report
    print("\n" + "=" * 80)
    print("                     VISUAL EVALUATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Model Evaluated: {model_name}")
    print("-" * 80)
    print(f"{'Ep':<4} | {'Outcome':<10} | {'Time to Merge':<15} | {'Avg TTC':<10} | {'Min TTC':<10}")
    print("-" * 80)
    for res in episodes_summary:
        merge_str = f"{res['time_to_merge']:.2f} s" if isinstance(res['time_to_merge'], float) else res['time_to_merge']
        avg_ttc_str = f"{res['avg_ttc']:.2f} s" if res['avg_ttc'] != 999.0 else "N/A"
        min_ttc_str = f"{res['min_ttc']:.2f} s" if res['min_ttc'] != 999.0 else "N/A"
        print(f"{res['episode']:<4} | {res['outcome']:<10} | {merge_str:<15} | {avg_ttc_str:<10} | {min_ttc_str:<10}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
