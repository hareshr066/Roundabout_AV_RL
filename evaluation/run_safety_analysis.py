import os
import sys
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.roundabout_env import RoundaboutEnv

def main():
    parser = argparse.ArgumentParser(description="Run Safety Analysis")
    parser.add_argument("--model", type=str, default="results/models/final_best_agent.zip", help="Path to best model")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (2 episodes)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, args.model)
    
    episodes = 2 if args.test_mode else args.episodes
    
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    env = RoundaboutEnv(
        gui=False,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        max_steps=400,
        verbose=False
    )

    all_results = []
    
    print(f"Running safety analysis for {episodes} episodes...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        episode_ttcs = []
        episode_jerks = []
        accelerations = []
        decelerations = []
        merge_speed = None
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Extract info for safety
            try:
                ttc = env._get_ttc_after_merge()
            except Exception:
                ttc = float('inf')
            
            if ttc < 100.0:
                episode_ttcs.append(ttc)
            
            accel = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            accelerations.append(accel)
            if accel < 0:
                decelerations.append(accel)
                
            # Jerk is rate of change of acceleration
            if len(accelerations) > 1:
                dt = getattr(env, 'dt', 0.1)
                jerk = (accelerations[-1] - accelerations[-2]) / dt
                episode_jerks.append(jerk)
                
            if info.get('reached_merge_zone', False) and merge_speed is None:
                merge_speed = float(obs[0])  # ego speed is obs[0]

        term_reason = info.get('termination_reason', '')
        is_success = (term_reason == 'success' or info.get('success', False))
        is_collision = (term_reason == 'collision' or info.get('collision', False))

        # Compute episode stats
        min_ttc = min(episode_ttcs) if episode_ttcs else 10.0
        max_dec = min(decelerations) if decelerations else 0.0
        critical_ttc_steps = sum(1 for t in episode_ttcs if t < 2.0)
        ttc_under_4_steps = sum(1 for t in episode_ttcs if t < 4.0)
        total_steps = len(episode_ttcs) if episode_ttcs else 1
        
        all_results.append({
            'episode': ep,
            'min_ttc': min_ttc,
            'max_deceleration': max_dec,
            'mean_jerk': float(np.mean(episode_jerks)) if episode_jerks else 0.0,
            'std_jerk': float(np.std(episode_jerks)) if episode_jerks else 0.0,
            'merge_speed': float(merge_speed) if merge_speed is not None else 0.0,
            'critical_ttc_ratio': critical_ttc_steps / total_steps,
            'ttc_under_4_ratio': ttc_under_4_steps / total_steps,
            'success': is_success,
            'collision': is_collision
        })

    df = pd.DataFrame(all_results)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "safety_analysis_results.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print("SAFETY ANALYSIS REPORT")
    print("="*50)
    print(f"Total Episodes Analyzed: {episodes}")
    print(f"Success Rate: {df['success'].mean()*100:.1f}%")
    print(f"Collision Rate: {df['collision'].mean()*100:.1f}%")
    print("-"*50)
    print(f"Min TTC across all episodes: {df['min_ttc'].min():.2f}s")
    print(f"Mean of Min-TTC per episode: {df['min_ttc'].mean():.2f}s (std: {df['min_ttc'].std():.2f})")
    print(f"5th Percentile Min-TTC: {np.percentile(df['min_ttc'], 5):.2f}s")
    print(f"Mean Max Deceleration: {df['max_deceleration'].mean():.2f} m/s^2")
    print(f"Mean Jerk: {df['mean_jerk'].mean():.2f} (std: {df['std_jerk'].mean():.2f})")
    print(f"Avg % steps with TTC < 2s: {df['critical_ttc_ratio'].mean()*100:.1f}%")
    print(f"Avg % steps with TTC < 4s: {df['ttc_under_4_ratio'].mean()*100:.1f}%")
    print("="*50)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
