import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def evaluate_model(model_path, num_episodes=100):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
        
    model = PPO.load(model_path)
    
    # Initialize environment
    env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
    
    success_count = 0
    collision_count = 0
    timeout_count = 0
    completion_times_steps = []
    completion_times_sec = []
    
    successful_completion_times_steps = []
    successful_completion_times_sec = []
    
    print(f"Running {num_episodes} deterministic evaluation episodes...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        reason = step_info.get("termination_reason", "timeout").upper()
        
        completion_time_sec = steps * env.dt
        completion_times_steps.append(steps)
        completion_times_sec.append(completion_time_sec)
        
        if reason == "SUCCESS":
            success_count += 1
            successful_completion_times_steps.append(steps)
            successful_completion_times_sec.append(completion_time_sec)
        elif reason == "COLLISION":
            collision_count += 1
        elif reason == "TIMEOUT":
            timeout_count += 1
            
        if (ep + 1) % 10 == 0 or (ep + 1) == num_episodes:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes completed...")
            
    env.close()
    
    # Compute metrics
    success_rate = (success_count / num_episodes) * 100
    collision_rate = (collision_count / num_episodes) * 100
    timeout_rate = (timeout_count / num_episodes) * 100
    
    avg_time_all_steps = np.mean(completion_times_steps)
    avg_time_all_sec = np.mean(completion_times_sec)
    
    avg_time_succ_steps = np.mean(successful_completion_times_steps) if successful_completion_times_steps else 0.0
    avg_time_succ_sec = np.mean(successful_completion_times_sec) if successful_completion_times_sec else 0.0
    
    print("\n=========================================")
    print("        EVALUATION REPORT")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Episodes: {num_episodes}")
    print("=========================================")
    print(f"SUCCESS RATE:         {success_rate:.2f}%")
    print(f"COLLISION RATE:       {collision_rate:.2f}%")
    print(f"TIMEOUT RATE:         {timeout_rate:.2f}%")
    print(f"AVG COMPLETION TIME (All):        {avg_time_all_steps:.1f} steps ({avg_time_all_sec:.2f} s)")
    print(f"AVG COMPLETION TIME (Successes):  {avg_time_succ_steps:.1f} steps ({avg_time_succ_sec:.2f} s)")
    print("=========================================\n")
    
    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_completion_time_all_steps": avg_time_all_steps,
        "avg_completion_time_all_sec": avg_time_all_sec,
        "avg_completion_time_succ_steps": avg_time_succ_steps,
        "avg_completion_time_succ_sec": avg_time_succ_sec
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Agent in Roundabout RL")
    parser.add_argument("--model", type=str, default="results/models/final_agent_b_curriculum.zip", help="Path to PPO model zip")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.episodes)
