import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.roundabout_env import RoundaboutEnv

def run_single_episode_headless():
    model_path = "results/models/final_agent_b_curriculum.zip"
    
    print("=" * 80)
    print("            RUNNING ROUNDABOUT AV SIMULATION ONCE (HEADLESS)")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Falling back to random actions...")
        model = None
    else:
        print(f"Loading trained PPO policy from: {model_path}")
        model = PPO.load(model_path)
        print("Model loaded successfully!")
        
    print("\nInitializing Headless SUMO Environment...")
    env = RoundaboutEnv(
        gui=False,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        max_steps=400,
        label="single_run_headless",
        verbose=False
    )
    
    try:
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        
        print("\nRunning simulation steps...")
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward
            
            # Print telemetry details
            print(f"Step {steps:3d} | Speed: {obs[0]:5.2f} m/s | Dist to Entry: {obs[1]:6.2f} m | Circ Dist: {obs[2]:5.2f} m | Action Accel: {action[0]:+5.2f} m/s^2 | Reward: {reward:+6.2f}")
            
        outcome = step_info.get("termination_reason", "timeout").upper()
        print("\n" + "=" * 80)
        print("                            RUN COMPLETED")
        print("=" * 80)
        print(f"Outcome:           {outcome}")
        print(f"Total Steps:       {steps} ({steps * env.dt:.1f} seconds of simulation)")
        print(f"Total Reward:      {total_reward:.2f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        env.close()
        print("\nSUMO Environment closed.")

if __name__ == "__main__":
    run_single_episode_headless()
