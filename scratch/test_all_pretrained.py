import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def evaluate(model_path, env_kwargs, label):
    env = RoundaboutEnv(gui=False, max_steps=200, verbose=False, **env_kwargs)
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"{label}: Error loading - {e}")
        env.close()
        return
        
    successes = 0
    collisions = 0
    timeouts = 0
    
    for ep in range(50):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
        reason = step_info.get("termination_reason", "timeout")
        if reason == "success":
            successes += 1
        elif reason == "collision":
            collisions += 1
        else:
            timeouts += 1
            
    print(f"{label:<40} | Success: {successes*2:>3d}% | Collision: {collisions*2:>3d}% | Timeout: {timeouts*2:>3d}%")
    env.close()

if __name__ == "__main__":
    model_dir = os.path.join("results", "models")
    
    # 1. final_agent_a_fixed_50.zip (Context-Aware + Reward Shaping, NO Spatial Curriculum)
    evaluate(
        os.path.join(model_dir, "final_agent_a_fixed_50.zip"),
        {
            "use_context_aware": True,
            "use_spatial_curriculum": False,
            "fixed_spawn_distance": 80.0,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        },
        "Agent A Fixed (Context + Reward Shaping)"
    )
    
    # 2. agent_spatial_curriculum_30k.zip (Variant 4: Context + Spatial Curriculum + Reward Shaping)
    evaluate(
        os.path.join(model_dir, "agent_spatial_curriculum_30k.zip"),
        {
            "use_context_aware": True,
            "use_spatial_curriculum": False,
            "fixed_spawn_distance": 80.0,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        },
        "Variant 4 (Spatial Curriculum + Reward Shaping)"
    )
    
    # 3. final_agent_b_curriculum.zip (Variant 5: Full Method)
    evaluate(
        os.path.join(model_dir, "final_agent_b_curriculum.zip"),
        {
            "use_context_aware": True,
            "use_spatial_curriculum": False,
            "fixed_spawn_distance": 80.0,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        },
        "Variant 5 (Full Method)"
    )
