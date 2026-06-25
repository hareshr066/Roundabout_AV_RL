import os
import sys
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def evaluate_model(model_path, env_kwargs, label="Pre-trained Model", num_episodes=50):
    print(f"\nEvaluating {label} at 80m spawn distance...")
    env = RoundaboutEnv(gui=False, max_steps=200, **env_kwargs)
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
        
    successes = 0
    collisions = 0
    timeouts = 0
    
    for ep in range(num_episodes):
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
            
    print(f"Results | Success: {successes/num_episodes*100:.1f}% | Collision: {collisions/num_episodes*100:.1f}% | Timeout: {timeouts/num_episodes*100:.1f}%")
    env.close()

if __name__ == "__main__":
    model_dir = os.path.join("results", "models")
    
    # Evaluate final_agent_b_curriculum.zip
    evaluate_model(
        os.path.join(model_dir, "final_agent_b_curriculum.zip"),
        {
            "use_context_aware": True,
            "use_spatial_curriculum": False,
            "fixed_spawn_distance": 80.0,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        },
        label="final_agent_b_curriculum.zip"
    )
    
    # Evaluate agent_spatial_curriculum_30k.zip
    evaluate_model(
        os.path.join(model_dir, "agent_spatial_curriculum_30k.zip"),
        {
            "use_context_aware": True,
            "use_spatial_curriculum": False,
            "fixed_spawn_distance": 80.0,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        },
        label="agent_spatial_curriculum_30k.zip"
    )
