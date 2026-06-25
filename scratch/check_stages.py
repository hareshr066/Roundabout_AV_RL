import os
import sys
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def evaluate_at_distances(model_path, env_kwargs, distances=[15.0, 30.0, 50.0, 80.0]):
    print(f"\n=========================================")
    print(f"Checking Model: {os.path.basename(model_path)}")
    print(f"=========================================")
    
    for dist in distances:
        kwargs = env_kwargs.copy()
        kwargs["use_spatial_curriculum"] = False
        kwargs["fixed_spawn_distance"] = dist
        kwargs["verbose"] = False
        
        env = RoundaboutEnv(gui=False, max_steps=200, **kwargs)
        try:
            model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"  Dist {dist}m: Error loading model - {e}")
            env.close()
            continue
            
        successes = 0
        collisions = 0
        timeouts = 0
        
        for ep in range(10):
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
                
        print(f"  Spawn Dist {dist:>4.1f}m | Success: {successes*10:>3d}% | Collision: {collisions*10:>3d}% | Timeout: {timeouts*10:>3d}%")
        env.close()

if __name__ == "__main__":
    model_dir = os.path.join("results", "models")
    
    evaluate_at_distances(
        os.path.join(model_dir, "ablation_v3_spatial.zip"),
        {
            "use_context_aware": True,
            "use_gap_reward": False,
            "fixed_hdv_ratio": 0.50
        }
    )
    
    evaluate_at_distances(
        os.path.join(model_dir, "ablation_v4_shaping.zip"),
        {
            "use_context_aware": True,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        }
    )
    
    evaluate_at_distances(
        os.path.join(model_dir, "ablation_v5_full.zip"),
        {
            "use_context_aware": True,
            "use_gap_reward": True,
            "fixed_hdv_ratio": 0.50
        }
    )
