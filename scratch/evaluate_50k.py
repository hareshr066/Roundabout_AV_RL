import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import PPO
from env.roundabout_env import RoundaboutEnv

def main():
    model_path = "results/models/agent_spatial_curriculum_30k.zip"
    print(f"Loading 50k model from {model_path}...")
    model = PPO.load(model_path)
    
    print("Initializing environment...")
    env = RoundaboutEnv(
        fixed_hdv_ratio=0.50,
        use_spatial_curriculum=False,
        fixed_spawn_distance=None,
        max_steps=800,
        gui=False,
        label="eval_50k_script"
    )
    
    num_episodes = 100
    successes = 0
    collisions = 0
    timeouts = 0
    
    episode_lengths = []
    merge_times = []
    
    print(f"Evaluating model over {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_idx = 0
        merge_step = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            step_idx += 1
            
            # Check if ego has merged into the circulating lane
            if env.ego_id in env.sim.conn.vehicle.getIDList():
                ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
                if "circ" in ego_lane and merge_step is None:
                    merge_step = step_idx
                    
        reason = step_info.get("termination_reason", "timeout")
        episode_lengths.append(step_idx)
        
        if merge_step is not None:
            # dt = 0.1s per step
            merge_times.append(merge_step * env.dt)
            
        if reason == "success":
            successes += 1
        elif reason == "timeout":
            timeouts += 1
        elif reason == "collision":
            collisions += 1
            
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes completed...")
            
    env.close()
    
    success_rate = (successes / num_episodes) * 100
    collision_rate = (collisions / num_episodes) * 100
    timeout_rate = (timeouts / num_episodes) * 100
    avg_merge_time = np.mean(merge_times) if merge_times else 0.0
    avg_ep_length = np.mean(episode_lengths) if episode_lengths else 0.0
    
    print("\n" + "=" * 50)
    print("              EVALUATION REPORT")
    print("=" * 50)
    print(f"  Success Rate:           {success_rate:.1f}%")
    print(f"  Collision Rate:         {collision_rate:.1f}%")
    print(f"  Timeout Rate:           {timeout_rate:.1f}%")
    print(f"  Average Merge Time:     {avg_merge_time:.2f} s")
    print(f"  Average Episode Length: {avg_ep_length:.1f} steps")
    print("=" * 50)

if __name__ == "__main__":
    main()
