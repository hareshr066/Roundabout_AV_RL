import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def check_speed():
    model_path = "results/models/agent_spatial_curriculum_30k.zip"
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    env = RoundaboutEnv(
        gui=False,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        max_steps=300,
        label="check_speed"
    )
    
    obs, info = env.reset()
    done = False
    step = 0
    
    print("-" * 120)
    print(f"{'Step':<5} | {'Speed':<8} | {'Lane':<20} | {'LanePos':<8} | {'Action':<8} | {'Reward':<8} | {'TTC':<8} | {'NearestCircDist':<15} | {'NearestCircSpd':<15}")
    print("-" * 120)
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # obs is [ego_speed, dist_to_entry, nearest_circ_dist, nearest_circ_speed, gap_size, hdv_ratio]
        ego_speed = obs[0]
        dist_to_entry = obs[1]
        nearest_circ_dist = obs[2]
        nearest_circ_speed = obs[3]
        gap_size = obs[4]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        try:
            lane_id = env.sim.conn.vehicle.getLaneID(env.ego_id)
            lane_pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
        except Exception:
            lane_id = "none"
            lane_pos = 0.0
            
        ttc = env._get_ttc_after_merge()
        ttc_str = f"{ttc:.2f}" if ttc != float('inf') else "inf"
        
        # print telemetry at every 5th step, or when lane changes, or at the end
        if step % 5 == 1 or done or "exit" in lane_id:
            print(f"{step:<5d} | {ego_speed:<8.3f} | {lane_id:<20} | {lane_pos:<8.3f} | {action[0]:<8.3f} | {reward:<8.4f} | {ttc_str:<8} | {nearest_circ_dist:<15.3f} | {nearest_circ_speed:<15.3f}")
            
    print("-" * 120)
    print(f"Episode finished in {step} steps. Reason: {info.get('termination_reason')}")
    env.close()

if __name__ == "__main__":
    check_speed()
