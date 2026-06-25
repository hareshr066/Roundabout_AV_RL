import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def run_gridlock_check():
    model_path = "results/models/agent_spatial_curriculum.zip"
    print(f"Loading PPO model: {model_path}")
    model = PPO.load(model_path)
    
    # We will test VERY_HIGH density which is the most prone to gridlock
    env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False, max_steps=800, verbose=False, traffic_density="very_high")
    
    obs, info = env.reset()
    done = False
    step = 0
    
    print("\n" + "="*80)
    print("                      ROUNDABOUT GRIDLOCK AUDIT")
    print("="*80)
    print(f"{'Step':<5} | {'Active Vehs':<12} | {'Avg Speed (m/s)':<17} | {'Ego Lane ID':<15} | {'Ego Speed':<9}")
    print("-"*80)
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step += 1
        
        veh_list = env.sim.conn.vehicle.getIDList()
        num_vehs = len(veh_list)
        
        if num_vehs > 0:
            avg_speed = np.mean([env.sim.conn.vehicle.getSpeed(v) for v in veh_list])
        else:
            avg_speed = 0.0
            
        ego_exists = env.ego_id in veh_list
        if ego_exists:
            ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
            ego_speed = env.sim.conn.vehicle.getSpeed(env.ego_id)
        else:
            ego_lane = "[REMOVED]"
            ego_speed = 0.0
            
        if step % 20 == 0 or done:
            print(f"{step:<5} | {num_vehs:<12} | {avg_speed:<17.2f} | {ego_lane:<15} | {ego_speed:<9.2f}")
            
    reason = step_info.get("termination_reason", "timeout").upper()
    print("-"*80)
    print(f"Episode ended at step {step} with outcome: {reason}")
    print("="*80 + "\n")
    
    env.close()

if __name__ == "__main__":
    run_gridlock_check()
