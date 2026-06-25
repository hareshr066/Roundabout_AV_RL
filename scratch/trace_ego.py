import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def run_trace():
    model_path = "results/models/final_agent_b_curriculum.zip"
    model = PPO.load(model_path)
    
    env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False, max_steps=800, verbose=False, traffic_density="low")
    
    obs, info = env.reset()
    done = False
    step = 0
    
    lines = []
    lines.append("="*80)
    lines.append("                FINAL AGENT B CURRICULUM TRAJECTORY TRACE (ALL STEPS)")
    lines.append("="*80)
    lines.append(f"{'Step':<5} | {'Lane ID':<15} | {'Edge ID':<12} | {'Speed (m/s)':<12} | {'Route Idx':<9} | {'Remaining Route'}")
    lines.append("-"*80)
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step += 1
        
        ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
        if ego_exists:
            lane_id = env.sim.conn.vehicle.getLaneID(env.ego_id)
            edge_id = env.sim.conn.vehicle.getRoadID(env.ego_id)
            speed = env.sim.conn.vehicle.getSpeed(env.ego_id)
            try:
                route_index = env.sim.conn.vehicle.getRouteIndex(env.ego_id)
                route = env.sim.conn.vehicle.getRoute(env.ego_id)
                remaining_route = route[route_index:]
            except Exception:
                route_index = -1
                remaining_route = []
            
            lines.append(f"{step:<5} | {lane_id:<15} | {edge_id:<12} | {speed:<12.2f} | {route_index:<9} | {remaining_route}")
        else:
            lines.append(f"{step:<5} | [REMOVED]       | -            | -            | -         | -")
            break
            
    reason = step_info.get("termination_reason", "timeout").upper()
    lines.append("-"*80)
    lines.append(f"Episode ended at step {step} with outcome: {reason}")
    lines.append("="*80 + "\n")
    
    with open("scratch/trace_output.txt", "w") as f:
        f.write("\n".join(lines))
        
    env.close()

if __name__ == "__main__":
    run_trace()
