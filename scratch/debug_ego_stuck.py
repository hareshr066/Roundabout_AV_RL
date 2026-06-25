import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def debug_ego():
    model_path = "results/models/final_agent_b_curriculum.zip"
    if not os.path.exists(model_path):
        model_path = "results/models/agent_spatial_curriculum_30k.zip"
        
    print(f"Using PPO agent model: {model_path}")
    model = PPO.load(model_path)
    
    env = RoundaboutEnv(
        gui=False,
        traffic_density="medium",
        fixed_hdv_ratio=0.50,
        fixed_spawn_distance=80.0,
        max_steps=800,
        label="debug_ego_sim"
    )
    
    obs, info = env.reset()
    done = False
    step = 0
    
    # We want to catch when ego is stopped (speed < 0.1) for more than 50 steps
    ego_stopped_steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step += 1
        
        ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
        if ego_exists:
            ego_spd = env.sim.conn.vehicle.getSpeed(env.ego_id)
            ego_lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
            ego_pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
            
            if ego_spd < 0.1:
                ego_stopped_steps += 1
            else:
                ego_stopped_steps = 0
                
            if ego_stopped_steps >= 40:
                print(f"\n[ALERT] Ego stopped for {ego_stopped_steps} steps at step {step}!")
                print(f"Ego State: Lane={ego_lane}, Pos={ego_pos:.2f} m, Speed={ego_spd:.2f} m/s")
                
                # Check leader
                try:
                    leader = env.sim.conn.vehicle.getLeader(env.ego_id, 100.0)
                    print(f"Ego Leader: {leader}")
                    if leader:
                        l_id, l_dist = leader
                        l_lane = env.sim.conn.vehicle.getLaneID(l_id)
                        l_pos = env.sim.conn.vehicle.getLanePosition(l_id)
                        l_spd = env.sim.conn.vehicle.getSpeed(l_id)
                        print(f"  Leader State: ID={l_id}, Lane={l_lane}, Pos={l_pos:.2f} m, Speed={l_spd:.2f} m/s")
                except Exception as e:
                    print(f"  Error getting leader: {e}")
                    
                # Print all vehicles on nearby lanes: circ_W_S_0, exit_S_0, :S_1_0, entry_S_0
                print("Nearby Vehicles:")
                for lane in ["circ_W_S_0", "exit_S_0", ":S_1_0", "entry_S_0", ":S_0_0"]:
                    try:
                        vehs = env.sim.conn.lane.getLastStepVehicleIDs(lane)
                        if vehs:
                            print(f"  Lane {lane}:")
                            for v in vehs:
                                v_spd = env.sim.conn.vehicle.getSpeed(v)
                                v_pos = env.sim.conn.vehicle.getLanePosition(v)
                                print(f"    - {v}: Pos={v_pos:.2f} m, Speed={v_spd:.2f} m/s")
                    except Exception as e:
                        pass
                
                # Print all vehicles in the simulation
                all_vehs = env.sim.conn.vehicle.getIDList()
                print(f"Total vehicles in simulation: {len(all_vehs)}")
                
                # Stop simulation so we don't spam
                break
                
    env.close()

if __name__ == "__main__":
    debug_ego()
