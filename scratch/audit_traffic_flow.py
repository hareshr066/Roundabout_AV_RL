import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def audit_flow():
    model_path = "results/models/agent_spatial_curriculum_30k.zip"
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # We will run with VERY_HIGH traffic density and 0.50 HDV ratio
    env = RoundaboutEnv(
        gui=False,
        traffic_density="very_high",
        fixed_hdv_ratio=0.50,
        max_steps=600,
        label="flow_audit_sim"
    )
    
    obs, info = env.reset()
    done = False
    step = 0
    
    # Track statistics over time
    entry_E_queues = []
    exit_S_queues = []
    circ_W_S_queues = []
    
    # Waiting times
    waiting_times = {} # veh_id -> time spent at speed < 0.1 m/s
    
    # Speeds on exit_S
    exit_S_speeds = []
    
    # Keep track of vehicles on exit_S
    exit_S_vehs_record = []

    print("Running simulation for flow and congestion audit...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step += 1
        
        # Get vehicles on entry_E, exit_S, circ_W_S
        try:
            # All active vehicles
            all_vehs = env.sim.conn.vehicle.getIDList()
            
            # Count queues (vehicles with speed < 0.1 m/s on these lanes)
            q_entry_E = 0
            q_exit_S = 0
            q_circ_W_S = 0
            
            vehs_on_exit_S = []
            
            for veh in all_vehs:
                lane = env.sim.conn.vehicle.getLaneID(veh)
                speed = env.sim.conn.vehicle.getSpeed(veh)
                
                # Track waiting time (speed < 0.1 m/s)
                if speed < 0.1:
                    waiting_times[veh] = waiting_times.get(veh, 0) + env.dt
                else:
                    # Reset or just keep accumulating? Usually we accumulate total waiting time in the episode.
                    pass
                
                if lane == "entry_E_0":
                    if speed < 0.1:
                        q_entry_E += 1
                elif lane == "exit_S_0":
                    if speed < 0.1:
                        q_exit_S += 1
                    exit_S_speeds.append(speed)
                    vehs_on_exit_S.append((veh, env.sim.conn.vehicle.getLanePosition(veh), speed))
                elif lane == "circ_W_S_0":
                    if speed < 0.1:
                        q_circ_W_S += 1
            
            entry_E_queues.append(q_entry_E)
            exit_S_queues.append(q_exit_S)
            circ_W_S_queues.append(q_circ_W_S)
            
            if len(vehs_on_exit_S) > 0:
                exit_S_vehs_record.append((step, vehs_on_exit_S))
                
        except Exception as e:
            print(f"TraCI retrieval error at step {step}: {e}")
            break
            
    env.close()
    
    # Calculate averages
    avg_q_entry_E = np.mean(entry_E_queues) if entry_E_queues else 0.0
    avg_q_exit_S = np.mean(exit_S_queues) if exit_S_queues else 0.0
    avg_q_circ_W_S = np.mean(circ_W_S_queues) if circ_W_S_queues else 0.0
    
    max_q_entry_E = np.max(entry_E_queues) if entry_E_queues else 0
    max_q_exit_S = np.max(exit_S_queues) if exit_S_queues else 0
    max_q_circ_W_S = np.max(circ_W_S_queues) if circ_W_S_queues else 0
    
    avg_wait_time = np.mean(list(waiting_times.values())) if waiting_times else 0.0
    max_wait_time = np.max(list(waiting_times.values())) if waiting_times else 0.0
    
    print("\n" + "="*80)
    print("                      CONGESTION & QUEUE AUDIT RESULTS")
    print("="*80)
    print(f"Total simulated steps: {step}")
    print(f"Average Queue Lengths (Speed < 0.1 m/s):")
    print(f"  - entry_E_0:  {avg_q_entry_E:.2f} vehs (Max: {max_q_entry_E})")
    print(f"  - exit_S_0:   {avg_q_exit_S:.2f} vehs (Max: {max_q_exit_S})")
    print(f"  - circ_W_S_0: {avg_q_circ_W_S:.2f} vehs (Max: {max_q_circ_W_S})")
    print("-" * 80)
    print(f"Average Waiting Time: {avg_wait_time:.2f} s (Max: {max_wait_time:.2f} s)")
    print(f"Average Speed on exit_S_0: {np.mean(exit_S_speeds):.2f} m/s" if exit_S_speeds else "No vehicles on exit_S_0")
    
    # Print out why vehicles queue on exit_S
    print("\nSample records of vehicles on exit_S_0:")
    # Look at the end of the simulation
    for record in exit_S_vehs_record[-5:]:
        s, vehs = record
        print(f"  Step {s}:")
        for v_id, pos, spd in sorted(vehs, key=lambda x: x[1]):
            print(f"    Vehicle {v_id:<12} | Position: {pos:.2f} m | Speed: {spd:.2f} m/s")
            
    print("="*80 + "\n")

if __name__ == "__main__":
    audit_flow()
