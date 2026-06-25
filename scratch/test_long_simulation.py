import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def test_long_sim():
    print("Running long simulation to observe traffic gridlocks...")
    # Initialize env
    env = RoundaboutEnv(
        gui=False,
        traffic_density="very_high",
        fixed_hdv_ratio=0.50,
        max_steps=1500,
        label="long_sim_audit"
    )
    
    # We will bypass the ego termination condition in our step loop by running the step
    # but not stopping the loop when ego finishes or terminates, or we can just step TraCI directly!
    # Yes, stepping TraCI directly is much simpler and avoids Gym environment reset/termination.
    
    env.sim.reset()
    
    # Let's run for 1500 steps (150 seconds of simulation time)
    step = 0
    max_steps = 1500
    
    # We want to log vehicle counts on all edges every 50 steps
    edges = ["entry_N", "entry_E", "entry_S", "entry_W", 
             "circ_N_W", "circ_W_S", "circ_S_E", "circ_E_N",
             "exit_N", "exit_E", "exit_S", "exit_W"]
             
    print(f"{'Step':<5} | " + " | ".join([f"{e:<10}" for e in edges]))
    print("-" * 145)
    
    while step < max_steps:
        env.sim.step()
        env._apply_dynamic_traffic_types()
        step += 1
        
        if step % 100 == 0:
            counts = []
            for edge in edges:
                try:
                    vehs = env.sim.conn.edge.getLastStepVehicleIDs(edge)
                    counts.append(len(vehs))
                except Exception:
                    counts.append(-1)
            print(f"{step:<5} | " + " | ".join([f"{c:<10}" for c in counts]))
            
            # Let's check if there are any stopped vehicles on exit_S
            try:
                exit_S_vehs = env.sim.conn.edge.getLastStepVehicleIDs("exit_S")
                if len(exit_S_vehs) > 0:
                    stopped = []
                    for v in exit_S_vehs:
                        spd = env.sim.conn.vehicle.getSpeed(v)
                        pos = env.sim.conn.vehicle.getLanePosition(v)
                        if spd < 0.1:
                            stopped.append(f"{v}(pos={pos:.1f}m)")
                    if stopped:
                        print(f"  --> Stopped on exit_S: {', '.join(stopped)}")
            except Exception:
                pass

    env.close()

if __name__ == "__main__":
    test_long_sim()
