import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def check_determinism():
    print("Checking determinism between environment resets...")
    env = RoundaboutEnv(
        gui=False,
        traffic_density="medium",
        fixed_hdv_ratio=0.50,
        label="determinism_check"
    )
    
    # Run Reset 1
    env.reset()
    vehs_run1 = []
    for _ in range(50):
        env.sim.step()
        env._apply_dynamic_traffic_types()
        vehs_run1.append(list(env.sim.conn.vehicle.getIDList()))
        
    env.close()
    
    # Run Reset 2
    env.reset()
    vehs_run2 = []
    for _ in range(50):
        env.sim.step()
        env._apply_dynamic_traffic_types()
        vehs_run2.append(list(env.sim.conn.vehicle.getIDList()))
        
    env.close()
    
    # Compare
    identical = True
    for i in range(50):
        if vehs_run1[i] != vehs_run2[i]:
            identical = False
            print(f"Difference at step {i}:")
            print(f"  Run 1: {vehs_run1[i][:5]} ... (total {len(vehs_run1[i])})")
            print(f"  Run 2: {vehs_run2[i][:5]} ... (total {len(vehs_run2[i])})")
            break
            
    if identical:
        print("SUCCESS/WARNING: Resets are 100% visually identical! The background traffic is completely deterministic.")
    else:
        print("INFO: Resets are different. Background traffic has stochasticity.")
        
if __name__ == "__main__":
    check_determinism()
