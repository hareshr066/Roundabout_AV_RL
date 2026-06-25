import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def diagnose_stops():
    print("Diagnosing stopped vehicles at the entrance and exit lanes...")
    env = RoundaboutEnv(
        gui=False,
        traffic_density="very_high",
        fixed_hdv_ratio=0.50,
        max_steps=400,
        label="diag_stops_sim"
    )
    
    env.sim.reset()
    
    # Run for 300 steps (30 seconds) to let queues build up
    for step in range(1, 301):
        env.sim.step()
        env._apply_dynamic_traffic_types()
        
        # Check every 50 steps
        if step % 50 == 0:
            print(f"\n--- Step {step} Diagnosis ---")
            
            # Check all entry and exit lanes
            lanes = ["entry_N_0", "entry_E_0", "entry_S_0", "entry_W_0",
                     "circ_N_W_0", "circ_W_S_0", "circ_S_E_0", "circ_E_N_0",
                     "exit_N_0", "exit_E_0", "exit_S_0", "exit_W_0"]
                     
            for lane in lanes:
                try:
                    vehs = env.sim.conn.lane.getLastStepVehicleIDs(lane)
                except Exception:
                    continue
                if not vehs:
                    continue
                
                # Sort vehicles by lane position (descending so front of the lane is first)
                vehs_with_pos = []
                for v in vehs:
                    try:
                        pos = env.sim.conn.vehicle.getLanePosition(v)
                        speed = env.sim.conn.vehicle.getSpeed(v)
                        vehs_with_pos.append((v, pos, speed))
                    except Exception:
                        pass
                vehs_with_pos.sort(key=lambda x: x[1], reverse=True)
                
                # Print information about the front vehicle in this lane
                front_v, front_pos, front_speed = vehs_with_pos[0]
                if front_speed < 0.5:
                    # It's stopped or slow
                    try:
                        leader_info = env.sim.conn.vehicle.getLeader(front_v, dist=50.0)
                        leader_str = f"Leader: {leader_info}" if leader_info else "Leader: None"
                    except Exception:
                        leader_str = "Leader: Err"
                        
                    # Let's also check if it's yielding to someone
                    # For junctions, we can check next TLS or junction state
                    print(f"  Lane {lane:<12} | Front: {front_v:<15} | Pos: {front_pos:6.1f}m | Speed: {front_speed:4.2f}m/s | {leader_str}")
                    
                    # If there's a queue, print the queue size
                    stopped_count = sum(1 for v, p, s in vehs_with_pos if s < 0.1)
                    print(f"    Total vehicles on lane: {len(vehs)}, Stopped (speed < 0.1): {stopped_count}")
                else:
                    # Front vehicle is moving
                    print(f"  Lane {lane:<12} | Front: {front_v:<15} | Pos: {front_pos:6.1f}m | Speed: {front_speed:4.2f}m/s (Moving)")

    env.close()

if __name__ == "__main__":
    diagnose_stops()
