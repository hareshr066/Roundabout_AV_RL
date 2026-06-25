import os
import sys
import numpy as np
import time

# Ensure workspace root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.roundabout_env import RoundaboutEnv

def collect_metrics():
    print("Starting simulation to collect precise traffic metrics...")
    env = RoundaboutEnv(
        gui=False,
        traffic_density="very_high",
        fixed_hdv_ratio=0.50,
        max_steps=500,
        label="metrics_collector"
    )
    
    obs, info = env.reset()
    
    total_vehs_in_network = []
    total_vehs_in_circ = []
    
    hdv_speeds = []
    av_speeds = []
    
    hdv_accels = []
    av_accels = []
    
    hdv_headways = []
    av_headways = []
    
    prev_speeds = {}
    
    done = False
    step_count = 0
    
    while not done:
        # Step with a dummy zero action for ego
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        try:
            veh_ids = env.sim.conn.vehicle.getIDList()
        except Exception:
            break
            
        bg_veh_ids = [v for v in veh_ids if v != "ego"]
        
        total_vehs_in_network.append(len(bg_veh_ids))
        
        circ_count = 0
        for veh in bg_veh_ids:
            try:
                lane = env.sim.conn.vehicle.getLaneID(veh)
                speed = env.sim.conn.vehicle.getSpeed(veh)
                vtype = env.sim.conn.vehicle.getTypeID(veh)
                
                # Check if in circulating ring
                if "circ" in lane:
                    circ_count += 1
                    
                # Track speed
                if "passenger_car" in vtype:
                    hdv_speeds.append(speed)
                elif "av_car" in vtype:
                    av_speeds.append(speed)
                    
                # Compute accel
                if veh in prev_speeds:
                    accel = (speed - prev_speeds[veh]) / 0.1
                    if "passenger_car" in vtype:
                        hdv_accels.append(accel)
                    elif "av_car" in vtype:
                        av_accels.append(accel)
                prev_speeds[veh] = speed
                
                # Track headway
                leader_info = env.sim.conn.vehicle.getLeader(veh, dist=100.0)
                if leader_info is not None:
                    leader_id, gap = leader_info
                    if leader_id:
                        if "passenger_car" in vtype:
                            hdv_headways.append(gap)
                        elif "av_car" in vtype:
                            av_headways.append(gap)
            except Exception:
                pass
                
        total_vehs_in_circ.append(circ_count)
        
    env.close()
    
    print("\n" + "=" * 40)
    print("         TRAFFIC METRICS DIAGNOSIS")
    print("=" * 40)
    print(f"Avg Vehicles in Network:          {np.mean(total_vehs_in_network):.2f}")
    print(f"Avg Vehicles in Circulating Ring: {np.mean(total_vehs_in_circ):.2f}")
    print(f"Avg HDV Speed:                    {np.mean(hdv_speeds):.2f} m/s")
    print(f"Avg AV Speed:                     {np.mean(av_speeds):.2f} m/s")
    print(f"HDV Speed StdDev:                 {np.std(hdv_speeds):.2f} m/s")
    print(f"AV Speed StdDev:                  {np.std(av_speeds):.2f} m/s")
    print(f"Avg HDV Accel:                    {np.mean(hdv_accels):.2f} m/s^2")
    print(f"Avg AV Accel:                     {np.mean(av_accels):.2f} m/s^2")
    print(f"HDV Accel StdDev:                 {np.std(hdv_accels):.2f} m/s^2")
    print(f"AV Accel StdDev:                  {np.std(av_accels):.2f} m/s^2")
    print(f"Avg HDV Headway:                  {np.mean(hdv_headways):.2f} m")
    print(f"Avg AV Headway:                   {np.mean(av_headways):.2f} m")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    collect_metrics()
