import os
import sys
import time
import numpy as np
import subprocess
from stable_baselines3 import PPO

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.roundabout_env import RoundaboutEnv

def audit():
    print("=" * 80)
    print("                      ROUNDABOUT RL SYSTEM AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. PPO MODEL AUDIT
    # -------------------------------------------------------------
    print("\n--- 1. PPO MODEL AUDIT ---")
    model_path = os.path.abspath("results/models/agent_spatial_curriculum_30k.zip")
    print(f"Absolute Model Path: {model_path}")
    exists = os.path.exists(model_path)
    print(f"File Exists:         {exists}")
    if exists:
        print(f"File Size:           {os.path.getsize(model_path):,} bytes")
        mtime = os.path.getmtime(model_path)
        print(f"Last Modified:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    else:
        print("WARNING: Model file does not exist!")

    # -------------------------------------------------------------
    # 2. SUMO NETWORK AUDIT
    # -------------------------------------------------------------
    print("\n--- 2. SUMO NETWORK AUDIT ---")
    net_path = os.path.abspath("sumo_network/roundabout.net.xml")
    print(f"Network File Path:   {net_path}")
    net_exists = os.path.exists(net_path)
    print(f"Network Exists:      {net_exists}")
    
    # Read the net.xml file to parse roundabout details or lengths
    if net_exists:
        # Check node positions and edge shapes if possible
        with open(net_path, "r", encoding="utf-8") as f:
            content = f.read()
        print("Parsing network dimensions from net.xml...")
        # Check node y coordinates for N and S
        # node id="N" x="0" y="30"
        # node id="N_far" x="0" y="200"
        # Let's search for length of entry_N
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(net_path)
            root = tree.getroot()
            # Find lane length for entry_N_0
            for lane in root.findall(".//lane"):
                lane_id = lane.get("id")
                if lane_id == "entry_N_0":
                    print(f"  Lane 'entry_N_0' Length: {lane.get('length')} m (Width: {lane.get('width')} m)")
                if lane_id == "circ_N_W_0":
                    print(f"  Lane 'circ_N_W_0' Length: {lane.get('length')} m")
        except Exception as e:
            print(f"  Error parsing XML: {e}")
    else:
        print("WARNING: Network file does not exist!")

    # -------------------------------------------------------------
    # 3. ROUTES AUDIT
    # -------------------------------------------------------------
    print("\n--- 3. ROUTES AUDIT ---")
    routes_path = os.path.abspath("sumo_network/routes.xml")
    print(f"Routes File Path:    {routes_path}")
    routes_exists = os.path.exists(routes_path)
    print(f"Routes Exists:       {routes_exists}")
    if routes_exists:
        try:
            tree = ET.parse(routes_path)
            root = tree.getroot()
            # Check ego route 'r_N_S'
            for route in root.findall(".//route"):
                if route.get("id") == "r_N_S":
                    print(f"  Ego Route 'r_N_S' Edges: {route.get('edges')}")
            # Check vTypes
            for vtype in root.findall(".//vType"):
                print(f"  vType: id={vtype.get('id'):<15} | guiShape={vtype.get('guiShape'):<15} | color={vtype.get('color')}")
        except Exception as e:
            print(f"  Error parsing routes XML: {e}")

    # -------------------------------------------------------------
    # 4, 5, 6, 7. EGO SPAWN, OBSERVATION, POLICY, ENV AUDIT (Run simulation)
    # -------------------------------------------------------------
    print("\n--- 4-7. SIMULATION RUN AND RUNTIME AUDIT ---")
    if exists:
        model = PPO.load(model_path)
        env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False, label="audit_sim")
        
        print(f"Environment Settings:")
        print(f"  Max Steps:         {env.max_steps}")
        print(f"  Spawn Distance:    {env.active_spawn_distance}")
        print(f"  HDV Ratio:         {env.active_hdv_ratio}")
        print(f"  Curriculum Stage:  {env.curriculum.current_stage}")
        
        obs, info = env.reset()
        print(f"\nInitial State (Reset):")
        print(f"  Observation Vector: {obs}")
        print(f"  Info:               {info}")
        
        # Check initial spawn status
        try:
            ego_spawned = env.ego_id in env.sim.conn.vehicle.getIDList()
            print(f"  Ego Spawned immediately on Reset: {ego_spawned}")
            if ego_spawned:
                lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
                pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
                speed = env.sim.conn.vehicle.getSpeed(env.ego_id)
                color = env.sim.conn.vehicle.getColor(env.ego_id)
                print(f"  Ego ID: {env.ego_id} | Lane: {lane} | Position: {pos:.2f} m | Speed: {speed:.2f} m/s | Color: {color}")
        except Exception as e:
            print(f"  Ego check failed on reset: {e}")

        # Telemetry trace
        print(f"\nPrinting first 20 simulation steps telemetry:")
        print("-" * 120)
        print(f"{'Step':<5} | {'Action (Accel)':<15} | {'Ego Speed':<10} | {'Ego Pos':<10} | {'Ego Lane':<15} | {'Dist to Entry':<15} | {'Nearest Circ Dist':<18} | {'Gap Size':<10}")
        print("-" * 120)

        for step in range(1, 21):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
            if ego_exists:
                lane = env.sim.conn.vehicle.getLaneID(env.ego_id)
                pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
                speed = env.sim.conn.vehicle.getSpeed(env.ego_id)
            else:
                lane, pos, speed = "none", 0.0, 0.0
                
            print(f"{step:<5d} | {action[0]:<15.4f} | {speed:<10.3f} | {pos:<10.3f} | {lane:<15} | {next_obs[1]:<15.3f} | {next_obs[2]:<18.3f} | {next_obs[4]:<10.3f}")
            obs = next_obs
            
        print("-" * 120)
        env.close()
    else:
        print("Skipping simulation run because model checkpoint is missing!")

    # -------------------------------------------------------------
    # 8. FILE DIFFERENCE AUDIT
    # -------------------------------------------------------------
    print("\n--- 8. FILE DIFFERENCE AUDIT ---")
    # Compare files using git diff/status
    try:
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        print("Git Status (modified/untracked files):")
        if status_res.stdout.strip():
            print(status_res.stdout)
        else:
            print("  No modified or untracked files.")
            
        # Get last commit hash
        commit_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        print(f"Current Git Commit: {commit_res.stdout.strip()}")
    except Exception as e:
        print(f"Git check failed: {e}")

if __name__ == "__main__":
    audit()
