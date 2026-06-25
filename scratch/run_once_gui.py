import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.roundabout_env import RoundaboutEnv

def run_single_episode_gui():
    model_path = "results/models/agent_spatial_curriculum_30k.zip"
    
    print("=" * 80)
    print("               RUNNING ROUNDABOUT AV SIMULATION ONCE (GUI)")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Falling back to random actions...")
        model = None
    else:
        print(f"Loading trained PPO policy from: {model_path}")
        model = PPO.load(model_path)
        print("Model loaded successfully!")
        
    print("\nLaunching SUMO-GUI...")
    # Initialize environment with GUI enabled
    env = RoundaboutEnv(
        gui=True,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        max_steps=400,
        label="single_run_gui"
    )
    
    try:
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        
        # Try to zoom and track ego vehicle in SUMO-GUI
        try:
            time.sleep(1.0) # Wait for GUI to render
            env.sim.conn.gui.trackVehicle("View #0", env.ego_id)
            env.sim.conn.gui.setZoom("View #0", 200.0)
        except Exception as e:
            print(f"Note: Could not set GUI track/zoom: {e}")
            
        print("\nStarting episode steps...")
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward
            
            # Print telemetry details
            print(f"Step {steps:3d} | Speed: {obs[0]:5.2f} m/s | Dist to Entry: {obs[1]:6.2f} m | Circ Gap: {obs[4]:5.2f} m | Action Accel: {action[0]:+5.2f} m/s^2 | Reward: {reward:+6.2f}")
            
            # Control simulation speed for viewing
            time.sleep(0.1)
            
        outcome = step_info.get("termination_reason", "timeout").upper()
        print("\n" + "=" * 80)
        print("                            RUN COMPLETED")
        print("=" * 80)
        print(f"Outcome:           {outcome}")
        print(f"Total Steps:       {steps} ({steps * env.dt:.1f} seconds of simulation)")
        print(f"Total Reward:      {total_reward:.2f}")
        print(f"Avg Speed:         {steps * env.dt / steps if steps > 0 else 0.0:.2f} m/s")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
    finally:
        env.close()
        print("\nSUMO-GUI closed.")

if __name__ == "__main__":
    run_single_episode_gui()
