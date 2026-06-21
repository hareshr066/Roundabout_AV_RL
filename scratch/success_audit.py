import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def run_success_audit():
    model_path = "results/models/agent_spatial_curriculum_30k.zip"
    print("=" * 80)
    print("                    ROUNDABOUT SUCCESS DETECTION AUDIT")
    print("=" * 80)
    print(f"Loading latest model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    model = PPO.load(model_path)
    
    # Initialize standard evaluation env
    env = RoundaboutEnv(
        gui=False,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        max_steps=600,  # Default max steps
        label="success_audit"
    )
    
    num_episodes = 20
    audit_results = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        # Trackers
        entered_ring = False
        completed_25 = False
        completed_50 = False
        reached_exit_lane = False
        removed_from_sim = False
        termination_reason = "TIMEOUT"
        
        last_lane_id = "none"
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Check ego existence and position via TraCI directly for debugging
            try:
                ego_exists = env.ego_id in env.sim.conn.vehicle.getIDList()
            except Exception:
                ego_exists = False
                
            if ego_exists:
                try:
                    lane_id = env.sim.conn.vehicle.getLaneID(env.ego_id)
                    last_lane_id = lane_id
                    
                    if "circ" in lane_id:
                        entered_ring = True
                        
                    if "circ_W_S" in lane_id:
                        completed_25 = True  # Completed first quadrant
                        
                    if "exit_S" in lane_id:
                        reached_exit_lane = True
                        completed_50 = True  # Completed second quadrant (reached exit)
                        
                except Exception as e:
                    pass
            else:
                removed_from_sim = True
                
        # Determine outcome from environment info
        outcome = step_info.get("termination_reason", "timeout").upper()
        
        audit_results.append({
            "episode": ep + 1,
            "entered_ring": "YES" if entered_ring else "NO",
            "completed_25": "YES" if completed_25 else "NO",
            "completed_50": "YES" if completed_50 else "NO",
            "reached_exit_lane": "YES" if reached_exit_lane else "NO",
            "removed_from_sim": "YES" if removed_from_sim else "NO",
            "termination_reason": outcome,
            "total_steps": steps,
            "last_lane": last_lane_id
        })
        
        print(f"Ep {ep+1:2d} | Ring: {audit_results[-1]['entered_ring']} | 25%: {audit_results[-1]['completed_25']} | ExitLane: {audit_results[-1]['reached_exit_lane']} | Removed: {audit_results[-1]['removed_from_sim']} | Reason: {audit_results[-1]['termination_reason']:<10} | Steps: {steps:3d} | Last Lane: {last_lane_id}")

    env.close()
    
    # Audit summary
    print("\n" + "=" * 80)
    print("                             AUDIT SUMMARY")
    print("=" * 80)
    entered_ring_count = sum(1 for r in audit_results if r["entered_ring"] == "YES")
    completed_25_count = sum(1 for r in audit_results if r["completed_25"] == "YES")
    reached_exit_count = sum(1 for r in audit_results if r["reached_exit_lane"] == "YES")
    removed_count = sum(1 for r in audit_results if r["removed_from_sim"] == "YES")
    
    success_count = sum(1 for r in audit_results if r["termination_reason"] == "SUCCESS")
    collision_count = sum(1 for r in audit_results if r["termination_reason"] == "COLLISION")
    timeout_count = sum(1 for r in audit_results if r["termination_reason"] == "TIMEOUT")
    
    print(f"Total Episodes Evaluated: {num_episodes}")
    print(f"Entered Ring:            {entered_ring_count}/{num_episodes} ({entered_ring_count/num_episodes*100:.1f}%)")
    print(f"Completed 25%:           {completed_25_count}/{num_episodes} ({completed_25_count/num_episodes*100:.1f}%)")
    print(f"Reached Exit Lane:       {reached_exit_count}/{num_episodes} ({reached_exit_count/num_episodes*100:.1f}%)")
    print(f"Removed from Simulation: {removed_count}/{num_episodes} ({removed_count/num_episodes*100:.1f}%)")
    print("-" * 80)
    print(f"SUCCESS Outcome:         {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"COLLISION Outcome:       {collision_count}/{num_episodes} ({collision_count/num_episodes*100:.1f}%)")
    print(f"TIMEOUT Outcome:         {timeout_count}/{num_episodes} ({timeout_count/num_episodes*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    run_success_audit()
