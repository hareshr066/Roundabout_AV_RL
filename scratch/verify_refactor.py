import os
import sys
import numpy as np

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.roundabout_env import RoundaboutEnv

def main():
    print("=" * 80)
    # 1. Initialize environment with Spatial Curriculum enabled
    env = RoundaboutEnv(
        use_spatial_curriculum=True,
        gui=False,
        label="verify_refactor"
    )
    
    # Reset once to start TraCI
    obs, info = env.reset()
    
    print("GEOMETRY VERIFICATION:")
    print("-" * 50)
    # Query lane lengths
    lane_lengths = {}
    for arm in ["N", "S", "E", "W"]:
        lane_id = f"entry_{arm}_0"
        length = env.sim.conn.lane.getLength(lane_id)
        lane_lengths[lane_id] = length
        print(f"  Lane {lane_id} length: {length:.3f} m")
    
    print("\nSPATIAL CURRICULUM STAGES VERIFICATION:")
    print("-" * 50)
    stages = [1, 2, 3, 4]
    stage_letters = {1: "A", 2: "B", 3: "C", 4: "D"}
    
    for stage in stages:
        # Force set the stage in SpatialCurriculumManager
        env.spatial_curriculum.current_stage = stage
        obs, info = env.reset()
        
        spawn_dist = env.active_spawn_distance
        ego_pos = env.sim.conn.vehicle.getLanePosition(env.ego_id)
        observed_dist = obs[1]
        
        # Calculate expected distance to entry: entry_length - ego_pos
        entry_length = lane_lengths["entry_N_0"]
        expected_dist = entry_length - ego_pos
        
        print(f"  Stage {stage_letters[stage]}:")
        print(f"    Target Spawn Distance from Entry: {spawn_dist:.1f} m")
        print(f"    SUMO Spawn Position along Lane  : {ego_pos:.3f} m")
        print(f"    Observed dist_to_entry (Obs[1]) : {observed_dist:.3f} m")
        print(f"    Expected dist_to_entry (calc)   : {expected_dist:.3f} m")
        
        match = abs(observed_dist - expected_dist) < 1e-4
        print(f"    Verification Match              : {'PASS' if match else 'FAIL'}")
    
    print("\nMERGE ZONE BOUNDARY:")
    print("-" * 50)
    print("  Merge Zone Boundary: dist_to_entry <= 30.0 m")
    
    print("\nEGO PROPAGATION TO MERGE ZONE VERIFICATION:")
    print("-" * 50)
    # Reset in Stage D (Full entry length, spawns at 0m position)
    env.spatial_curriculum.current_stage = 4
    obs, info = env.reset()
    
    print(f"  Reset at Stage D. Initial dist_to_entry: {obs[1]:.3f} m")
    
    reached_merge = False
    for step in range(1, 200):
        # Action: Apply maximum acceleration (2.0 m/s^2) to drive forward
        action = np.array([2.0], dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        ego_pos = env.sim.conn.vehicle.getLanePosition(env.ego_id) if env.ego_id in env.sim.conn.vehicle.getIDList() else 0.0
        dist_to_entry = obs[1]
        
        if dist_to_entry <= 30.0 and not reached_merge:
            reached_merge = True
            print(f"  Reached Merge Zone (<= 30m) at Step {step}:")
            print(f"    Ego Position along Lane  : {ego_pos:.3f} m")
            print(f"    Observed dist_to_entry   : {dist_to_entry:.3f} m")
            break
            
    print(f"  Simulation Verification           : {'PASS' if reached_merge else 'FAIL'}")
    
    env.close()
    print("=" * 80)

if __name__ == "__main__":
    main()
