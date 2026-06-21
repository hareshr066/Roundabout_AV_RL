import os
import sys

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def test_roundabout_gym_env():
    print("Initializing RoundaboutEnv...")
    env = RoundaboutEnv(
        config_file="sumo_network/roundabout.sumocfg",
        gui=False,
        max_steps=50
    )
    
    try:
        # 1. Reset Environment
        print("\n[Step 1] Resetting environment...")
        obs, info = env.reset()
        print("Initial Observation:")
        print(f"  Ego Speed: {obs[0]:.2f} m/s")
        print(f"  Distance to Entry: {obs[1]:.2f} m")
        print(f"  Nearest Circulating Vehicle Distance: {obs[2]:.2f} m")
        print(f"  Nearest Circulating Vehicle Speed: {obs[3]:.2f} m/s")
        print(f"  Circulating Gap Size: {obs[4]:.2f} m")
        print(f"  Current HDV Ratio: {obs[5]:.2f}")
        print(f"Initial Info: {info}")
        
        # Verify observation shape
        assert obs.shape == env.observation_space.shape, f"Observation shape mismatch! Expected {env.observation_space.shape}, got {obs.shape}"
        assert env.action_space.low[0] == -4.0, "Action space low boundary mismatch!"
        assert env.action_space.high[0] == 2.0, "Action space high boundary mismatch!"
        print("Observation and Action space assertions PASSED.")
        
        # 2. Step the Environment with random actions
        print("\n[Step 2] Stepping environment with random acceleration inputs...")
        print("-" * 120)
        print(f"{'Step':<5} | {'Action':<7} | {'Ego Speed':<10} | {'DistEntry':<10} | {'CircDist':<9} | {'CircSpd':<8} | {'GapSize':<8} | {'HDVRatio':<8} | {'Reward':<8} | {'Done':<5}")
        print("-" * 120)
        
        for step in range(30):
            # Sample acceleration in [-4.0, +2.0]
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            print(f"{step+1:<5} | {action[0]:<7.2f} | {obs[0]:<10.2f} | {obs[1]:<10.2f} | {obs[2]:<9.2f} | {obs[3]:<8.2f} | {obs[4]:<8.2f} | {obs[5]:<8.2f} | {reward:<8.4f} | {str(done):<5}")
            
            if done:
                print("-" * 120)
                print(f"Episode finished! Reason: {info.get('termination_reason')}")
                break
        else:
            print("-" * 120)
            
        print("\nGymnasium environment interaction test PASSED!")
        
    except Exception as e:
        print(f"\nGymnasium environment test FAILED with error: {e}", file=sys.stderr)
        raise e
        
    finally:
        print("Closing environment...")
        env.close()
        print("Environment closed cleanly.")

if __name__ == "__main__":
    test_roundabout_gym_env()
