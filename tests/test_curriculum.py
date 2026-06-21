import os
import sys

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

def test_curriculum_advancement():
    print("Initializing RoundaboutEnv with a curriculum window of 5 episodes for quick verification...")
    # Set target_success_rate = 0.80 and window = 5.
    # This means the stage will advance if at least 4 out of 5 episodes succeed.
    env = RoundaboutEnv(
        config_file="sumo_network/roundabout.sumocfg",
        gui=False,
        max_steps=50,
        target_success_rate=0.80,
        curriculum_window=5
    )
    
    try:
        # Check starting condition
        print(f"\nInitial State: Stage {env.curriculum.current_stage} ({env.curriculum.current_description})")
        print(f"Initial HDV Ratio: {env.curriculum.current_hdv_ratio}")
        
        # --- Stage 1 -> Stage 2 Transition Test ---
        print("\n[Test Phase 1] Simulating 5 consecutive successful episodes...")
        for ep in range(5):
            env.reset()
            # Force mock outcome: Success
            # We step the environment and then trigger termination manually to mock outcomes
            obs, reward, terminated, truncated, info = env.step([1.0])  # acceleration command
            
            # Manually inject success outcome to test metrics updates
            # (Simulating that the agent successfully merged)
            success = True
            env.curriculum.update_metrics(success)
            
            status = env.curriculum.get_status()
            print(f"  Episode {ep+1} | Success: {success} | Rolling Success Rate: {status['rolling_success_rate']*100:.1f}% | Stage: {status['current_stage']}")
            
        print(f"\nAfter Phase 1: Current Stage is {env.curriculum.current_stage} (HDV Ratio: {env.curriculum.current_hdv_ratio})")
        assert env.curriculum.current_stage == 2, f"Expected Stage 2, got Stage {env.curriculum.current_stage}"
        print("Stage 1 -> Stage 2 transition PASSED.")
        
        # --- Stage 2 -> Stage 3 Transition Test ---
        print("\n[Test Phase 2] Simulating 3 successes and 2 failures (Success Rate: 60%, should NOT advance)...")
        outcomes = [True, False, True, False, True]
        for ep, success in enumerate(outcomes):
            env.reset()
            env.step([0.5])
            env.curriculum.update_metrics(success)
            status = env.curriculum.get_status()
            print(f"  Episode {ep+1} | Success: {success} | Rolling Success Rate: {status['rolling_success_rate']*100:.1f}% | Stage: {status['current_stage']}")
            
        print(f"\nAfter Phase 2: Current Stage is {env.curriculum.current_stage} (HDV Ratio: {env.curriculum.current_hdv_ratio})")
        assert env.curriculum.current_stage == 2, f"Expected Stage 2, got Stage {env.curriculum.current_stage}"
        print("Non-advancement safety rule verification PASSED.")
        
        # --- Stage 2 -> Stage 3 Success Test ---
        print("\n[Test Phase 3] Simulating 5 consecutive successes to trigger advancement from Stage 2 -> Stage 3...")
        for ep in range(5):
            env.reset()
            env.step([1.0])
            env.curriculum.update_metrics(True)
            status = env.curriculum.get_status()
            print(f"  Episode {ep+1} | Success: True | Rolling Success Rate: {status['rolling_success_rate']*100:.1f}% | Stage: {status['current_stage']}")
            
        print(f"\nAfter Phase 3: Current Stage is {env.curriculum.current_stage} (HDV Ratio: {env.curriculum.current_hdv_ratio})")
        assert env.curriculum.current_stage == 3, f"Expected Stage 3, got Stage {env.curriculum.current_stage}"
        print("Stage 2 -> Stage 3 transition PASSED.")
        
        print("\nAll curriculum manager functionality tests PASSED!")
        
    except Exception as e:
        print(f"\nCurriculum test FAILED with error: {e}", file=sys.stderr)
        raise e
        
    finally:
        print("Closing environment...")
        env.close()
        print("Environment closed cleanly.")

if __name__ == "__main__":
    test_curriculum_advancement()
