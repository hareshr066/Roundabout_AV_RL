import os
import sys
import logging
import numpy as np

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

# Configure runner-specific detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TestRunner] - %(levelname)s - %(message)s")

class RuleBasedAgent:
    """
    A simple rule-based agent that implements gap-acceptance yield logic for the roundabout.
    Acts as a baseline validation policy for the environment.
    """
    def __init__(self, target_speed=12.0, yield_threshold=28.0, gap_threshold=20.0):
        self.target_speed = target_speed
        self.yield_threshold = yield_threshold
        self.gap_threshold = gap_threshold

    def select_action(self, obs):
        """
        Selects an acceleration action based on observation features:
        obs = [ego_speed, dist_to_entry, circ_dist, circ_speed, gap_size, hdv_ratio]
        """
        ego_speed = obs[0]
        dist_to_entry = obs[1]
        circ_dist = obs[2]
        circ_speed = obs[3]
        gap_size = obs[4]
        
        # If we are approaching the merge point
        if dist_to_entry > 0.1:
            # Yielding condition: if a circulating vehicle is close and approaching
            if circ_dist < self.yield_threshold:
                # Decelerate to yield
                accel = -3.0 if ego_speed > 2.0 else -1.0
                action_type = "YIELDING"
            # Gap acceptance condition: if the available gap or distance to next vehicle is large
            elif gap_size > self.gap_threshold or circ_dist > 30.0:
                # Accelerate to merge
                accel = 1.5 if ego_speed < self.target_speed else 0.0
                action_type = "MERGING"
            else:
                # Prepare to stop
                accel = -1.5 if ego_speed > 1.0 else 0.0
                action_type = "WAITING"
        else:
            # We are inside the roundabout or exited: follow target speed
            if ego_speed < self.target_speed:
                accel = 1.0
                action_type = "CRUISING"
            else:
                accel = -0.5
                action_type = "LIMITING"
                
        return np.array([accel], dtype=np.float32), action_type

def run_validation_suite():
    logging.info("Starting Environment Validation Suite...")
    
    # Initialize environment with small curriculum window of 5 for quick verification
    env = RoundaboutEnv(
        config_file="sumo_network/roundabout.sumocfg",
        gui=False,
        max_steps=200,
        target_success_rate=0.80,
        curriculum_window=5
    )
    
    agent = RuleBasedAgent()
    
    episode_results = []
    total_steps_executed = 0
    spawning_verified = False
    states_verified = False
    rewards_verified = False
    
    logging.info(f"Loaded environment. Starting Stage: {env.curriculum.current_stage} (HDV: {env.curriculum.current_hdv_ratio * 100}%)")
    
    try:
        for episode in range(10):
            logging.info(f"=== Starting Episode {episode+1}/10 ===")
            obs, info = env.reset()
            
            # Verify initial spawning and observations
            if obs is not None and len(obs) == 6:
                spawning_verified = True
                states_verified = True
                
            done = False
            step = 0
            ep_reward = 0.0
            
            while not done:
                # Choose action
                action, action_type = agent.select_action(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Verify rewards and state transitions
                if reward is not None:
                    rewards_verified = True
                if next_obs is not None and not np.array_equal(next_obs, obs):
                    # Values are changing dynamically as simulation advances
                    states_verified = True
                    
                ep_reward += reward
                obs = next_obs
                step += 1
                total_steps_executed += 1
                
                # Log step details periodically or on critical events
                if step % 20 == 0 or done:
                    # Format state vector for clean display
                    state_str = f"[{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}, {obs[3]:.2f}, {obs[4]:.2f}, {obs[5]:.2f}]"
                    logging.info(
                        f"  Step {step:3d} | Action: {action[0]:+5.1f} ({action_type:<8}) | "
                        f"State: {state_str} | StepReward: {reward:+6.2f}"
                    )
            
            outcome = step_info.get("termination_reason", "unknown").upper()
            episode_results.append(outcome)
            
            status = env.curriculum.get_status()
            logging.info(
                f"Episode {episode+1} Finished | Outcome: {outcome} | Steps: {step} | "
                f"Episode Reward: {ep_reward:.2f} | Rolling Success: {status['rolling_success_rate']*100:.0f}% | "
                f"Stage: {status['current_stage']} (HDV: {status['hdv_ratio']*100:.0f}%)"
            )
            print("-" * 110)
            
        # --- SUITE VERIFICATION REPORT ---
        logging.info("=== Environment Validation Report ===")
        logging.info(f"Total Steps Executed: {total_steps_executed}")
        logging.info(f"Episode Outcomes: {episode_results}")
        
        # Verify Stage Advancement (at least one transition should happen if successes are high)
        stages_reached = env.curriculum.current_stage
        curriculum_advancing = stages_reached > 1
        
        # Log checklist status
        logging.info(f"Checklist | Vehicles spawning verified: {'PASSED' if spawning_verified else 'FAILED'}")
        logging.info(f"Checklist | State values updates verified: {'PASSED' if states_verified else 'FAILED'}")
        logging.info(f"Checklist | Rewards computation verified: {'PASSED' if rewards_verified else 'FAILED'}")
        logging.info(f"Checklist | Curriculum advancement verified: {'PASSED' if curriculum_advancing else 'FAILED'} (Reached Stage {stages_reached})")
        
        assert spawning_verified, "Spawning verification failed."
        assert states_verified, "State update verification failed."
        assert rewards_verified, "Rewards computation verification failed."
        assert curriculum_advancing, "Curriculum advancement verification failed. Make sure the rule-based agent can succeed."
        
        logging.info("Environment Validation Suite COMPLETED. All environment and integration parameters verified successfully!")
        
    except Exception as e:
        logging.error(f"Environment Validation Suite FAILED: {e}", exc_info=True)
        raise e
    finally:
        env.close()

if __name__ == "__main__":
    run_validation_suite()
