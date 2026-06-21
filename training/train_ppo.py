import os
import sys
import logging
from collections import deque
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [Trainer] - %(levelname)s - %(message)s")

class RoundaboutMetricsCallback(BaseCallback):
    """
    Custom Stable-Baselines3 callback to record custom episode statistics:
    - Success rate
    - Collision rate
    - Timeout rate
    - Curriculum stage progression
    Writes directly to TensorBoard and prints updates to the console.
    """
    def __init__(self, agent_name, verbose=0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        
        # Deques to track rolling averages over the last 100 episodes
        self.success_history = deque(maxlen=100)
        self.collision_history = deque(maxlen=100)
        self.timeout_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.length_history = deque(maxlen=100)
        
    def _on_step(self) -> bool:
        # Check if an episode ended in any of the vector environments
        for info in self.locals.get("infos", []):
            if "episode" in info:  # Key added by Monitor wrapper
                self.episode_count += 1
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                
                # Fetch custom environment outcomes passed through step() info dict
                success = info.get("success", False)
                collision = info.get("collision", False)
                timeout = info.get("timeout", False)
                stage = info.get("curriculum_stage", -1)
                hdv_ratio = info.get("hdv_ratio", 0.5)
                
                if success:
                    self.success_count += 1
                if collision:
                    self.collision_count += 1
                if timeout:
                    self.timeout_count += 1
                    
                self.success_history.append(success)
                self.collision_history.append(collision)
                self.timeout_history.append(timeout)
                self.reward_history.append(reward)
                self.length_history.append(length)
                
                success_rate = self.success_count / self.episode_count
                collision_rate = self.collision_count / self.episode_count
                timeout_rate = self.timeout_count / self.episode_count
                
                avg_reward = np.mean(self.reward_history)
                avg_length = np.mean(self.length_history)
                
                # Log metrics to TensorBoard logger
                self.logger.record("episode/reward", reward)
                self.logger.record("episode/success_rate", success_rate)
                self.logger.record("episode/collision_rate", collision_rate)
                self.logger.record("episode/timeout_rate", timeout_rate)
                self.logger.record("episode/avg_reward", avg_reward)
                self.logger.record("episode/avg_length", avg_length)
                
                # Progress reward is requested to be plotted in TB
                progress_reward = info.get("episode_progress_reward", 0.0)
                self.logger.record("episode/progress_reward", progress_reward)
                
                # Log reward components to TensorBoard
                self.logger.record("reward_components/progress_reward", progress_reward)
                self.logger.record("reward_components/jerk_penalty", info.get("episode_jerk_penalty", 0.0))
                self.logger.record("reward_components/timeout_penalty", info.get("episode_timeout_penalty", 0.0))
                self.logger.record("reward_components/success_reward", info.get("episode_success_reward", 0.0))
                self.logger.record("reward_components/collision_penalty", info.get("episode_collision_penalty", 0.0))
                self.logger.record("reward_components/merge_reward", info.get("merge_reward", 0.0))
                self.logger.record("reward_components/safe_merge_bonus", info.get("safe_merge_bonus", 0.0))
                self.logger.record("reward_components/unsafe_gap_penalty", info.get("unsafe_gap_penalty", 0.0))
                self.logger.record("reward_components/entry_delay_penalty", info.get("entry_delay_penalty", 0.0))
                self.logger.record("reward_components/waiting_penalty", info.get("waiting_penalty", 0.0))
                self.logger.record("reward_components/successful_exit_bonus", info.get("successful_exit_bonus", 0.0))
                self.logger.record("reward_components/efficient_merge_bonus", info.get("efficient_merge_bonus", 0.0))
                
                # Log TTC & Timing statistics
                self.logger.record("episode/avg_ttc", info.get("avg_ttc", 999.0))
                self.logger.record("episode/min_ttc", info.get("min_ttc", 999.0))
                self.logger.record("episode/time_to_merge", info.get("time_to_merge", 0.0))

                if stage != -1:
                    self.logger.record("episode/curriculum_stage", stage)
                    self.logger.record("episode/hdv_ratio", hdv_ratio)
                    
                # Console updates
                logging.info(
                    f"[{self.agent_name}] Episode {self.episode_count:3d} Finished | "
                    f"Reward: {reward:+6.2f} | Success Rate: {success_rate*100:5.1f}% | "
                    f"Collision Rate: {collision_rate*100:5.1f}% | "
                    f"Timeout Rate: {timeout_rate*100:5.1f}% | "
                    f"Avg Episode Reward: {avg_reward:+6.2f} | "
                    f"Avg Episode Length: {avg_length:.1f} steps | "
                    f"Stage: {stage} (HDV: {hdv_ratio*100:.0f}%)"
                )
                
                # Summary printout every 100 episodes
                if self.episode_count % 100 == 0:
                    rolling_success = (sum(self.success_history) / len(self.success_history)) * 100
                    rolling_collision = (sum(self.collision_history) / len(self.collision_history)) * 100
                    rolling_timeout = (sum(self.timeout_history) / len(self.timeout_history)) * 100
                    
                    print(f"\n=========================================")
                    print(f"   DIAGNOSTIC SUMMARY: LAST 100 EPISODES")
                    print(f"   Agent: {self.agent_name} | Total Episodes: {self.episode_count}")
                    print(f"=========================================")
                    print(f"   SUCCESS %:   {rolling_success:.2f}%")
                    print(f"   COLLISION %: {rolling_collision:.2f}%")
                    print(f"   TIMEOUT %:   {rolling_timeout:.2f}%")
                    print(f"   AVG REWARD:  {avg_reward:.2f}")
                    print(f"   AVG LENGTH:  {avg_length:.2f}")
                    print(f"=========================================\n", flush=True)
        return True

def train_agent(agent_type, total_timesteps=10000):
    """
    Configures and runs PPO training for the requested agent.
    
    Args:
        agent_type (str): 'A' for fixed 50% HDV, 'B' for curriculum learning.
        total_timesteps (int): Number of steps to train.
    """
    logging.info(f"=== Configuring Agent {agent_type} Training ===")
    
    # Establish result directories
    tb_log_dir = os.path.join("results", "logs", "tb")
    model_save_dir = os.path.join("results", "models")
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 1. Instantiate the environments
    if agent_type == "A":
        # Agent A: Fixed environment with 50% HDV ratio
        train_env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
        agent_name = "Agent_A_Fixed_50"
    else:
        # Agent B: Curriculum environment, starting at Stage 1 (0% HDV)
        train_env = RoundaboutEnv(fixed_hdv_ratio=None, gui=False, target_success_rate=0.80, curriculum_window=5)
        agent_name = "Agent_B_Curriculum"
        
    # Create evaluation environment to check both policies under a standard 50% HDV mix
    eval_env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
    
    # Wrap in Monitor wrapper (crucial to register episode events for callbacks)
    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)
    
    # Wrap in DummyVecEnv (required by Stable-Baselines3)
    train_vec_env = DummyVecEnv([lambda: train_env])
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # 2. Configure Callbacks
    metrics_callback = RoundaboutMetricsCallback(agent_name=agent_name)
    
    # EvalCallback evaluates the policy on the separate eval_env and saves the best model
    eval_callback = EvalCallback(
        eval_env=eval_vec_env,
        best_model_save_path=os.path.join(model_save_dir, f"best_{agent_name.lower()}"),
        log_path=os.path.join("results", "logs", f"eval_{agent_name.lower()}"),
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    callbacks = CallbackList([metrics_callback, eval_callback])
    
    # 3. Instantiate PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=tb_log_dir
    )
    
    logging.info(f"Model initialized. Starting training of {agent_name} for {total_timesteps} steps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=agent_name
        )
        
        # Save final model
        final_model_path = os.path.join(model_save_dir, f"final_{agent_name.lower()}.zip")
        model.save(final_model_path)
        logging.info(f"Successfully saved final model for {agent_name} to: {final_model_path}")
        
    except Exception as e:
        logging.error(f"Training of {agent_name} failed: {e}", exc_info=True)
        raise e
    finally:
        # Clean up connections
        train_env.close()
        eval_env.close()
        logging.info(f"Environments closed for {agent_name}.\n")

if __name__ == "__main__":
    # In a full research setup, we would train for 100,000+ timesteps.
    # We use 21,000 timesteps here to trigger and test the 100-episode diagnostic summary logs.
    
    # Train Agent A (Fixed)
    train_agent(agent_type="A", total_timesteps=21000)
    
    # Train Agent B (Curriculum)
    train_agent(agent_type="B", total_timesteps=21000)
    
    print("\nAll training runs completed successfully!")
