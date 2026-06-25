import os
import sys
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.roundabout_env import RoundaboutEnv

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [AblationTrainer] - %(levelname)s - %(message)s")

def train_ablation_variants():
    model_dir = os.path.join("results", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    configs = {
        "v1_baseline": {
            "name": "1. Baseline PPO",
            "total_timesteps": 12288,
            "env_kwargs": {
                "use_context_aware": False,
                "use_spatial_curriculum": False,
                "fixed_spawn_distance": 80.0,
                "use_gap_reward": False,
                "fixed_hdv_ratio": 0.50,
                "verbose": False
            }
        },
        "v2_context": {
            "name": "2. + Context-Aware Observations",
            "total_timesteps": 12288,
            "env_kwargs": {
                "use_context_aware": True,
                "use_spatial_curriculum": False,
                "fixed_spawn_distance": 80.0,
                "use_gap_reward": False,
                "fixed_hdv_ratio": 0.50,
                "verbose": False
            }
        },
        "v3_spatial": {
            "name": "3. + Spatial Curriculum",
            "total_timesteps": 16384,
            "env_kwargs": {
                "use_context_aware": True,
                "use_spatial_curriculum": True,
                "spatial_window_size": 5,
                "spatial_target_success_rate": 0.70,
                "use_gap_reward": False,
                "fixed_hdv_ratio": 0.50,
                "verbose": False
            }
        },
        "v4_shaping": {
            "name": "4. + Gap-Acceptance Reward Shaping",
            "total_timesteps": 49152,
            "env_kwargs": {
                "use_context_aware": True,
                "use_spatial_curriculum": True,
                "spatial_window_size": 5,
                "spatial_target_success_rate": 0.70,
                "use_gap_reward": True,
                "fixed_hdv_ratio": 0.50,
                "verbose": False
            }
        },
        "v5_full": {
            "name": "5. Full Method",
            "total_timesteps": 49152,
            "env_kwargs": {
                "use_context_aware": True,
                "use_spatial_curriculum": True,
                "spatial_window_size": 5,
                "spatial_target_success_rate": 0.70,
                "use_gap_reward": True,
                "fixed_hdv_ratio": None,  # Enables HDV Curriculum
                "curriculum_window": 5,
                "target_success_rate": 0.70,
                "verbose": False
            }
        }
    }
    
    for key, cfg in configs.items():
        model_path = os.path.join(model_dir, f"ablation_{key}.zip")
        logging.info("=" * 80)
        logging.info(f"STARTING TRAINING FOR VARIANT: {cfg['name']}")
        logging.info("=" * 80)
        
        # We overwrite the ablation models to ensure we train them with the new step budget
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logging.info(f"Removed old model at {model_path} to re-train.")
            except Exception as e:
                logging.warning(f"Could not remove old model {model_path}: {e}")
            
        # Create env
        env = RoundaboutEnv(gui=False, max_steps=200, **cfg["env_kwargs"])
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize PPO model
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0
        )
        
        total_steps = cfg["total_timesteps"]
        logging.info(f"Training for {total_steps} steps...")
        model.learn(total_timesteps=total_steps)
        
        # Save model
        model.save(model_path)
        env.close()
        logging.info(f"Saved trained model to {model_path}")
        
    logging.info("Ablation study training completed successfully.")

if __name__ == "__main__":
    train_ablation_variants()

