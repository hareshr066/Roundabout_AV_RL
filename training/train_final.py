import os
import sys
import yaml
import argparse
import time
import json
import socket
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from env.roundabout_env import RoundaboutEnv
from training.train_spatial_curriculum import SpatialMetricsCallback

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Final PPO training for Roundabout RL")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (2048 steps)")
    parser.add_argument("--gpu", action="store_true", help="Force CUDA usage")
    parser.add_argument("--name", type=str, default="final_best_agent", help="Model name")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ppo_config_path = os.path.join(project_root, "configs", "ppo_config.yaml")
    env_config_path = os.path.join(project_root, "configs", "env_config.yaml")

    ppo_config = load_config(ppo_config_path) if os.path.exists(ppo_config_path) else {}
    env_config = load_config(env_config_path) if os.path.exists(env_config_path) else {}

    total_timesteps = 2048 if args.test_mode else args.timesteps
    device = "cuda" if args.gpu else "auto"

    results_dir = os.path.join(project_root, "results", "models")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training: {args.name}")
    print(f"Total timesteps: {total_timesteps}")

    env = RoundaboutEnv(
        use_context_aware=True, 
        use_gap_reward=True, 
        use_spatial_curriculum=True, 
        spatial_window_size=30, 
        spatial_target_success_rate=0.80, 
        fixed_hdv_ratio=0.50, 
        max_steps=800
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Try to extract hyperparameters from config or use defaults
    hyperparams = ppo_config.get("ppo", {})
    if not hyperparams:
        hyperparams = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }

    # Remove keys that are passed explicitly or are not valid PPO constructor args
    for key in ["policy", "verbose", "device", "tensorboard_log", "policy_kwargs", "activation_fn"]:
        hyperparams.pop(key, None)

    model = PPO("MlpPolicy", vec_env, device=device, verbose=1, tensorboard_log=os.path.join(project_root, "results", "logs", "tb"), **hyperparams)

    # Callbacks
    eval_env = RoundaboutEnv(
        use_context_aware=True, 
        use_gap_reward=True, 
        use_spatial_curriculum=False, # Disable curriculum for evaluation
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50, 
        max_steps=800
    )
    eval_env = Monitor(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(eval_vec_env, best_model_save_path=results_dir,
                                 log_path=results_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=checkpoints_dir,
                                             name_prefix=args.name)

    spatial_callback = SpatialMetricsCallback(agent_name=args.name)

    callback_list = CallbackList([eval_callback, checkpoint_callback, spatial_callback])

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    finally:
        end_time = time.time()
        final_model_path = os.path.join(results_dir, f"{args.name}.zip")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        metadata = {
            "name": args.name,
            "timesteps": total_timesteps,
            "ppo_config": ppo_config,
            "env_config": env_config,
            "start_time": start_time,
            "end_time": end_time,
            "training_duration": end_time - start_time,
            "hostname": socket.gethostname(),
            "device": device
        }

        metadata_path = os.path.join(results_dir, f"{args.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training finished.")
        print(f"Summary: Model '{args.name}' trained for {total_timesteps} steps in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
