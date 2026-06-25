import os
import sys
import shutil
import time
import traceback
import numpy as np

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

results = {
    "GPU": "FAIL",
    "CUDA": "FAIL",
    "SUMO": "FAIL",
    "TraCI": "FAIL",
    "Environment": "FAIL",
    "Curriculum": "FAIL",
    "PPO Training": "FAIL",
    "Model Save/Load": "FAIL",
    "TensorBoard": "FAIL"
}

details = {}

def run_validation():
    print("=" * 80)
    print("                  RESEARCH WORKSTATION VALIDATION SUITE")
    print("=" * 80)

    # --- CHECK 1: GPU & CUDA ---
    print("\n[CHECK 1] GPU & CUDA Verification...")
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        details["PyTorch Version"] = pytorch_version
        details["CUDA Available"] = cuda_available
        
        print(f"  PyTorch Version: {pytorch_version}")
        print(f"  CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            details["GPU Name"] = gpu_name
            details["CUDA Version"] = cuda_version
            print(f"  GPU Name: {gpu_name}")
            print(f"  CUDA Version: {cuda_version}")
            
            # Stable-Baselines3 Device Check
            from stable_baselines3 import PPO
            import gymnasium as gym
            dummy_env = gym.make('CartPole-v1')
            model = PPO('MlpPolicy', dummy_env, verbose=0, device='cuda')
            sb3_uses_cuda = str(model.device) == 'cuda'
            details["SB3 PPO Uses CUDA"] = sb3_uses_cuda
            print(f"  Stable-Baselines3 PPO Uses CUDA: {sb3_uses_cuda}")
            
            # Small Tensor Operation on GPU
            x = torch.tensor([10.0, 20.0, 30.0], device='cuda')
            y = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            z = x * y
            z_sum = z.sum().item()
            details["GPU Tensor Op Sum"] = z_sum
            print(f"  GPU Tensor Op Result (Sum of [10, 40, 90]): {z_sum}")
            
            if sb3_uses_cuda and abs(z_sum - 140.0) < 1e-4:
                results["GPU"] = "PASS"
                results["CUDA"] = "PASS"
        else:
            print("  FAIL: CUDA is not available to PyTorch.")
    except Exception as e:
        print(f"  FAIL: Exception in GPU/CUDA Verification: {e}")
        details["GPU/CUDA Error"] = str(e)
        traceback.print_exc()

    # --- CHECK 2: SUMO & TRACI ---
    print("\n[CHECK 2] SUMO & TraCI Verification...")
    try:
        sumo_home = os.environ.get("SUMO_HOME")
        details["SUMO_HOME"] = sumo_home
        print(f"  SUMO_HOME: {sumo_home}")
        
        if sumo_home:
            sumo_exe = os.path.join(sumo_home, "bin", "sumo.exe")
            sumo_gui = os.path.join(sumo_home, "bin", "sumo-gui.exe")
            sumo_exe_exists = os.path.exists(sumo_exe)
            sumo_gui_exists = os.path.exists(sumo_gui)
            details["sumo_exe_exists"] = sumo_exe_exists
            details["sumo_gui_exists"] = sumo_gui_exists
            print(f"  sumo executable: {sumo_exe} (Exists: {sumo_exe_exists})")
            print(f"  sumo-gui executable: {sumo_gui} (Exists: {sumo_gui_exists})")
        else:
            sumo_exe_exists = False
            sumo_gui_exists = False
            print("  FAIL: SUMO_HOME is not set.")
            
        import traci
        from env.traci_connection import TraCIConnection
        details["TraCI Import"] = "Success"
        print("  TraCI Import: Success")
        
        # Load config and run a short simulation (10 steps)
        config_file = os.path.join("sumo_network", "roundabout.sumocfg")
        print(f"  Loading network config: {config_file}...")
        manager = TraCIConnection(
            config_path=config_file,
            gui=False,
            step_length=0.1,
            label="val_sumo_traci"
        )
        conn = manager.start()
        print("  SUMO simulation started successfully.")
        
        # Advance simulation
        for step in range(10):
            manager.step()
        print("  Advanced simulation 10 steps successfully.")
        manager.close()
        print("  TraCI Connection closed.")
        
        if sumo_home and sumo_exe_exists and sumo_gui_exists:
            results["SUMO"] = "PASS"
        if conn is not None:
            results["TraCI"] = "PASS"
    except Exception as e:
        print(f"  FAIL: Exception in SUMO/TraCI verification: {e}")
        details["SUMO/TraCI Error"] = str(e)
        traceback.print_exc()

    # --- CHECK 3: FILE STRUCTURE ---
    print("\n[CHECK 3] File Structure Verification...")
    required_dirs = ["env", "curriculum", "training", "evaluation", "results", "sumo_network"]
    required_files = [
        "env/roundabout_env.py",
        "env/traci_connection.py",
        "curriculum/curriculum_manager.py",
        "curriculum/spatial_curriculum_manager.py",
        "training/train_ppo.py",
        "training/train_spatial_curriculum.py",
        "evaluation/evaluate_ppo.py",
        "sumo_network/roundabout.sumocfg",
        "sumo_network/roundabout.net.xml",
        "sumo_network/routes.xml"
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    missing_files = [f for f in required_files if not os.path.isfile(f)]
    details["Missing Directories"] = missing_dirs
    details["Missing Files"] = missing_files
    
    print(f"  Required Directories: {required_dirs}")
    print(f"  Missing Directories: {missing_dirs}")
    print(f"  Required Files Count: {len(required_files)}")
    print(f"  Missing Files: {missing_files}")

    # --- CHECK 4: ENVIRONMENT & CURRICULUM ---
    print("\n[CHECK 4] Environment & Curriculum Suite Verification...")
    try:
        from env.roundabout_env import RoundaboutEnv
        
        # Initialize Curriculum env
        env = RoundaboutEnv(
            config_file="sumo_network/roundabout.sumocfg",
            gui=False,
            max_steps=50,
            fixed_hdv_ratio=None,  # triggers CurriculumManager
            target_success_rate=0.80,
            curriculum_window=5
        )
        
        # Verify Curriculum Initialization
        curr_inited = env.curriculum is not None and env.curriculum.current_stage == 1
        details["Curriculum Initialized"] = curr_inited
        print(f"  Curriculum manager initialized at Stage 1: {curr_inited}")
        
        # Reset and verify state extraction and vehicle spawning
        obs, info = env.reset()
        state_extracted = obs is not None and len(obs) == 6
        details["State Extracted"] = state_extracted
        print(f"  Initial observation extracted (shape: {obs.shape if obs is not None else None}): {state_extracted}")
        
        ego_spawned = "ego" in env.sim.conn.vehicle.getIDList()
        details["Ego Vehicle Spawned"] = ego_spawned
        print(f"  Ego vehicle exists in SUMO: {ego_spawned}")
        
        # Step and verify reward calculation
        action = np.array([0.5], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        reward_ok = reward is not None and isinstance(reward, (float, np.float32, np.float64))
        details["Reward Calculated"] = reward_ok
        print(f"  Reward returned from step: {reward} (Valid float: {reward_ok})")
        
        # Step until episode termination
        steps = 1
        done = terminated or truncated
        while not done and steps < 60:
            next_obs, reward, terminated, truncated, step_info = env.step(np.array([1.0], dtype=np.float32))
            done = terminated or truncated
            steps += 1
            
        episode_terminated = done
        details["Episode Terminated"] = episode_terminated
        details["Termination Reason"] = step_info.get("termination_reason")
        print(f"  Episode terminated/truncated after {steps} steps: {episode_terminated} (Reason: {step_info.get('termination_reason')})")
        
        # Verify curriculum updates are registered
        curr_status = env.curriculum.get_status()
        curriculum_updated = curr_status["total_episodes"] > 0
        details["Curriculum Updates Registered"] = curriculum_updated
        print(f"  Curriculum updated after episode: {curriculum_updated} (Episodes tracked: {curr_status['total_episodes']})")
        
        env.close()
        
        if curr_inited and curriculum_updated:
            results["Curriculum"] = "PASS"
        if state_extracted and ego_spawned and reward_ok and episode_terminated:
            results["Environment"] = "PASS"
            
    except Exception as e:
        print(f"  FAIL: Exception in Environment/Curriculum verification: {e}")
        details["Env/Curriculum Error"] = str(e)
        traceback.print_exc()

    # --- CHECK 5: PPO TRAINING & INFERENCE ---
    print("\n[CHECK 5] PPO Training (1000 steps) & Inference Verification...")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        tb_log_dir = os.path.join("results", "logs", "tb_val_test")
        model_save_path = os.path.join("results", "models", "val_test_model.zip")
        
        if os.path.exists(tb_log_dir):
            shutil.rmtree(tb_log_dir)
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
            
        train_env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
        train_env = Monitor(train_env)
        train_vec_env = DummyVecEnv([lambda: train_env])
        
        print("  Initializing PPO...")
        model = PPO(
            policy="MlpPolicy",
            env=train_vec_env,
            learning_rate=3e-4,
            n_steps=500,  # 500 steps per rollout
            batch_size=64,
            n_epochs=2,
            verbose=1,
            device="cuda",
            tensorboard_log=tb_log_dir
        )
        print("  PPO Initialized.")
        
        print("  Training model for 1000 steps...")
        model.learn(total_timesteps=1000, tb_log_name="PPO_Val")
        print(f"  Training finished. Total timesteps: {model.num_timesteps}")
        rollouts_executed = model.num_timesteps >= 1000
        details["Rollouts Executed"] = rollouts_executed
        
        train_env.close()
        
        print("  Saving model...")
        model.save(model_save_path)
        model_saved = os.path.exists(model_save_path)
        details["Model Saved"] = model_saved
        print(f"  Model saved to {model_save_path}: {model_saved}")
        
        print("  Reloading model...")
        reloaded_model = PPO.load(model_save_path)
        print("  Model reloaded successfully.")
        
        print("  Running evaluation episode (inference check)...")
        eval_env = RoundaboutEnv(fixed_hdv_ratio=0.50, gui=False)
        obs, info = eval_env.reset()
        done = False
        eval_steps = 0
        eval_reward = 0.0
        
        while not done and eval_steps < 200:
            action, _ = reloaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            eval_reward += reward
            eval_steps += 1
            
        eval_env.close()
        inference_works = eval_steps > 0
        details["Evaluation Steps"] = eval_steps
        details["Evaluation Reward"] = eval_reward
        details["Inference Works"] = inference_works
        print(f"  Evaluation episode finished in {eval_steps} steps with reward {eval_reward:.2f}. Inference works: {inference_works}")
        
        if rollouts_executed:
            results["PPO Training"] = "PASS"
        if model_saved and inference_works:
            results["Model Save/Load"] = "PASS"
            
    except Exception as e:
        print(f"  FAIL: Exception in PPO Training/Inference verification: {e}")
        details["PPO Error"] = str(e)
        traceback.print_exc()

    # --- CHECK 6: TENSORBOARD LOGS ---
    print("\n[CHECK 6] TensorBoard Verification...")
    try:
        tb_log_dir = os.path.join("results", "logs", "tb_val_test")
        tb_dir_exists = os.path.isdir(tb_log_dir)
        print(f"  TensorBoard log directory '{tb_log_dir}' exists: {tb_dir_exists}")
        
        event_files = []
        if tb_dir_exists:
            for root, dirs, files in os.walk(tb_log_dir):
                for file in files:
                    if "tfevents" in file:
                        event_files.append(os.path.join(root, file))
        
        event_files_found = len(event_files) > 0
        details["TensorBoard Event Files Found"] = event_files_found
        print(f"  TensorBoard Event Files Found: {event_files_found}")
        
        from tensorboard.backend.event_processing import event_accumulator
        tb_readable = False
        if event_files_found:
            ea = event_accumulator.EventAccumulator(event_files[0])
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            print(f"  TensorBoard can read logs. Sample scalar tags: {tags[:5]}")
            tb_readable = len(tags) > 0
            details["TB Readable Tags Count"] = len(tags)
            
        if tb_dir_exists and event_files_found and tb_readable:
            results["TensorBoard"] = "PASS"
            
    except Exception as e:
        print(f"  FAIL: Exception in TensorBoard verification: {e}")
        details["TensorBoard Error"] = str(e)
        traceback.print_exc()

    # --- REPORT & TABLES ---
    all_passed = all(status == "PASS" for status in results.values())
    
    print("\n" + "=" * 50)
    print("               SUMMARY VALIDATION TABLE")
    print("=" * 50)
    for comp, status in results.items():
        print(f"  {comp:<25} : {status}")
    print("=" * 50)
    
    if all_passed:
        print("\nRESEARCH WORKSTATION READY\n")
    else:
        print("\nWORKSTATION VALIDATION FAILED\n")
        
    return all_passed

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
