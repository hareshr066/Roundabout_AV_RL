import os
import sys
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.roundabout_env import RoundaboutEnv

def evaluate_policy(env, model=None, policy_type="ppo", episodes=10):
    successes = 0
    collisions = 0
    timeouts = 0
    merge_times = []
    avg_ttcs = []
    total_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_ttcs = []
        steps = 0
        
        while not (done or truncated):
            if policy_type == "ppo" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            elif policy_type == "random":
                action = env.action_space.sample()
            elif policy_type == "idm":
                action = np.array([0.0]) # Let SUMO control with default car following
            elif policy_type == "rule_based":
                # observation assumption based on prompt: 
                # obs[0]=ego_speed, obs[2]=nearest_dist, obs[4]=gap_size
                ego_speed = obs[0] if len(obs) > 0 else 0
                nearest_dist = obs[2] if len(obs) > 2 else 50
                gap_size = obs[4] if len(obs) > 4 else 50
                
                if gap_size > 20.0 and nearest_dist > 15.0:
                    action = np.array([1.5])
                elif nearest_dist < 5.0:
                    action = np.array([-3.0])
                else:
                    action = np.array([0.5])
            else:
                action = np.array([0.0])
                
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            try:
                t = env._get_ttc_after_merge()
                if t < 100.0:
                    ep_ttcs.append(t)
            except Exception:
                pass
            
        total_rewards.append(ep_reward)
        avg_ttcs.append(float(np.mean(ep_ttcs)) if ep_ttcs else 10.0)
        
        if info.get('success', False):
            successes += 1
            dt = getattr(env, 'dt', 0.1)
            merge_times.append(steps * dt)
        elif info.get('collision', False):
            collisions += 1
        else:
            timeouts += 1
            
    return {
        'success_rate': successes / episodes,
        'collision_rate': collisions / episodes,
        'timeout_rate': timeouts / episodes,
        'avg_merge_time': np.mean(merge_times) if merge_times else 0.0,
        'avg_ttc': np.mean(avg_ttcs),
        'avg_reward': np.mean(total_rewards)
    }

def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per baseline")
    parser.add_argument("--test-mode", action="store_true", help="Run 2 episodes per baseline")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "results", "models", "final_best_agent.zip")
    
    episodes = 2 if args.test_mode else args.episodes
    
    print("Setting up environment...")
    env = RoundaboutEnv(
        gui=False,
        use_spatial_curriculum=False,
        fixed_spawn_distance=80.0,
        fixed_hdv_ratio=0.50,
        max_steps=400,
        verbose=False
    )
    
    results = []
    
    # 1. PPO Best Model
    try:
        model = PPO.load(model_path)
        print(f"Evaluating PPO Best Model over {episodes} episodes...")
        ppo_res = evaluate_policy(env, model=model, policy_type="ppo", episodes=episodes)
        ppo_res['policy'] = "PPO (Ours)"
        results.append(ppo_res)
    except Exception as e:
        print(f"Failed to load PPO model, skipping: {e}")
        
    # 2. SUMO Default (IDM)
    print(f"Evaluating SUMO Default (IDM) over {episodes} episodes...")
    idm_res = evaluate_policy(env, policy_type="idm", episodes=episodes)
    idm_res['policy'] = "SUMO Default (IDM)"
    results.append(idm_res)
    
    # 3. Random Policy
    print(f"Evaluating Random Policy over {episodes} episodes...")
    rand_res = evaluate_policy(env, policy_type="random", episodes=episodes)
    rand_res['policy'] = "Random"
    results.append(rand_res)
    
    # 4. Rule-Based
    print(f"Evaluating Rule-Based Policy over {episodes} episodes...")
    rule_res = evaluate_policy(env, policy_type="rule_based", episodes=episodes)
    rule_res['policy'] = "Rule-Based Gap Acceptance"
    results.append(rule_res)
    
    # Save & Print Results
    df = pd.DataFrame(results)
    cols = ['policy', 'success_rate', 'collision_rate', 'timeout_rate', 'avg_merge_time', 'avg_ttc', 'avg_reward']
    df = df[cols]
    
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "baseline_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    # Print formatted table
    print(df.to_string(index=False))
    print("="*80)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
