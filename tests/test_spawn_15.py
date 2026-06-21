import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.roundabout_env import RoundaboutEnv

env = RoundaboutEnv(
    fixed_hdv_ratio=0.50,
    use_spatial_curriculum=False,
    fixed_spawn_distance=15.0,
    gui=False
)

for ep in range(5):
    print(f"--- EPISODE {ep} ---")
    obs, info = env.reset()
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step += 1
        if step % 20 == 0 or done:
            print(f"Step {step:3d} | Speed: {obs[0]:.2f} | Dist: {obs[1]:.2f} | TermReason: {step_info.get('termination_reason')}")
env.close()
