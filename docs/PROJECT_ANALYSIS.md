# 🚗 Roundabout RL — Complete Project Analysis
# Last Updated: 2026-09-14

## What Is This Project?

This project teaches a **self-driving car** how to **safely enter a roundabout** using **Reinforcement Learning (RL)**. The car learns by trial-and-error in a SUMO traffic simulator, getting rewards for good behavior (safe merging) and penalties for bad behavior (crashes, waiting too long).

### The Core Problem
Merging into a roundabout is hard for self-driving cars because:
- Human drivers are unpredictable
- The car must find a safe gap in circulating traffic
- It needs to time its entry perfectly — not too aggressive (crash), not too cautious (block traffic)

### The Solution: Curriculum Learning
Instead of throwing the car into the hardest scenario immediately, we teach it step-by-step:
1. **HDV Curriculum**: Start with easy traffic (all cooperative AVs) → gradually increase human drivers (0%→25%→50%→75%→100%)
2. **Spatial Curriculum**: Start the car close to the roundabout (15m) → move it farther back (30m→50m→80m)

---

## Architecture Overview

```
RoundaboutRL/
├── sumo_network/     → SUMO road network (roundabout with 4 arms)
├── env/              → Gymnasium RL environment (6D observations, continuous action)
├── curriculum/       → HDV ratio + spatial spawn distance curriculum managers
├── training/         → PPO training scripts (Stable-Baselines3)
├── evaluation/       → Ablation study, penetration study, safety analysis, baselines
├── results/          → Models, logs, figures, CSV data
├── configs/          → YAML config files for reproducibility
├── web/              → Interactive web dashboard (real-time simulation viewer)
├── paper/            → IEEE LaTeX research paper skeleton
├── notebooks/        → Jupyter analysis notebooks
├── tests/            → Unit tests
├── docs/             → This file + implementation plan
└── web_server.py     → Tornado WebSocket server for dashboard
```

---

## Key Technical Details

### RL Environment
- **Action Space**: Continuous acceleration [-4.0, +2.0] m/s²
- **Observation Space** (6D):
  1. Ego Speed (m/s)
  2. Distance to Entry (m)
  3. Nearest Circulating Vehicle Distance (m)
  4. Nearest Circulating Vehicle Speed (m/s)
  5. Gap Size (m)
  6. HDV Ratio (0-1)

### Reward Structure
- +100 for successful exit
- -200 for collision
- -300 for timeout
- Progress reward for moving toward entry
- Jerk penalty for uncomfortable driving

### Algorithm
- PPO (Proximal Policy Optimization) via Stable-Baselines3
- MLP Policy with [256, 256] hidden layers
- Learning rate: 3e-4, Gamma: 0.99

### Current Results (Ablation Study)
| Variant | Success Rate | Collision Rate | Timeout Rate |
|---------|-------------|----------------|-------------|
| Baseline PPO | 0% | 0% | 100% |
| + Context-Aware | 0% | 0% | 100% |
| + Spatial Curriculum | 0% | 0% | 100% |
| + Reward Shaping | **55%** | 45% | 0% |
| Full Method | 0% | 0% | 100% |

> **Note**: These results are from undertrained models (12K-49K steps). With 500K+ steps, results should improve dramatically.

### Penetration Study (Spatial Curriculum 30K model)
| HDV Ratio | Success Rate | Avg TTC |
|-----------|-------------|---------|
| 0% | 100% | 4.89s |
| 25% | 100% | 4.18s |
| 50% | 100% | 3.77s |
| 75% | 100% | 3.63s |
| 100% | 100% | 4.17s |

---

## Trained Models Available

| Model | Description |
|-------|-------------|
| `agent_spatial_curriculum_30k.zip` | Primary model (spatial curriculum, 30K steps) |
| `final_agent_a_fixed_50.zip` | Fixed 50% HDV baseline |
| `final_agent_b_curriculum.zip` | HDV curriculum agent |
| `ablation_v1_baseline.zip` | Ablation: baseline PPO |
| `ablation_v2_context.zip` | Ablation: + context-aware |
| `ablation_v3_spatial.zip` | Ablation: + spatial curriculum |
| `ablation_v4_shaping.zip` | Ablation: + reward shaping |
| `ablation_v5_full.zip` | Ablation: full method |

---

## How to Run

```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Train final agent
python training/train_final.py --timesteps 500000

# Run evaluations
python evaluation/run_ablation_study.py
python evaluation/run_penetration_study.py
python evaluation/run_safety_analysis.py
python evaluation/run_baseline_comparison.py

# Generate figures
python evaluation/generate_figures.py

# Launch web dashboard
python web_server.py
# Open http://localhost:8888

# Run tests
pytest tests/ -v
```

---

## Dependencies
- Python 3.10+
- SUMO (with SUMO_HOME environment variable)
- gymnasium==0.29.1
- stable-baselines3==2.3.2
- traci==1.20.0
- torch>=2.1.0
- numpy, pandas, matplotlib
