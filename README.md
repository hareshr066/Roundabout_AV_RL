<div align="center">

# 🚗 Roundabout RL: Curriculum Learning for Mixed-Autonomy Intersections

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-orange.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/SB3-2.3.2-blueviolet.svg)](https://stable-baselines3.readthedocs.io/)
[![SUMO](https://img.shields.io/badge/SUMO-1.20.0-green.svg)](https://eclipse.dev/sumo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A reinforcement learning approach to navigating mixed-autonomy roundabout intersections using Proximal Policy Optimization (PPO) and Dual Curriculum Learning.*

[**Read the Full Paper PDF**](./docs/Roundabout_RL_Comprehensive_Project_Report.pdf) | [**View Implementation Plan**](./docs/IMPLEMENTATION_PLAN.md)

</div>

---

## 🎥 Simulation Demo

<div align="center">
  <!-- TODO: Replace the placeholder below with the actual demo GIF or video -->
  <img src="media/demo_placeholder.gif" alt="Roundabout Simulation Demo" width="700"/>
  <br>
  <i>Watch the trained PPO agent smoothly merge into a roundabout with 50% Human-Driven Vehicles (HDVs), maintaining optimal safety buffers and minimizing jerk.</i>
</div>

> **Tip:** You can generate a video of your simulation running by using the built-in screen recorder in SUMO-GUI or capturing the web dashboard. Place it in a `media/` folder and name it `demo.gif` or `demo.mp4`.

---

## 🌟 Project Highlights

- **Dual-Curriculum Training**: Utilizes both a *Spatial Curriculum* (gradually increasing spawn distance to the merge point) and an *HDV Penetration Curriculum* to ensure robust learning.
- **100% Success Rate**: Achieved zero collisions and 100% success rate across all evaluated Human-Driven Vehicle (HDV) penetration ratios (0% to 100%).
- **Interactive Web Dashboard**: Built-in Tornado web server featuring a live dashboard with WebSockets for real-time telemetry streaming, metrics tracking, and 2D canvas rendering.
- **Comprehensive Evaluation**: Rigorous evaluation suite including Ablation Studies, Safety Analysis, Baseline Comparisons (IDM vs. Rule-based vs. PPO), and Penetration Studies.

---

## 📊 Key Results

Our PPO agent demonstrated state-of-the-art performance when compared to traditional car-following models (IDM) and rule-based controllers in a mixed-autonomy environment:

| Controller | Success Rate | Collision Rate | Mean Merge Time (s) | Min TTC (s) |
|------------|--------------|----------------|---------------------|-------------|
| **PPO (Ours)** | **100%** | **0%** | **9.75** | **4.82** |
| Baseline IDM | 0% (Timeout) | 0% | N/A | >10.0 |
| Rule-Based | 0% | 100% | N/A | 0.0 |

*Detailed experimental data can be found in the [Results Directory](./results/).*

---

## 🏗️ Architecture & Full Plan

The project is structured around a modular architecture to facilitate training, evaluation, and deployment:

### Implementation Roadmap
1. ✅ **Environment Design**: Custom Gymnasium wrapper (`roundabout_env.py`) interfacing with SUMO via TraCI. 6D Continuous State Space & 1D Continuous Action Space (Acceleration).
2. ✅ **Reward Shaping**: Multi-objective reward function incorporating progress, collision penalties, jerk penalties for passenger comfort, and gap-based safety rewards.
3. ✅ **Curriculum Learning**: Implemented multi-stage progression to tackle the sparse reward problem inherent in intersection merging.
4. ✅ **Training & Hyperparameters**: Scaled PPO with specific configurations for context-aware state processing.
5. ✅ **Live Dashboard**: Web server (`web_server.py`) and UI (`web/`) for monitoring training and inference in real-time.
6. ✅ **Evaluation & Publishing**: Scripts generating LaTeX tables, 300 DPI figures, and automated PDF comprehensive reports.

### Directory Structure

```text
RoundaboutRL/
├── configs/          # YAML configurations for Env, PPO, and Experiments
├── docs/             # Implementation plans and generated PDF reports
├── env/              # Gymnasium environment (roundabout_env.py)
├── evaluation/       # Robustness tests, baselines, and safety analysis
├── paper/            # IEEE LaTeX paper source and figures
├── results/          # Models (e.g., final_best_agent.zip) and CSV logs
├── sumo_network/     # SUMO .net.xml and routing configuration
├── training/         # PPO training loop and custom callbacks
├── web/              # Live dashboard HTML/JS frontend
└── web_server.py     # Tornado server and WebSocket telemetry
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.13 and SUMO installed. Add SUMO to your system path:
* Download from [Eclipse SUMO](https://eclipse.dev/sumo/)
* Ensure `SUMO_HOME` environment variable is set (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).

### 2. Installation
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
Launch the web server to interact with the environment:
```bash
python web_server.py
```
*Open `http://localhost:8080` in your browser.*

### 4. Training a New Agent
To train from scratch using the full curriculum:
```bash
python training/train_final.py
```

### 5. Running Evaluations
Evaluate safety metrics or run the ablation study:
```bash
python evaluation/run_safety_analysis.py
python evaluation/run_ablation_study.py --test-mode
```

---
<div align="center">
  <i>Developed for Advanced Reinforcement Learning Research in Autonomous Driving</i>
</div>
