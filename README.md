# Curriculum Reinforcement Learning for Robust Autonomous Vehicle Entry in Mixed-Autonomy Roundabouts

This repository contains the source code, simulation configurations, and training pipelines for the research project investigating **Curriculum Reinforcement Learning (CRL)** to achieve robust autonomous vehicle (AV) insertion in mixed-autonomy roundabout intersections. The project uses SUMO for traffic simulation, Gymnasium for structuring the RL environment, and Stable-Baselines3 (PPO) for policy training.

---

## 1. Project Directory Structure

Below is the directory tree and the purpose of each folder:

```text
RoundaboutRL/
│
├── sumo_network/     # SUMO configuration files (.sumocfg, .net.xml, .rou.xml, etc.)
├── env/              # Gymnasium environment wrappers and custom observation/reward logic
├── curriculum/       # Curriculum progression, stage schedules, and transition heuristics
├── training/         # PPO model instantiation, training loops, and callbacks
├── evaluation/       # Performance evaluation, inference scripts, and robustness testing
├── results/          # Artifacts generated during training and evaluation
│   ├── logs/         # TensorBoard directories and training CSV reports
│   ├── figures/      # Training curve plots, evaluation graphs, and visual analysis
│   └── models/       # Saved policy weights, best checkpoints, and final models
├── configs/          # Hyperparameter definitions (YAML/JSON) for env and models
├── paper/            # Academic paper drafts, LaTeX templates, bib files, and assets
├── notebooks/        # Jupyter notebooks for data analysis and quick experimentation
├── tests/            # Unit tests for rewards, observation shapes, and simulator interfacing
├── .gitignore        # Version control exclude lists (configured for research runs)
├── requirements.txt  # Pip dependencies specifier
├── environment.yml   # Conda environment definition (alternative setup)
└── README.md         # Project documentation (this file)
```

---

## 2. Directory Purpose Details

*   **`sumo_network/`**: Stores network files generated via SUMO (e.g., `netedit` output files), route files defining human/AV flow rates, and the main `.sumocfg` configuration.
*   **`env/`**: Contains the core Gymnasium environment. This bridges PySim/SUMO via TraCI. It specifies:
    *   *Action Space*: Acceleration, lane changes, or target speed adjustments.
    *   *Observation Space*: State vectors of nearby vehicles, entry speed, gap size.
    *   *Reward Functions*: Collision penalties, progress rewards, safety margins (TTC/headway).
*   **`curriculum/`**: Houses the curriculum learning schedules. Rather than training in heavy traffic immediately, the curriculum scales up parameters (e.g., higher traffic density, lower gap sizes, higher human driver aggressiveness) across multiple training stages.
*   **`training/`**: Handles RL policy training. Uses PyTorch-based PPO from Stable-Baselines3. Manages hyperparameter optimization, custom training callbacks (evaluations, checkpointing), and saving models.
*   **`evaluation/`**: Runs inference using trained checkpoints. Compares model performance across baseline controllers (e.g., SUMO's default IDM/Krauss model) and tests generalizability under out-of-distribution traffic rates.
*   **`results/`**: Outputs directory. Segmented into `logs/` (tensorboard files), `figures/` (published-ready plots), and `models/` (training checkpoints).
*   **`configs/`**: Keeps hyperparameters external to python files. Allows configuring routes, reward coefficients, network parameters, and SB3 hyperparameters (`learning_rate`, `batch_size`, `gamma`, etc.) in single YAML/JSON configurations.
*   **`paper/`**: Dedicated folder for LaTeX documents and drafting the associated research paper.
*   **`notebooks/`**: Exploratory data analysis (EDA), plotting training dynamics, and reviewing evaluation logs.
*   **`tests/`**: Unit test suites validating step rules, collision handlers, observation normalization, and ensuring environment regression is prevented.

---

## 3. Installation Instructions

Follow these steps to set up the software stack on your system (specifically configured for Windows systems).

### A. SUMO (Simulation of Urban MObility)
1. Download the Windows Installer (`sumo-win64-<version>.msi`) from the [SUMO Downloads Page](https://sumo.dlr.de/docs/Downloads.html).
2. Install the package to a standard path (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).
3. Set up the **`SUMO_HOME`** environment variable:
   * **System Properties** > **Environment Variables**.
   * Add a new System Variable:
     * **Variable Name**: `SUMO_HOME`
     * **Variable Value**: `C:\Program Files (x86)\Eclipse\Sumo` (adjust according to your chosen installation path).
4. Add the SUMO binary folder to your system `Path`:
   * Find the `Path` variable under System Variables and click **Edit**.
   * Add a new entry: `%SUMO_HOME%\bin` (or `C:\Program Files (x86)\Eclipse\Sumo\bin`).
5. Open a terminal (CMD or PowerShell) and verify the installation:
   ```bash
   sumo
   sumo-gui
   ```

### B. Python Virtual Environment Setup

Choose either **Option 1 (venv)** or **Option 2 (Conda)**.

#### Option 1: Using `venv` (Native Python)
Run the following commands in your terminal (ensure you are at the workspace root directory `Roundabout_RL`):

1. **Create the environment:**
   ```powershell
   python -m venv venv
   ```
2. **Activate the environment:**
   * On **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * On **Command Prompt (CMD)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
3. **Upgrade pip and install dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### Option 2: Using `Conda` (Miniconda/Anaconda)
If you prefer Conda, build the environment using the provided `environment.yml` file:

1. **Create the environment:**
   ```powershell
   conda env create -f environment.yml
   ```
2. **Activate the environment:**
   ```powershell
   conda activate roundabout_rl
   ```

---

### C. Deep Learning & RL Framework Configuration

#### PyTorch
The standard `requirements.txt` installs a CPU-compatible PyTorch build suitable for low-dimensional RL observation spaces. If you plan to train using a CUDA-enabled GPU (recommended if training with convolutional observations or using large parallel networks):

Go to the [PyTorch Get Started Page](https://pytorch.org/get-started/locally/) and run the corresponding command for your CUDA version:
```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### TraCI (Traffic Control Interface)
TraCI connects Python scripts to the running SUMO simulator. The library is installed via:
```bash
pip install traci sumolib
```
Python will automatically resolve the connection to SUMO using the `SUMO_HOME` environment variable configured in Step A.

#### Gymnasium & Stable-Baselines3
Gymnasium provides the standard environment API, and Stable-Baselines3 provides the PPO implementation:
```bash
pip install gymnasium
pip install stable-baselines3[extra]
```
*(The `[extra]` tag includes TensorBoard support, OpenCV for image processing, and additional wrapper elements).*

---

## 4. Quick Verification
To verify the installation of dependencies within your active environment, run:
```powershell
python -c "import torch; import gymnasium; import stable_baselines3; import traci; print('All core libraries imported successfully!')"
```
This should output:
`All core libraries imported successfully!`
