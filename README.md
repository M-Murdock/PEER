# PEER:  Policy Exploration and Explainability for Robotic control

A toolkit for training and testing robot policies in MuJoCo environments, with an interactive shared autonomy demonstration system.

## Overview

This repository contains two main components:

1. **Black Box Training** - A flexible interface for training PPO agents on any MuJoCo environment
2. **Shared Autonomy Simulator** - An interactive demonstration of human-robot collaboration with multiple goal inference methods

## Quick Start

### Training a Robot Policy

Use `black_box.py` to train and evaluate agents on MuJoCo environments:

```python
from black_box import BlackBox
import numpy as np

# Initialize with custom starting position
black_box = BlackBox(
    start_pos=np.array([-3.14, -1]), 
    env_id="Pusher-v5"
)

# Train the agent
black_box.train(timesteps=100000)

# Run and visualize
black_box.run(render="human")
```

**Key Parameters:**
- `env_id`: Any MuJoCo environment (see [available environments](https://gymnasium.farama.org/environments/mujoco/))
- `start_pos`: Custom initial joint positions for evaluation
- `timesteps`: Total training steps (default: 1000)

### Shared Autonomy Simulator

Run `dot_simulator_SA.py` to begin the interactive SA system:

```bash
python dot_simulator_SA.py
```

**Controls:**
- Arrow keys: Move the dot
- Click checkboxes: Toggle probability/goal visualizations
- GUI selection: Choose inference and arbitration methods

## Project Structure

```
PEER/
├── black_box.py              # Main training/evaluation interface
├── dot_simulator_SA.py       # Shared autonomy simulator
├── franka_test.py           # Franka robot example with custom wrappers
├── read_npy.py              # Utility for inspecting Q-tables
│
├── trained_policies/        # Pre-trained Q-tables for SA system
│   ├── q_table_topleft.npy
│   ├── q_table_orbit.npy
│   └── ...
│
├── training/                # Policy training utilities
│   └── policy_drawing_correspondences.py
│
└── util/                    # Shared autonomy components
    ├── predictors.py        # Bayesian, MaxEnt, CRF inference
    ├── SA_types.py          # Inference/Assistance/Arbitration enums
    ├── selector.py          # GUI selection interface
    └── shared_auto.py       # Assistance policy methods
```

## Shared Autonomy System

The dot simulator consists of three SA concepts:

### 1. Inference Methods
Predict user intent from observations:
- **Bayesian**: Probabilistic belief updates
- **Maximum Entropy**: Model uncertainty in user goals
- **CRF (Conditional Random Field)**: Sequence-based prediction

### 2. Assistance Methods
Generate robot actions based on inferred intent:
- **Distribution**: Blend actions weighted by goal probabilities

### 3. Arbitration Methods
Combine human and robot inputs:
- **Linear**: Fixed blend ratio (`γ` parameter)
- **Probabilistic**: Weight by robot confidence
- **Only User**: Pure human control (no assistance)

<!-- ## Examples -->

<!-- ### Custom Environment Wrappers

The `franka_test.py` demonstrates custom Gym wrappers:

```python
# Constrain movement to XY plane only
env = XYOnlyActionWrapper(env, z_index=2)

# Add goal-reaching reward structure
env = GoToPointWrapper(env, target=[2.0, 2.0])
``` -->

<!-- ## Analyzing Trained Policies

Inspect Q-table values using `read_npy.py`:

```python
from read_npy import read

q_table = read("q_table_orbit.npy")
print(q_table[5])  # View Q-values for state 5
``` -->

## Requirements

```
gymnasium
stable-baselines3
numpy
pygame
mujoco
```

Install with:
```bash
pip install gymnasium stable-baselines3 numpy pygame mujoco
```

## Resources

- **MuJoCo Environments**: [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
- **Robot Models**: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/)
- **Training Guide**: [Gymnasium Agent Training](https://gymnasium.farama.org/introduction/train_agent/)
- **MuJoCo Tutorial**: [tayalmanan28/MuJoCo-Tutorial](https://github.com/tayalmanan28/MuJoCo-Tutorial/blob/main/tutorial/tutorial_2.ipynb)

## Configuration

### Black Box Settings
- `MODEL_PATH`: Location to save/load trained models
- `start_pos`: Initial joint angles for evaluation (radians)
- `render`: Visualization mode (`"human"`, `"rgb_array"`, `"ansi"`)

### Dot Simulator Settings
Modify defaults in `Dot_Simulator.DEFAULTS`:
- `GAMMA`: Arbitration blend ratio (0.0-1.0)
- `GRID_SIZE`: Discretization resolution
- `DOT_SPEED`: Movement speed per frame

## Adding New Policies

To add new policies to the shared autonomy system:
1. Train a Q-table using the training utilities
2. Save as `.npy` in `trained_policies/`
3. Add visualization mapping in `policy_drawing_correspondences.py` (each entry contains the name of a Q-table file and the corresponding visualization of that policy, like a circle/square/etc.)


<!-- ## Getting Started
- Use `black_box.py` to train and run any of the MuJoCo environments.

- `dot_simulator_SA.py` creates a simple simulation environment in which you control a dot with arrow keys. This is a shared autonomy system with multiple trained policies.

    - `trained_policies/` contains all the trained policies used in the SA system
    - `training/` contains all the code for training the policies 

- `util.py` contains the utilities needed for `dot_simulator_SA.py`
    - `predictors.py`: Techniques for the inference step
    - `SA_types.py`: Enums of the options for SA inference, assistance, and arbitration steps
    - `selector.py`: Creates a gui which lets users select an item from a list
    - `shared_auto.py`: Assistance methods
 



## Resources 
Available robot models: https://github.com/google-deepmind/mujoco_menagerie/ 

Training agent: https://gymnasium.farama.org/introduction/train_agent/ 

mujoco tutorial: https://github.com/tayalmanan28/MuJoCo-Tutorial/blob/main/tutorial/tutorial_2.ipynb 

https://www.gymlibrary.dev/environments/mujoco/  -->

